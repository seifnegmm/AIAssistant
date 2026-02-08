"""Google Cloud Text-to-Speech via REST API with API key authentication.

Uses the REST endpoint directly to avoid service account requirements.
Outputs raw PCM16 (16kHz, mono, 16-bit signed LE) suitable for Simli avatar.
"""

import logging
from typing import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Google Cloud TTS V1 REST endpoint
TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# WAV RIFF header magic bytes — used to detect if response contains a WAV header
WAV_RIFF_MAGIC = b"RIFF"
WAV_HEADER_SIZE = 44


class TextToSpeech:
    """Text-to-Speech using Google Cloud TTS REST API.

    Produces raw PCM16 audio (no WAV header) at 16kHz mono,
    compatible with Simli avatar's audio input requirements.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._voice_name = settings.tts_voice
        self._sample_rate = settings.audio_sample_rate

    async def initialize(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            "TextToSpeech initialized (REST API, voice=%s, sample_rate=%d)",
            self._voice_name,
            self._sample_rate,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("TextToSpeech closed")

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """Synthesize text to raw PCM16 audio with language support.

        Args:
            text: Text to convert to speech.
            language: Language code ("en" or "ar"). Defaults to "en".

        Returns:
            Raw PCM16 audio bytes (16kHz, mono, 16-bit signed LE).
            WAV header is stripped automatically.
        """
        if not self._client:
            raise RuntimeError("TextToSpeech not initialized. Call initialize() first.")

        if not text or not text.strip():
            return b""

        # Select voice and language code based on detected language
        if language == "ar":
            language_code = "ar-XA"
            voice_name = settings.tts_voice_ar
        else:
            language_code = "en-US"
            voice_name = self._voice_name

        request_body = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": self._sample_rate,
                "speakingRate": 1.0,
            },
        }

        try:
            response = await self._client.post(
                TTS_URL,
                params={"key": settings.google_api_key},
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            import base64

            audio_content = base64.b64decode(data["audioContent"])

            # Only strip WAV header if one is actually present (RIFF magic)
            if (
                audio_content[:4] == WAV_RIFF_MAGIC
                and len(audio_content) > WAV_HEADER_SIZE
            ):
                raw_pcm = audio_content[WAV_HEADER_SIZE:]
            else:
                raw_pcm = audio_content

            duration_ms = (len(raw_pcm) / (self._sample_rate * 2)) * 1000
            logger.info(
                "TTS synthesized %d bytes of PCM16 (%.0fms) for text: '%s'",
                len(raw_pcm),
                duration_ms,
                text[:60],
            )
            return raw_pcm

        except httpx.HTTPStatusError as e:
            logger.error(
                "TTS API error %d: %s", e.response.status_code, e.response.text[:200]
            )
            raise
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            raise

    async def synthesize_chunked(
        self, text: str, language: str = "en", max_chars: int = 200
    ) -> AsyncIterator[bytes]:
        """Synthesize long text in chunks for streaming playback.

        Splits text at sentence boundaries and yields PCM16 audio
        for each chunk, enabling progressive audio playback.

        Args:
            text: Full text to synthesize.
            language: Language code ("en" or "ar"). Defaults to "en".
            max_chars: Maximum characters per chunk.

        Yields:
            Raw PCM16 audio bytes per sentence/chunk.
        """
        chunks = self._split_text(text, max_chars)

        for chunk in chunks:
            audio = await self.synthesize(chunk, language=language)
            if audio:
                yield audio

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        """Split text at sentence boundaries respecting max_chars.

        Tries to split at '.', '!', '?' boundaries. Falls back to
        splitting at spaces if sentences are too long.
        """
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_chars:
                chunks.append(remaining)
                break

            # Try to find a sentence boundary within max_chars
            split_pos = -1
            for delim in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = remaining.rfind(delim, 0, max_chars)
                if pos > split_pos:
                    split_pos = pos + len(delim) - 1  # Include the punctuation

            # Fall back to splitting at a space
            if split_pos <= 0:
                split_pos = remaining.rfind(" ", 0, max_chars)

            # Last resort: hard split
            if split_pos <= 0:
                split_pos = max_chars

            chunk = remaining[: split_pos + 1].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_pos + 1 :].strip()

        return chunks
