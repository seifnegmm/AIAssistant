"""Google Cloud Speech-to-Text V2 via REST API with API key authentication.

Uses the REST endpoint directly to avoid service account requirements.
Audio format: LINEAR16/PCM16, 16kHz, mono.
"""

import base64
import logging
from typing import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Google Cloud STT V2 REST endpoint
STT_V2_URL = (
    "https://speech.googleapis.com/v2/projects/-/locations/global/recognizers/_"
)

# V1 REST endpoint (more stable for single-shot recognition with API key)
STT_V1_URL = "https://speech.googleapis.com/v1/speech:recognize"


class StreamingSTT:
    """Speech-to-Text using Google Cloud Speech REST API.

    Uses the V1 REST API with API key for single-shot recognition
    and chunked processing for streaming-like behavior.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._sample_rate = settings.audio_sample_rate

    async def initialize(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            "StreamingSTT initialized (REST API, sample_rate=%d)", self._sample_rate
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("StreamingSTT closed")

    async def transcribe(
        self, audio_bytes: bytes, language: str = "en", auto_detect: bool = False
    ) -> tuple[str, str]:
        """Transcribe a complete audio chunk to text.

        Args:
            audio_bytes: Raw PCM16 audio (16kHz, mono, 16-bit signed LE).
            language: Language code ('en' for English, 'ar' for Arabic).
            auto_detect: If True, use multi-language detection (en-US, ar-SA).

        Returns:
            Tuple of (transcribed_text, detected_language_code).
            Language code will be 'en' or 'ar' based on detection.
        """
        if not self._client:
            raise RuntimeError("StreamingSTT not initialized. Call initialize() first.")

        if not audio_bytes or len(audio_bytes) < 100:
            return "", "en"

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        if auto_detect:
            language_code = "en-US"
            alternative_language_codes = ["ar-SA"]
        else:
            language_code = "ar-SA" if language == "ar" else "en-US"
            alternative_language_codes = []

        config = {
            "encoding": "LINEAR16",
            "sampleRateHertz": self._sample_rate,
            "languageCode": language_code,
            "model": "latest_long",
            "enableAutomaticPunctuation": True,
        }

        if alternative_language_codes:
            config["alternativeLanguageCodes"] = alternative_language_codes

        request_body = {
            "config": config,
            "audio": {
                "content": audio_b64,
            },
        }

        try:
            response = await self._client.post(
                STT_V1_URL,
                params={"key": settings.google_api_key},
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.debug("STT returned no results")
                return "", language

            transcript = results[0].get("alternatives", [{}])[0].get("transcript", "")
            confidence = results[0].get("alternatives", [{}])[0].get("confidence", 0)
            detected_lang_code = results[0].get("languageCode", language_code)

            detected_lang = "ar" if detected_lang_code.startswith("ar") else "en"

            logger.info(
                "STT transcribed: '%s' (confidence=%.2f, language=%s)",
                transcript[:80],
                confidence,
                detected_lang,
            )
            return transcript, detected_lang

        except httpx.HTTPStatusError as e:
            logger.error(
                "STT API error %d: %s", e.response.status_code, e.response.text[:200]
            )
            raise
        except Exception as e:
            logger.error("STT transcription failed: %s", e)
            raise

    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes], chunk_duration_ms: int = 3000
    ) -> AsyncIterator[tuple[str, str]]:
        """Process an audio stream in chunks and yield partial transcripts.

        Accumulates audio data and transcribes every `chunk_duration_ms` worth of audio.
        This simulates streaming by doing repeated single-shot recognition with auto-detect.

        Args:
            audio_stream: Async iterator of raw PCM16 audio chunks.
            chunk_duration_ms: How often to transcribe (in milliseconds).

        Yields:
            Tuples of (transcript, detected_language).
        """
        bytes_per_ms = (self._sample_rate * 2) // 1000
        chunk_size = bytes_per_ms * chunk_duration_ms
        buffer = bytearray()

        async for chunk in audio_stream:
            buffer.extend(chunk)

            if len(buffer) >= chunk_size:
                transcript, lang = await self.transcribe(
                    bytes(buffer), auto_detect=True
                )
                buffer.clear()
                if transcript:
                    yield transcript, lang

        if len(buffer) > 100:
            transcript, lang = await self.transcribe(bytes(buffer), auto_detect=True)
            if transcript:
                yield transcript, lang
