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

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe a complete audio chunk to text.

        Args:
            audio_bytes: Raw PCM16 audio (16kHz, mono, 16-bit signed LE).

        Returns:
            Transcribed text string, or empty string if nothing recognized.
        """
        if not self._client:
            raise RuntimeError("StreamingSTT not initialized. Call initialize() first.")

        if not audio_bytes or len(audio_bytes) < 100:
            return ""

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        request_body = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self._sample_rate,
                "languageCode": "en-US",
                "model": "latest_long",
                "enableAutomaticPunctuation": True,
            },
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
                return ""

            transcript = results[0].get("alternatives", [{}])[0].get("transcript", "")
            confidence = results[0].get("alternatives", [{}])[0].get("confidence", 0)
            logger.info(
                "STT transcribed: '%s' (confidence=%.2f)", transcript[:80], confidence
            )
            return transcript

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
    ) -> AsyncIterator[str]:
        """Process an audio stream in chunks and yield partial transcripts.

        Accumulates audio data and transcribes every `chunk_duration_ms` worth of audio.
        This simulates streaming by doing repeated single-shot recognition.

        Args:
            audio_stream: Async iterator of raw PCM16 audio chunks.
            chunk_duration_ms: How often to transcribe (in milliseconds).

        Yields:
            Partial transcript strings.
        """
        bytes_per_ms = (self._sample_rate * 2) // 1000  # 16-bit = 2 bytes per sample
        chunk_size = bytes_per_ms * chunk_duration_ms
        buffer = bytearray()

        async for chunk in audio_stream:
            buffer.extend(chunk)

            if len(buffer) >= chunk_size:
                transcript = await self.transcribe(bytes(buffer))
                buffer.clear()
                if transcript:
                    yield transcript

        # Transcribe any remaining audio
        if len(buffer) > 100:
            transcript = await self.transcribe(bytes(buffer))
            if transcript:
                yield transcript
