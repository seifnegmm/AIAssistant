"""Language detection utility using Lingua for accurate short-text detection.

Hybrid approach:
1. Fast Arabic script detection via Unicode range (U+0600 to U+06FF)
2. Lingua-py for statistical language detection (handles code-switching)
3. Session-based caching to avoid repeated detection
"""

import logging
import re
from typing import Literal
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

# Precompile Arabic Unicode range regex for speed
ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")

# Language type for type safety
LanguageCode = Literal["en", "ar"]


class LanguageDetector:
    """Detects language from text using hybrid approach."""

    def __init__(self) -> None:
        """Initialize Lingua detector with English and Arabic only."""
        # Build detector with only English and Arabic for speed
        self._detector = (
            LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.ARABIC)
            .with_preloaded_language_models()
            .build()
        )
        self._session_cache: dict[str, LanguageCode] = {}
        logger.info("LanguageDetector initialized with Lingua (en, ar)")

    def detect(self, text: str, session_id: str | None = None) -> LanguageCode:
        """Detect language from text with optional session caching.

        Args:
            text: Text to analyze (user message)
            session_id: Optional session ID for caching language preference

        Returns:
            Language code: "en" or "ar"
        """
        if not text or not text.strip():
            return "en"  # Default to English for empty input

        # Check session cache first
        if session_id and session_id in self._session_cache:
            cached_lang = self._session_cache[session_id]
            logger.debug(
                f"Using cached language for session {session_id}: {cached_lang}"
            )
            return cached_lang

        # Fast path: Check for Arabic script
        if ARABIC_SCRIPT_PATTERN.search(text):
            detected = "ar"
            logger.debug(f"Arabic script detected: {text[:50]}...")
        else:
            # Statistical detection using Lingua
            detected_language = self._detector.detect_language_of(text)
            if detected_language == Language.ARABIC:
                detected = "ar"
            else:
                detected = "en"
            logger.debug(f"Lingua detected: {detected} for text: {text[:50]}...")

        # Cache result for this session
        if session_id:
            self._session_cache[session_id] = detected
            logger.info(f"Cached language {detected} for session {session_id}")

        return detected

    def update_session_language(self, session_id: str, language: LanguageCode) -> None:
        """Manually update cached language for a session (e.g., from user preference).

        Args:
            session_id: Session identifier
            language: Language code to cache
        """
        self._session_cache[session_id] = language
        logger.info(f"Manually set language {language} for session {session_id}")

    def clear_session_cache(self, session_id: str) -> None:
        """Clear cached language for a session.

        Args:
            session_id: Session identifier to clear
        """
        if session_id in self._session_cache:
            del self._session_cache[session_id]
            logger.debug(f"Cleared language cache for session {session_id}")


# Global singleton instance
_detector_instance: LanguageDetector | None = None


def get_language_detector() -> LanguageDetector:
    """Get global LanguageDetector singleton instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector()
    return _detector_instance
