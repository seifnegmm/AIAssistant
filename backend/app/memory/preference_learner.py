"""Preference Learner - Extracts user preferences from conversation history.

Uses Gemini at low temperature to identify and store user preferences
in ChromaDB's user_preferences collection via MemoryManager.
"""

import json
import logging
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a preference extraction system. Analyze the following conversation
and extract any user preferences, habits, or personal information the user has shared.

Return a JSON array of preferences. Each preference should have:
- "key": a short identifier (e.g., "favorite_color", "preferred_language", "dietary_restriction")
- "value": the preference value (e.g., "blue", "Python", "vegetarian")
- "category": one of ["communication", "technical", "personal", "workflow", "dietary", "general"]
- "confidence": a float 0.0-1.0 indicating how confident you are this is a real preference

Rules:
- Only extract EXPLICIT preferences stated by the user (not inferences)
- Minimum confidence threshold: 0.7
- Do NOT extract temporary requests or one-time instructions
- Focus on lasting preferences that should be remembered
- Return an empty array [] if no preferences are found

Return ONLY the JSON array, no other text."""


class PreferenceLearner:
    """Extracts user preferences from conversations using Gemini."""

    def __init__(self, memory_manager=None) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        self.min_confidence = 0.7
        self._memory_manager = memory_manager

    async def learn_from_exchange(
        self,
        user_message: str,
        ai_response: str,
        memory_manager=None,
    ) -> list[dict]:
        """Convenience wrapper: analyze a single user/assistant exchange."""
        mm = memory_manager or self._memory_manager
        if mm is None:
            raise RuntimeError("No memory_manager provided")
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response},
        ]
        return await self.analyze_conversation(messages, mm)

    async def analyze_conversation(
        self,
        messages: list[dict],
        memory_manager=None,
    ) -> list[dict]:
        """Analyze recent messages and extract preferences.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            memory_manager: MemoryManager instance for storing preferences.

        Returns:
            List of extracted preference dicts that were stored.
        """
        mm = memory_manager or self._memory_manager
        if mm is None:
            raise RuntimeError("No memory_manager provided")

        if not messages:
            return []

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        try:
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=EXTRACTION_PROMPT),
                    HumanMessage(content=conversation_text),
                ]
            )

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            preferences = json.loads(raw)
            if not isinstance(preferences, list):
                return []

            stored = []
            for pref in preferences:
                confidence = float(pref.get("confidence", 0))
                if confidence < self.min_confidence:
                    continue

                key = pref.get("key", "")
                value = pref.get("value", "")
                category = pref.get("category", "general")

                if not key or not value:
                    continue

                await mm.save_preference(
                    key=key,
                    value=value,
                    category=category,
                    confidence=confidence,
                    source="inferred",
                )
                stored.append(pref)
                logger.info(
                    "Learned preference: %s = %s (%.1f)", key, value, confidence
                )

            return stored

        except json.JSONDecodeError:
            logger.warning("Failed to parse preference extraction response")
            return []
        except Exception:
            logger.exception("Preference learning failed")
            return []
