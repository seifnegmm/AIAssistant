"""Custom MongoDB chat history store — single document per session.

Replaces LangChain's ``MongoDBChatMessageHistory`` (one doc per message)
with a clean, human-readable schema that stores all messages for a
session in a single MongoDB document.

Document schema::

    {
        "session_id": "abc-123",
        "created_at": "2026-02-08T10:30:00Z",
        "updated_at": "2026-02-08T11:00:00Z",
        "messages": [
            {
                "role": "user",
                "content": "Hello!",
                "timestamp": "2026-02-08T10:30:00Z"
            },
            {
                "role": "assistant",
                "content": "Hey there! I'm Hope...",
                "timestamp": "2026-02-08T10:30:02Z",
                "tool_calls": [
                    {
                        "name": "recall_memory",
                        "args": {"query": "user preferences"},
                        "result": "..."
                    }
                ]
            }
        ]
    }

Implements ``BaseChatMessageHistory`` so LangGraph/LangChain can use it
as a drop-in replacement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pymongo import MongoClient
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chat_sessions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_to_clean_dict(msg: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a clean, readable dict.

    For AIMessages the enriched ``tool_calls_with_results`` list (set by
    ``graph.py`` via ``additional_kwargs``) takes priority over the raw
    ``tool_calls`` property so that each tool call already carries its
    ``result``.  Standalone ``ToolMessage`` entries are kept as a safety
    net but should no longer appear in normal operation.
    """
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }

    entry: dict[str, Any] = {
        "role": role_map.get(msg.type, msg.type),
        "content": msg.content,
        "timestamp": _now_iso(),
    }

    # Capture tool calls from AI messages — prefer enriched list with results
    if isinstance(msg, AIMessage):
        enriched = (msg.additional_kwargs or {}).get("tool_calls_with_results")
        if enriched:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "result": tc.get("result"),
                }
                for tc in enriched
            ]
        elif msg.tool_calls:
            # Fallback: raw tool_calls without results (non-streaming path)
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc["name"],
                    "args": tc["args"],
                }
                for tc in msg.tool_calls
            ]

    # Safety net: standalone ToolMessage (only from non-streaming ``chat()``)
    if isinstance(msg, ToolMessage):
        entry["tool_call_id"] = msg.tool_call_id
        entry["tool_name"] = msg.name or ""

    return entry


def _clean_dict_to_message(entry: dict[str, Any]) -> BaseMessage:
    """Reconstruct a LangChain message from our clean dict.

    For ``assistant`` messages that carry ``tool_calls`` (each with an
    optional ``result``), we rebuild the LangChain ``AIMessage`` with a
    plain ``tool_calls`` list (LangChain doesn't know about ``result``)
    *and* stash the enriched list in ``additional_kwargs`` so callers
    that care can still access it.
    """
    role = entry.get("role", "user")
    content = entry.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        kwargs: dict[str, Any] = {}
        raw_tcs = entry.get("tool_calls")
        if raw_tcs:
            kwargs["additional_kwargs"] = {
                "tool_calls_with_results": raw_tcs,
            }
        return AIMessage(content=content, **kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=entry.get("tool_call_id", ""),
            name=entry.get("tool_name", ""),
        )
    else:
        return HumanMessage(content=content)


class ChatHistoryStore(BaseChatMessageHistory):
    """MongoDB chat history — one document per session, embedded message array.

    Compatible with LangChain's ``BaseChatMessageHistory`` interface so it
    works as a drop-in replacement for ``MongoDBChatMessageHistory``.

    Uses **pymongo** (synchronous) because LangChain's ``BaseChatMessageHistory``
    defines ``messages`` as a synchronous property.
    """

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.session_id = session_id
        self.database_name = database_name
        self.collection_name = collection_name

        self._client = MongoClient(connection_string)
        self._db = self._client[database_name]
        self._collection: Collection = self._db[collection_name]

        # Ensure index on session_id for fast lookups
        self._collection.create_index("session_id", unique=True, sparse=True)

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:
        """Load all messages for this session from MongoDB."""
        doc = self._collection.find_one({"session_id": self.session_id})
        if not doc or not doc.get("messages"):
            return []
        return [_clean_dict_to_message(entry) for entry in doc["messages"]]

    def add_message(self, message: BaseMessage) -> None:
        """Append a single message to this session's document (upsert)."""
        entry = _message_to_clean_dict(message)
        now = _now_iso()

        self._collection.update_one(
            {"session_id": self.session_id},
            {
                "$push": {"messages": entry},
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "session_id": self.session_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append multiple messages in a single MongoDB operation."""
        if not messages:
            return
        entries = [_message_to_clean_dict(m) for m in messages]
        now = _now_iso()

        self._collection.update_one(
            {"session_id": self.session_id},
            {
                "$push": {"messages": {"$each": entries}},
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "session_id": self.session_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    def clear(self) -> None:
        """Remove all messages for this session."""
        self._collection.delete_one({"session_id": self.session_id})

    # ------------------------------------------------------------------
    # Extra helpers (not part of BaseChatMessageHistory)
    # ------------------------------------------------------------------

    def get_clean_history(self) -> list[dict[str, Any]]:
        """Return the raw clean message dicts (for REST API responses)."""
        doc = self._collection.find_one({"session_id": self.session_id})
        if not doc or not doc.get("messages"):
            return []
        return doc["messages"]

    def get_session_doc(self) -> dict[str, Any] | None:
        """Return the full session document."""
        doc = self._collection.find_one(
            {"session_id": self.session_id},
            {"_id": 0},
        )
        return doc
