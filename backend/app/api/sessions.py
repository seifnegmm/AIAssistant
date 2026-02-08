"""Session management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import Settings, get_settings
from app.models.sessions import SessionSummary

logger = logging.getLogger(__name__)
router = APIRouter()

COLLECTION = "chat_sessions"


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    settings: Settings = Depends(get_settings),
) -> list[dict[str, Any]]:
    """Return all chat sessions with metadata.

    Each document in ``chat_sessions`` has the shape::

        { session_id, messages: [...], created_at, updated_at }
    """
    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    collection = db[COLLECTION]

    cursor = (
        collection.find(
            {},
            {
                "session_id": 1,
                "messages": 1,
                "updated_at": 1,
                "_id": 0,
            },
        )
        .sort("updated_at", -1)
        .limit(50)
    )

    sessions = []
    async for doc in cursor:
        messages = doc.get("messages", [])
        sessions.append(
            {
                "session_id": doc["session_id"],
                "message_count": len(messages),
            }
        )

    client.close()
    return sessions


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return the full message history for a session.

    Returns the messages array from the single-document-per-session schema.
    Each message has: role, content, timestamp, and optionally tool_calls.
    """
    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    collection = db[COLLECTION]

    doc = await collection.find_one(
        {"session_id": session_id},
        {"messages": 1, "_id": 0},
    )
    client.close()

    if doc is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": doc.get("messages", []),
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Delete a chat session and its history."""
    client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    result = await db[COLLECTION].delete_one({"session_id": session_id})
    client.close()

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}
