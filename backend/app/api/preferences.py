"""User preference management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def list_preferences(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return all learned user preferences from ChromaDB."""
    try:
        import chromadb

        client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        collection = client.get_or_create_collection("user_preferences")
        results = collection.get(limit=100, include=["documents", "metadatas"])

        preferences = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                preferences.append(
                    {
                        "id": results["ids"][i],
                        "content": doc,
                        "metadata": meta,
                    }
                )

        return {"preferences": preferences, "count": len(preferences)}
    except Exception as exc:
        logger.error("Failed to fetch preferences: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("")
async def clear_preferences(
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Clear all learned user preferences."""
    try:
        import chromadb

        client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        client.delete_collection("user_preferences")
        return {"status": "cleared"}
    except Exception as exc:
        logger.error("Failed to clear preferences: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
