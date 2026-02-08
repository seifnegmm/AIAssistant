"""Health check endpoint for infrastructure monitoring."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
import chromadb

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def _check_mongodb(settings: Settings) -> dict[str, Any]:
    """Ping MongoDB and return status."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        client: AsyncIOMotorClient = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=2000,
        )
        await client.admin.command("ping")
        client.close()
        return {"status": "healthy"}
    except Exception as exc:
        logger.warning("MongoDB health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


async def _check_chromadb(settings: Settings) -> dict[str, Any]:
    """Heartbeat ChromaDB and return status."""
    try:
        client = chromadb.HttpClient(
            host=settings.chromadb_host,
            port=settings.chromadb_port,
        )
        heartbeat = client.heartbeat()
        return {"status": "healthy", "heartbeat": heartbeat}
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


@router.get("")
async def health_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return aggregate health of all backend services."""
    mongo_status = await _check_mongodb(settings)
    chroma_status = await _check_chromadb(settings)

    services = {
        "mongodb": mongo_status,
        "chromadb": chroma_status,
    }

    overall = (
        "healthy"
        if all(s["status"] == "healthy" for s in services.values())
        else "degraded"
    )

    return {
        "status": overall,
        "services": services,
    }
