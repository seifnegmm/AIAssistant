"""Tests for health check endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Health endpoint returns 200 with service status."""
    response = await client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "services" in data
    assert "mongodb" in data["services"]
    assert "chromadb" in data["services"]
