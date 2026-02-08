"""Session models for conversation management."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Session(BaseModel):
    """Conversation session metadata."""

    session_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0


class SessionSummary(BaseModel):
    """Summary of a session for list views."""

    session_id: str
    message_count: int = 0


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[Session]
    total: int
