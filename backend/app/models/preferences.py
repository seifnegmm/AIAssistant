"""User preference models for the learning system."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Preference(BaseModel):
    """A learned user preference."""

    key: str
    value: str
    category: str = "general"
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    source: str = "inferred"
    learned_at: datetime = Field(default_factory=datetime.utcnow)


class PreferenceListResponse(BaseModel):
    """Response for listing preferences."""

    preferences: list[Preference]
    total: int
