"""Message models for WebSocket and API communication."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message sender role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """WebSocket message type discriminator."""

    TEXT = "text"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    STATUS = "status"
    ERROR = "error"


class IncomingMessage(BaseModel):
    """Message received from client via WebSocket."""

    type: MessageType
    content: Optional[str] = None
    audio_data: Optional[str] = None  # base64-encoded PCM16 16kHz

    def is_text(self) -> bool:
        return self.type == MessageType.TEXT and self.content is not None

    def is_audio(self) -> bool:
        return self.type == MessageType.AUDIO and self.audio_data is not None


class OutgoingMessage(BaseModel):
    """Message sent to client via WebSocket."""

    type: MessageType
    content: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    """Persisted chat message for history."""

    id: UUID = Field(default_factory=uuid4)
    session_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
