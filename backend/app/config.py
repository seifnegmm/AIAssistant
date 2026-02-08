"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "AI Assistant"
    environment: str = "development"
    log_level: str = "debug"
    debug: bool = True

    # Google AI
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"

    # Web Search
    tavily_api_key: str = ""

    # MongoDB
    mongodb_uri: str = "mongodb://mongodb:27017"
    mongodb_database: str = "ai_assistant"

    # ChromaDB
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000

    # Voice
    tts_voice: str = "en-US-Chirp3-HD-Leda"
    audio_sample_rate: int = 16000

    # Simli Avatar
    simli_api_key: str = ""
    simli_face_id: str = ""

    # CORS
    frontend_url: str = "http://localhost:3000"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


settings = Settings()


def get_settings() -> Settings:
    """FastAPI dependency for injecting settings."""
    return settings
