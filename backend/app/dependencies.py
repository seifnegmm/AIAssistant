"""Dependency injection providers for FastAPI."""

from functools import lru_cache

from app.memory.manager import MemoryManager


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """Return singleton MemoryManager instance."""
    return MemoryManager()
