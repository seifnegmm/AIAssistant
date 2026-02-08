"""Dependency injection providers for FastAPI."""

from app.memory.manager import MemoryManager
from app.scheduler.service import SchedulerService

# Global singleton instances (thread-safe for async contexts)
_memory_manager: MemoryManager | None = None
_scheduler: SchedulerService | None = None


def get_memory_manager() -> MemoryManager:
    """Return singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_scheduler() -> SchedulerService:
    """Return singleton SchedulerService instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SchedulerService()
    return _scheduler
