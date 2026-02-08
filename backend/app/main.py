"""FastAPI application entry point with lifespan management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import websocket_chat
from app.api.router import api_router
from app.config import settings
from app.dependencies import get_memory_manager, get_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("Starting AI Assistant backend...")

    # Initialize memory manager (connects to MongoDB + ChromaDB)
    memory_manager = get_memory_manager()
    await memory_manager.initialize()
    logger.info("Memory manager initialized successfully")

    # Initialize scheduler (for scheduled Telegram messages)
    scheduler = get_scheduler()
    await scheduler.initialize()
    logger.info("Scheduler initialized successfully")

    yield

    # Cleanup
    await scheduler.shutdown()
    await memory_manager.close()
    logger.info("AI Assistant backend shut down cleanly")


app = FastAPI(
    title="AI Assistant API",
    description="Cloud-based AI assistant with speech, visuals, and learning capabilities",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

# Mount WebSocket endpoint (outside /api prefix to match frontend expectations)
app.websocket("/ws/chat")(websocket_chat)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
