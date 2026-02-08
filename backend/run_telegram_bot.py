"""Standalone script to run the Telegram bot.

Run this script separately from the main FastAPI server:
    python -m app.telegram.run_bot

Or simply:
    python backend/run_telegram_bot.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.dependencies import get_memory_manager
from app.telegram.bot import get_bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("telegram_bot.log"),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    """Run the Telegram bot."""
    logger.info("=" * 50)
    logger.info("Starting Hope Telegram Bot...")
    logger.info("=" * 50)

    memory_manager = None
    try:
        # Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        logger.info("Memory manager initialized successfully")

        # Start bot
        bot = get_bot()
        await bot.start_polling()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down...")
        # Cleanup
        if memory_manager:
            await memory_manager.close()
        logger.info("Bot shut down cleanly")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if memory_manager:
            await memory_manager.close()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
