"""
Persistent scheduler service using APScheduler with MongoDB job store.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.triggers.date import DateTrigger
from apscheduler.job import Job
from telegram import Bot
from telegram.error import TelegramError
from pymongo import MongoClient

from app.config import settings
from app.scheduler.time_parser import TimeParser

logger = logging.getLogger(__name__)


# Module-level function for APScheduler (must be serializable)
async def _send_telegram_message_job(bot_token: str, user_id: str, message: str):
    """
    Standalone function to send Telegram message.
    Used as scheduled job callback (must be module-level for serialization).

    Args:
        bot_token: Telegram bot token
        user_id: Telegram user ID to send message to
        message: Message text to send
    """
    try:
        bot = Bot(token=bot_token)
        await bot.send_message(chat_id=user_id, text=f"⏰ Reminder: {message}")
        logger.info(f"✅ Sent scheduled message to user {user_id}")
    except TelegramError as e:
        logger.error(f"Failed to send scheduled message to {user_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending scheduled message: {e}")
    finally:
        # Clean up bot instance
        if "bot" in locals():
            await bot.shutdown()


class SchedulerService:
    """
    Manages scheduled tasks with persistent storage in MongoDB.
    """

    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.telegram_bot: Optional[Bot] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the scheduler with MongoDB job store."""
        if self._initialized:
            logger.warning("Scheduler already initialized")
            return

        try:
            # Configure MongoDB job store (use synchronous client for APScheduler)
            mongo_client = MongoClient(settings.mongodb_uri)
            jobstores = {
                "default": MongoDBJobStore(
                    database=settings.mongodb_database,
                    collection="scheduled_jobs",
                    client=mongo_client,
                )
            }

            # Configure scheduler
            self.scheduler = AsyncIOScheduler(jobstores=jobstores)

            # Initialize Telegram bot for sending scheduled messages
            if settings.telegram_bot_token:
                self.telegram_bot = Bot(token=settings.telegram_bot_token)
            else:
                logger.warning(
                    "Telegram bot token not configured, scheduled messages won't work"
                )

            # Start the scheduler
            self.scheduler.start()
            self._initialized = True

            logger.info("✅ Scheduler initialized successfully")
            logger.info(f"Active jobs: {len(self.scheduler.get_jobs())}")

        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler shut down")

    async def schedule_telegram_message(
        self, user_id: str, message: str, when: str, job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule a Telegram message to be sent at a specific time.

        Args:
            user_id: Telegram user ID to send message to
            message: Message content
            when: Natural language time expression (e.g., "in 30 minutes", "tomorrow at 9am")
            job_id: Optional custom job ID (auto-generated if not provided)

        Returns:
            Dict with job info: {
                "job_id": str,
                "scheduled_time": datetime,
                "message": str,
                "formatted_time": str
            }

        Raises:
            ValueError: If time parsing fails or scheduler not initialized
        """
        if not self._initialized:
            raise ValueError("Scheduler not initialized. Call initialize() first.")

        if not self.telegram_bot:
            raise ValueError("Telegram bot not configured")

        # Parse the time expression
        scheduled_time = TimeParser.parse(when)
        if not scheduled_time:
            raise ValueError(f"Could not parse time expression: '{when}'")

        # Ensure time is in the future
        if scheduled_time <= datetime.now():
            raise ValueError(f"Scheduled time must be in the future: {scheduled_time}")

        # Generate job ID if not provided
        if not job_id:
            job_id = f"telegram_{user_id}_{int(scheduled_time.timestamp())}"

        # Schedule the job using module-level function (serializable)
        job = self.scheduler.add_job(
            func=_send_telegram_message_job,
            trigger=DateTrigger(run_date=scheduled_time),
            args=[settings.telegram_bot_token, user_id, message],
            id=job_id,
            name=f"Telegram message to {user_id}",
            replace_existing=True,  # Replace if job_id exists
        )

        formatted_time = TimeParser.format_scheduled_time(scheduled_time)

        logger.info(
            f"Scheduled message for user {user_id} at {scheduled_time} (job_id: {job_id})"
        )

        return {
            "job_id": job.id,
            "scheduled_time": scheduled_time,
            "message": message,
            "formatted_time": formatted_time,
            "user_id": user_id,
        }

    async def _send_telegram_message(self, user_id: str, message: str):
        """
        Internal method to send a Telegram message.
        Called by the scheduler at the scheduled time.
        """
        try:
            await self.telegram_bot.send_message(chat_id=user_id, text=message)
            logger.info(f"✅ Sent scheduled message to user {user_id}")
        except TelegramError as e:
            logger.error(f"Failed to send scheduled message to {user_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending scheduled message: {e}")

    def get_scheduled_jobs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of scheduled jobs, optionally filtered by user ID.

        Args:
            user_id: Optional user ID to filter jobs

        Returns:
            List of job dicts with: job_id, scheduled_time, message, formatted_time
        """
        if not self._initialized:
            return []

        jobs = self.scheduler.get_jobs()

        result = []
        for job in jobs:
            # Extract user_id from args if available
            job_user_id = job.args[0] if job.args else None

            # Filter by user_id if specified
            if user_id and job_user_id != user_id:
                continue

            # Get message from args
            message = job.args[1] if len(job.args) > 1 else "N/A"

            # Get scheduled time
            scheduled_time = job.next_run_time

            result.append(
                {
                    "job_id": job.id,
                    "user_id": job_user_id,
                    "scheduled_time": scheduled_time,
                    "formatted_time": TimeParser.format_scheduled_time(scheduled_time),
                    "message": message[:50] + "..." if len(message) > 50 else message,
                }
            )

        # Sort by scheduled time
        result.sort(key=lambda x: x["scheduled_time"])

        return result

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a scheduled job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled, False if not found
        """
        if not self._initialized:
            return False

        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel job {job_id}: {e}")
            return False

    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific job."""
        if not self._initialized:
            return None

        try:
            job: Job = self.scheduler.get_job(job_id)
            if not job:
                return None

            return {
                "job_id": job.id,
                "user_id": job.args[0] if job.args else None,
                "message": job.args[1] if len(job.args) > 1 else "N/A",
                "scheduled_time": job.next_run_time,
                "formatted_time": TimeParser.format_scheduled_time(job.next_run_time),
                "name": job.name,
            }
        except Exception as e:
            logger.error(f"Error getting job info: {e}")
            return None
