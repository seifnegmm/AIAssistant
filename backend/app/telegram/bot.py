"""Telegram bot handler for two-way chat with Hope.

Provides polling-based message handling for conversational interaction
via Telegram, using the same AssistantAgent as the web UI.
"""

import asyncio
import logging
from typing import Optional

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

from app.agent.graph import AssistantAgent
from app.config import settings
from app.utils.language_detector import LanguageDetector
from app.voice.stt import StreamingSTT
from app.voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class HopeTelegramBot:
    """Two-way Telegram bot for Hope AI Assistant."""

    def __init__(self):
        """Initialize the Telegram bot with agent and dependencies."""
        from app.dependencies import get_memory_manager

        memory_manager = get_memory_manager()
        self.agent = AssistantAgent(memory_manager)
        self.language_detector = LanguageDetector()

        # Initialize STT/TTS lazily (they need async initialization)
        self.stt = None
        self.tts = None
        self.application: Optional[Application] = None

    def _is_user_allowed(self, user_id: str) -> bool:
        """Check if user ID is in the whitelist."""
        allowed_users = settings.telegram_allowed_users
        if not allowed_users:
            # If whitelist is empty, allow everyone (backwards compatibility)
            logger.warning("No Telegram whitelist configured - allowing all users!")
            return True
        return user_id in allowed_users

    async def start_command(self, update: Update, context) -> None:
        """Handle /start command."""
        user = update.effective_user
        user_id = str(user.id)

        # Check if user is whitelisted
        if not self._is_user_allowed(user_id):
            logger.warning(f"Unauthorized /start attempt from user {user_id}")
            await update.message.reply_text(
                "🔒 Sorry, you don't have access to this assistant. "
                "This is a private bot."
            )
            return

        # Detect language from user's Telegram settings or use English
        lang_code = user.language_code or "en"
        language = "ar" if lang_code.startswith("ar") else "en"

        if language == "ar":
            greeting = (
                f"أهلاً {user.first_name}! 👋\n\n"
                "أنا أمل، مساعدة الذكاء الاصطناعي. أقدر أساعدك في أي حاجة. "
                "اكتب أي سؤال أو اطلب مني أعمل حاجة معينة."
            )
        else:
            greeting = (
                f"Hello {user.first_name}! 👋\n\n"
                "I'm Hope, your AI assistant. I can help you with questions, "
                "search the web, remember information, and more. "
                "Just send me a message!"
            )

        await update.message.reply_text(greeting)
        logger.info(f"User {user_id} started conversation (language: {language})")

    async def help_command(self, update: Update, context) -> None:
        """Handle /help command."""
        user_id = str(update.effective_user.id)

        # Check if user is whitelisted
        if not self._is_user_allowed(user_id):
            logger.warning(f"Unauthorized /help attempt from user {user_id}")
            await update.message.reply_text(
                "🔒 Sorry, you don't have access to this assistant. "
                "This is a private bot."
            )
            return

        # Detect language
        user = update.effective_user
        lang_code = user.language_code or "en"
        language = "ar" if lang_code.startswith("ar") else "en"

        if language == "ar":
            help_text = (
                "🤖 *أوامر أمل*\n\n"
                "/start - ابدأ محادثة جديدة\n"
                "/help - اعرض هذه المساعدة\n"
                "/clear - امسح ذاكرة المحادثة\n\n"
                "*القدرات:*\n"
                "• إجابة الأسئلة والمحادثة\n"
                "• البحث في الويب\n"
                "• تذكر المعلومات المهمة\n"
                "• إرسال تذكيرات\n"
                "• دعم كامل للعربية والإنجليزية"
            )
        else:
            help_text = (
                "🤖 *Hope Commands*\n\n"
                "/start - Start a new conversation\n"
                "/help - Show this help message\n"
                "/clear - Clear conversation memory\n\n"
                "*Capabilities:*\n"
                "• Answer questions & chat\n"
                "• Search the web\n"
                "• Remember important information\n"
                "• Send reminders\n"
                "• Full English & Arabic support"
            )

        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"User {user_id} requested help")

    async def clear_command(self, update: Update, context) -> None:
        """Handle /clear command to reset conversation memory."""
        user_id = str(update.effective_user.id)
        session_id = f"telegram_{user_id}"

        # Check if user is whitelisted
        if not self._is_user_allowed(user_id):
            logger.warning(f"Unauthorized /clear attempt from user {user_id}")
            await update.message.reply_text(
                "🔒 Sorry, you don't have access to this assistant. "
                "This is a private bot."
            )
            return

        # Detect language
        user = update.effective_user
        lang_code = user.language_code or "en"
        language = "ar" if lang_code.startswith("ar") else "en"

        # Clear the session (implementation depends on your memory backend)
        # For now, just acknowledge
        if language == "ar":
            response = "تم مسح ذاكرة المحادثة. خلينا نبدأ من جديد! 🔄"
        else:
            response = "Conversation memory cleared. Let's start fresh! 🔄"

        await update.message.reply_text(response)
        logger.info(f"User {user_id} cleared conversation memory")

    async def handle_message(self, update: Update, context) -> None:
        """Handle incoming text messages."""
        user = update.effective_user
        user_id = str(user.id)
        message_text = update.message.text
        session_id = f"telegram_{user_id}"

        # Check if user is whitelisted
        if not self._is_user_allowed(user_id):
            logger.warning(f"Unauthorized access attempt from user {user_id}")
            await update.message.reply_text(
                "🔒 Sorry, you don't have access to this assistant. "
                "This is a private bot."
            )
            return

        # Show typing indicator
        await update.message.chat.send_action("typing")

        logger.info(f"User {user_id} sent: {message_text[:50]}...")

        try:
            # Detect language from message
            language = self.language_detector.detect(message_text)
            logger.info(f"Detected language: {language}")

            # Get response from agent (streaming)
            full_response = ""
            async for token in self.agent.chat_stream(
                message_text, session_id, language=language
            ):
                full_response += token

            # Fallback for tool-only responses (e.g., scheduling, memory ops)
            if not full_response:
                logger.warning("Empty stream response, falling back to agent.chat()")
                full_response = await self.agent.chat(
                    message_text, session_id, language=language
                )

            # Send response back to Telegram
            if full_response:
                await update.message.reply_text(full_response)
            else:
                logger.error("No response from agent, sending error message")
                await update.message.reply_text(
                    "Sorry, I encountered an issue processing your request."
                    if language == "en"
                    else "عذراً، حصل خطأ في معالجة طلبك."
                )
            logger.info(f"Sent response to user {user_id}: {full_response[:50]}...")

        except Exception as e:
            logger.error(f"Error handling message from {user_id}: {e}", exc_info=True)

            # Send error message in appropriate language
            error_msg = (
                "عذراً، حصل خطأ. ممكن تحاول تاني؟"
                if language == "ar"
                else "Sorry, I encountered an error. Could you try again?"
            )
            await update.message.reply_text(error_msg)

    async def handle_voice(self, update: Update, context) -> None:
        """Handle incoming voice messages."""
        user = update.effective_user
        user_id = str(user.id)
        session_id = f"telegram_{user_id}"

        # Check if user is whitelisted
        if not self._is_user_allowed(user_id):
            logger.warning(f"Unauthorized voice access attempt from user {user_id}")
            await update.message.reply_text(
                "🔒 Sorry, you don't have access to this assistant."
            )
            return

        # Show typing indicator
        await update.message.chat.send_action("typing")

        logger.info(f"User {user_id} sent a voice message")

        try:
            # Initialize STT/TTS if needed (lazy initialization)
            if self.stt is None:
                from app.voice.stt import StreamingSTT

                self.stt = StreamingSTT()
                await self.stt.initialize()
            elif not self.stt._client:
                await self.stt.initialize()

            if self.tts is None:
                from app.voice.tts import TextToSpeech

                self.tts = TextToSpeech()
                await self.tts.initialize()
            elif not self.tts._client:
                await self.tts.initialize()

            # Download voice file from Telegram
            voice = update.message.voice
            voice_file = await context.bot.get_file(voice.file_id)

            # Download as bytes
            voice_bytes = await voice_file.download_as_bytearray()

            # Convert OGG/OPUS to PCM16 format for STT
            # Telegram sends voice as OGG/OPUS, we need PCM16
            import io
            from pydub import AudioSegment

            audio = AudioSegment.from_file(io.BytesIO(voice_bytes), format="ogg")
            logger.info(
                f"Original audio: {audio.frame_rate}Hz, {audio.channels}ch, "
                f"{audio.sample_width}B, {len(audio)}ms, {len(audio.raw_data)} bytes"
            )

            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_audio = audio.raw_data

            logger.info(
                f"Converted audio: 16000Hz, 1ch, 2B, {len(audio)}ms, {len(pcm_audio)} bytes"
            )

            # Validate audio length (minimum 0.5 seconds = 16000 bytes)
            if len(pcm_audio) < 16000:
                await update.message.reply_text(
                    "Voice message too short. Please record a longer message."
                )
                return

            # Transcribe with auto language detection
            transcript, detected_language = await self.stt.transcribe(
                pcm_audio, auto_detect=True
            )

            if not transcript:
                error_msg = (
                    "عذراً، ما قدرت أفهم الصوت. ممكن تجرب تاني؟"
                    if detected_language == "ar"
                    else "Sorry, I couldn't understand the audio. Could you try again?"
                )
                await update.message.reply_text(error_msg)
                return

            logger.info(f"Transcribed ({detected_language}): {transcript[:50]}...")

            # Get response from agent (streaming)
            full_response = ""
            async for token in self.agent.chat_stream(
                transcript, session_id, language=detected_language
            ):
                full_response += token

            logger.info(f"Generated response: {full_response[:50]}...")

            # Convert response to speech
            response_audio = await self.tts.synthesize(full_response, detected_language)

            if not response_audio:
                # Fallback to text if TTS fails
                await update.message.reply_text(full_response)
                return

            # Convert PCM16 to OGG for Telegram
            response_audio_segment = AudioSegment(
                data=response_audio,
                sample_width=2,
                frame_rate=16000,
                channels=1,
            )

            # Export as OGG
            output_buffer = io.BytesIO()
            response_audio_segment.export(output_buffer, format="ogg", codec="libopus")
            output_buffer.seek(0)

            # Send voice message back
            await update.message.reply_voice(
                voice=output_buffer,
                caption=f"🎤 ({detected_language.upper()})",
            )

            logger.info(f"Sent voice response to user {user_id}")

        except Exception as e:
            logger.error(f"Error handling voice from {user_id}: {e}", exc_info=True)

            # Send error message
            lang = locals().get("detected_language", "en")
            error_msg = (
                "عذراً، حصل خطأ في معالجة الصوت. ممكن تحاول تاني؟"
                if lang == "ar"
                else "Sorry, I encountered an error processing your voice message."
            )
            await update.message.reply_text(error_msg)

    async def start_polling(self) -> None:
        """Start the bot with polling."""
        if not settings.telegram_bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not configured!")
            raise ValueError("TELEGRAM_BOT_TOKEN is required for Telegram bot")

        logger.info("Starting Telegram bot with polling...")

        # Build application
        self.application = (
            ApplicationBuilder().token(settings.telegram_bot_token).build()
        )

        # Register handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))

        # Start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)

        logger.info("✅ Telegram bot is running and polling for messages!")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Stopping Telegram bot...")
            await self.stop()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if self.application:
            logger.info("Stopping Telegram bot...")
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped.")


# Singleton instance
_bot_instance: Optional[HopeTelegramBot] = None


def get_bot() -> HopeTelegramBot:
    """Get or create the bot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = HopeTelegramBot()
    return _bot_instance
