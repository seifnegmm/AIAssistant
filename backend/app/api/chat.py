"""WebSocket endpoint for real-time chat with the AI assistant."""

import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.graph import AssistantAgent
from app.config import settings
from app.dependencies import get_memory_manager
from app.memory.preference_learner import PreferenceLearner
from app.utils.language_detector import LanguageDetector
from app.voice.stt import StreamingSTT
from app.voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


async def websocket_chat(websocket: WebSocket) -> None:
    """Handle WebSocket connections for real-time chat.

    Protocol:
        Client sends JSON: {"type": "text", "content": "..."}
        Client sends binary: raw PCM16 audio chunks (buffered until stop_audio)
        Client sends JSON: {"type": "stop_audio"} to trigger STT on buffered audio
        Server sends JSON: {"type": "text"|"stream"|"status"|"error", "content": "...",
                            "session_id": "...", "timestamp": "..."}
    """
    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
    await websocket.accept()
    logger.info("WebSocket connected: session_id=%s", session_id)

    await _send_message(websocket, "status", "Connected", session_id)

    memory_manager = get_memory_manager()
    lang_detector = LanguageDetector()

    try:
        agent = AssistantAgent(memory_manager)
    except Exception as exc:
        logger.exception("Failed to create agent for session %s", session_id)
        await _send_message(
            websocket, "error", f"Agent initialization failed: {exc}", session_id
        )
        await websocket.close(code=1011)
        return

    # Buffer to accumulate PCM audio chunks while the user is speaking
    audio_buffer = bytearray()

    try:
        while True:
            ws_message = await websocket.receive()

            # --- Graceful disconnect ---
            if ws_message.get("type") == "websocket.disconnect":
                logger.info("WebSocket disconnect received: session_id=%s", session_id)
                break

            # --- Binary frame: raw PCM audio from the microphone ---
            # Accumulate chunks; don't transcribe yet.
            if "bytes" in ws_message and ws_message["bytes"]:
                audio_buffer.extend(ws_message["bytes"])
                continue

            # --- Text frame: JSON messages ---
            raw = ws_message.get("text")
            if not raw:
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await _send_message(websocket, "error", "Invalid JSON", session_id)
                continue

            msg_type = data.get("type", "text")
            content = data.get("content", "")

            if msg_type == "stop_audio":
                # User released the mic — transcribe the accumulated buffer
                if len(audio_buffer) < 3200:
                    # Less than ~100ms of audio at 16kHz/16bit — too short
                    logger.debug(
                        "Audio buffer too small (%d bytes), ignoring", len(audio_buffer)
                    )
                    audio_buffer.clear()
                    await _send_message(
                        websocket, "error", "Recording too short", session_id
                    )
                    continue
                audio_bytes = bytes(audio_buffer)
                audio_buffer.clear()
                logger.info(
                    "Processing buffered audio: %d bytes (~%.1fs) for session %s",
                    len(audio_bytes),
                    len(audio_bytes) / (16000 * 2),
                    session_id,
                )
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                await _handle_audio_message(
                    websocket,
                    agent,
                    memory_manager,
                    session_id,
                    audio_b64,
                    lang_detector,
                )
            elif msg_type == "text" and content.strip():
                await _handle_text_message(
                    websocket, agent, memory_manager, session_id, content, lang_detector
                )
            elif msg_type == "audio" and content:
                await _handle_audio_message(
                    websocket, agent, memory_manager, session_id, content, lang_detector
                )
            else:
                await _send_message(
                    websocket, "error", "Empty or unsupported message", session_id
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session_id=%s", session_id)
    except Exception as exc:
        logger.exception("WebSocket error for session %s", session_id)
        try:
            await _send_message(websocket, "error", str(exc), session_id)
        except Exception:
            pass


async def _handle_text_message(
    websocket: WebSocket,
    agent: AssistantAgent,
    memory_manager,
    session_id: str,
    content: str,
    lang_detector: LanguageDetector,
) -> None:
    """Process a text message through the LangGraph agent with streaming."""
    await _send_message(websocket, "status", "Thinking...", session_id)

    detected_language = lang_detector.detect(content, session_id)

    try:
        # Stream the agent response token-by-token.
        # The agent persists both user + AI messages to MongoDB after streaming.
        full_response = ""
        async for token in agent.chat_stream(
            content, session_id, language=detected_language
        ):
            full_response += token
            await _send_message(websocket, "stream", token, session_id)

        # If no tokens were streamed, use non-streaming fallback.
        # The non-streaming path also persists to MongoDB internally.
        if not full_response:
            full_response = await agent.chat(
                content, session_id, language=detected_language
            )
            await _send_message(websocket, "text", full_response, session_id)
        else:
            # Send final complete message
            await _send_message(websocket, "text", full_response, session_id)

        # Learn preferences in background (fire-and-forget)
        asyncio.create_task(
            _learn_preferences(memory_manager, session_id, content, full_response)
        )

    except Exception as exc:
        logger.exception("Error processing message in session %s", session_id)
        await _send_message(websocket, "error", f"Processing error: {exc}", session_id)


async def _learn_preferences(
    memory_manager,
    session_id: str,
    user_message: str,
    ai_response: str,
) -> None:
    """Background task to extract and store user preferences from conversation."""
    try:
        learner = PreferenceLearner(memory_manager)
        await learner.learn_from_exchange(user_message, ai_response)
    except Exception as exc:
        logger.warning("Preference learning failed for session %s: %s", session_id, exc)


async def _handle_audio_message(
    websocket: WebSocket,
    agent: AssistantAgent,
    memory_manager,
    session_id: str,
    audio_b64: str,
    lang_detector: LanguageDetector,
) -> None:
    """Process audio: STT → Agent → TTS → send audio back."""
    await _send_message(websocket, "status", "Processing audio...", session_id)

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        await _send_message(websocket, "error", "Invalid audio data", session_id)
        return

    try:
        stt = StreamingSTT()
        await stt.initialize()
        transcript, detected_language = await stt.transcribe(
            audio_bytes, auto_detect=True
        )

        if not transcript.strip():
            await _send_message(
                websocket, "error", "Could not transcribe audio", session_id
            )
            return

        await _send_message(websocket, "transcript", transcript, session_id)

        full_response = ""
        async for token in agent.chat_stream(
            transcript, session_id, language=detected_language
        ):
            full_response += token
            await _send_message(websocket, "stream", token, session_id)

        if not full_response:
            full_response = await agent.chat(
                transcript, session_id, language=detected_language
            )
            await _send_message(websocket, "text", full_response, session_id)
        else:
            await _send_message(websocket, "text", full_response, session_id)

        # Learn preferences in background (fire-and-forget)
        asyncio.create_task(
            _learn_preferences(memory_manager, session_id, transcript, full_response)
        )

        # Synthesize speech and send audio back as base64 PCM16
        tts = TextToSpeech()
        await tts.initialize()
        pcm_audio = await tts.synthesize(full_response, language=detected_language)

        if pcm_audio:
            audio_response_b64 = base64.b64encode(pcm_audio).decode("ascii")
            await websocket.send_json(
                {
                    "type": "audio",
                    "content": audio_response_b64,
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "format": "pcm16_16khz_mono",
                }
            )

    except Exception as exc:
        logger.exception("Audio processing error in session %s", session_id)
        await _send_message(
            websocket, "error", f"Audio processing error: {exc}", session_id
        )


async def _send_message(
    websocket: WebSocket,
    msg_type: str,
    content: str,
    session_id: str,
) -> None:
    """Send a structured JSON message over the WebSocket."""
    payload = {
        "type": msg_type,
        "content": content,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await websocket.send_json(payload)
