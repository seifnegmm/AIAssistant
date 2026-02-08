"""LangGraph ReAct agent for the AI assistant.

Builds a prebuilt *ReAct* agent that can call memory tools and stream
token-by-token responses back to the caller.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.agent.prompts import SYSTEM_PROMPT, build_system_prompt
from app.agent.tools import build_tools
from app.config import settings
from app.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class AssistantAgent:
    """Wraps a LangGraph ReAct agent with memory-aware tools."""

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory_manager = memory_manager

        # Build LLM
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            streaming=True,
        )

        # Build tools (binds memory_manager internally)
        self._tools = build_tools(memory_manager)

        # Build the ReAct agent graph
        self._graph = create_react_agent(
            model=self._llm,
            tools=self._tools,
        )

        logger.info(
            "AssistantAgent initialised with model=%s, tools=%d",
            settings.gemini_model,
            len(self._tools),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        session_id: str,
        *,
        history: list[BaseMessage] | None = None,
        language: str = "en",
    ) -> str:
        """Send a message and return the full response (non-streaming).

        Also persists both the user message and AI response to MongoDB via
        LangChain's ``MongoDBChatMessageHistory`` so there is a single
        authoritative history store.

        Args:
            message: User's text message.
            session_id: Current session identifier.
            history: Optional prior messages for context.
            language: Language code for response ('en' or 'ar'). Default 'en'.

        Returns:
            The assistant's text response.
        """
        messages = self._build_messages(message, session_id, history, language)
        config = {"configurable": {"thread_id": session_id}}

        result = await self._graph.ainvoke(
            {"messages": messages},
            config=config,
        )

        # Extract the agent's new messages (everything after our input)
        all_result_msgs: list[BaseMessage] = result["messages"]
        # Our input had len(messages) items; the agent appended its own
        new_messages = all_result_msgs[len(messages) :]

        # The final text response is the last AI message
        response_text = ""
        for msg in reversed(new_messages):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = msg.content
                break

        # Persist the full exchange: user message + all agent steps
        # (AI tool-call messages, ToolMessages, final AI response)
        history_store = self._memory_manager.get_langchain_chat_history(session_id)
        to_persist: list[BaseMessage] = [HumanMessage(content=message)]
        to_persist.extend(new_messages)
        history_store.add_messages(to_persist)

        return response_text

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        *,
        history: list[BaseMessage] | None = None,
        language: str = "en",
    ) -> AsyncIterator[str]:
        """Send a message and yield response tokens as they arrive.

        Only tokens from the **final** LLM call (the user-visible answer) are
        streamed.  Intermediate LLM calls (tool-selection steps) are silently
        captured so their tool-call metadata can be merged into the persisted
        MongoDB document.

        After streaming completes, persists exactly TWO entries per turn:
        ``[HumanMessage, AIMessage]`` where the ``AIMessage`` carries the
        final text *and* an enriched ``tool_calls`` list (each entry includes
        the ``result`` returned by the tool).

        Args:
            message: User's text message.
            session_id: Current session identifier.
            history: Optional prior messages for context.
            language: Language code for response ('en' or 'ar'). Default 'en'.

        Yields:
            Chunks of the assistant's **final** response text.
        """
        messages = self._build_messages(message, session_id, history, language)
        config = {"configurable": {"thread_id": session_id}}

        final_response = ""

        # --- Intermediate-step accumulators ---
        # Pending tool calls from the latest intermediate AIMessage.
        # Keyed by tool_call id → {name, args, result}
        pending_tool_calls: dict[str, dict] = {}
        # All resolved tool calls across the entire turn (preserves order).
        resolved_tool_calls: list[dict] = []
        # True while the current LLM invocation is an intermediate step
        # (i.e. the model is deciding which tool to call, NOT answering the user).
        _is_intermediate_step = False
        # Track whether we've seen a tool_call_chunk in the current LLM stream.
        _saw_tool_call_in_current_stream = False

        async for event in self._graph.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            # ── Streaming tokens from the LLM ──────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk is None:
                    continue

                if isinstance(chunk, AIMessageChunk):
                    # If this chunk carries tool-call fragments, mark the
                    # current LLM invocation as an intermediate step.
                    if chunk.tool_call_chunks:
                        _saw_tool_call_in_current_stream = True
                        for tc_chunk in chunk.tool_call_chunks:
                            tc_id = tc_chunk.get("id", "")
                            if tc_id and tc_id not in pending_tool_calls:
                                pending_tool_calls[tc_id] = {
                                    "name": tc_chunk.get("name", ""),
                                    "args": tc_chunk.get("args", ""),
                                    "id": tc_id,
                                    "result": None,
                                }

                    # Only yield text to the user if this is NOT an
                    # intermediate (tool-calling) step.
                    if not _saw_tool_call_in_current_stream:
                        content = chunk.content
                        if content:
                            final_response += content
                            yield content

            # ── LLM call finished ──────────────────────────────────────
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, AIMessage):
                    if output.tool_calls:
                        # Intermediate step — update pending_tool_calls with
                        # the fully-assembled tool_calls from LangChain.
                        for tc in output.tool_calls:
                            tc_id = tc.get("id", "")
                            pending_tool_calls[tc_id] = {
                                "name": tc["name"],
                                "args": tc["args"],
                                "id": tc_id,
                                "result": None,
                            }
                    # If this was NOT an intermediate step and we got text,
                    # it should already have been streamed above.

                # Reset per-LLM-call flag for the next invocation.
                _saw_tool_call_in_current_stream = False

            # ── Tool execution finished ────────────────────────────────
            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output")
                tool_name = event.get("name", "")
                if output is not None:
                    tool_content = output if isinstance(output, str) else str(output)
                    # Attach result to the matching pending tool call.
                    matched = False
                    for tc_id, tc_info in pending_tool_calls.items():
                        if tc_info["name"] == tool_name and tc_info["result"] is None:
                            tc_info["result"] = tool_content
                            resolved_tool_calls.append(tc_info)
                            matched = True
                            break
                    if not matched:
                        # Fallback: tool call id not found, store anyway
                        resolved_tool_calls.append(
                            {
                                "name": tool_name,
                                "args": {},
                                "id": "",
                                "result": tool_content,
                            }
                        )

        # ── Persist exactly ONE assistant message ──────────────────────
        # Build a single AIMessage that holds both the final answer text
        # and the enriched tool_calls list (each with its result).
        ai_kwargs: dict = {}
        if resolved_tool_calls:
            ai_kwargs["additional_kwargs"] = {
                "tool_calls_with_results": resolved_tool_calls,
            }

        history_store = self._memory_manager.get_langchain_chat_history(session_id)
        history_store.add_messages(
            [
                HumanMessage(content=message),
                AIMessage(content=final_response, **ai_kwargs),
            ]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_text: str,
        session_id: str,
        history: list[BaseMessage] | None,
        language: str = "en",
    ) -> list[BaseMessage]:
        """
        Construct the message list including system prompt + history from MongoDB.

        Args:
            user_text: The user's message text
            session_id: Session identifier for formatting
            history: Previous conversation messages
            language: Language code ('en' or 'ar') for bilingual system prompt
        """
        # Build dynamic system prompt based on detected language
        system_text = build_system_prompt(language).format(
            current_time=datetime.now(timezone.utc).strftime(
                "%A, %B %d, %Y at %I:%M %p UTC"
            ),
            session_id=session_id,
        )

        messages: list[BaseMessage] = [SystemMessage(content=system_text)]

        # Load conversation history from MongoDB
        if history:
            messages.extend(history)
        else:
            # Load from persistent storage via langchain-mongodb
            lc_history = self._memory_manager.get_langchain_chat_history(session_id)
            for msg in lc_history.messages:
                messages.append(msg)

        # Add the current user message
        messages.append(HumanMessage(content=user_text))

        return messages

    @property
    def system_prompt(self) -> str:
        """Return the formatted system prompt (for debugging)."""
        return SYSTEM_PROMPT.format(
            current_time=datetime.now(timezone.utc).strftime(
                "%A, %B %d, %Y at %I:%M %p UTC"
            ),
            session_id="debug",
        )
