"""Agent tools for memory recall and saving."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import settings

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

_memory_manager: MemoryManager | None = None


def build_tools(memory_manager: MemoryManager) -> list:
    global _memory_manager  # noqa: PLW0603
    _memory_manager = memory_manager
    tools = [
        recall_memory,
        save_memory,
        update_memory,
        delete_memory,
        recall_preferences,
        get_current_time,
    ]

    # Add web search tool only if API key is configured
    if settings.tavily_api_key:
        tools.append(web_search)
    else:
        logger.warning("TAVILY_API_KEY not set - web search tool disabled")

    return tools


def _get_manager() -> MemoryManager:
    if _memory_manager is None:
        raise RuntimeError("Tools not initialised – call build_tools() first")
    return _memory_manager


def _run_async(coro):
    """Bridge async MemoryManager calls from sync tool functions.

    LangGraph runs tools in a ThreadPoolExecutor where there is no running
    event loop.  We create a fresh loop in that thread and run the coroutine
    on it.
    """
    return asyncio.run(coro)


@tool
def recall_memory(query: str) -> str:
    """Search long-term memory for information relevant to the query.

    Use this tool to recall facts, context, or past conversations that may
    be relevant to the current discussion.

    Args:
        query: A natural-language description of what to look for.
    """
    manager = _get_manager()
    try:
        results = _run_async(manager.recall_memory_with_scores(query, k=5))

        if not results:
            return "No relevant memories found."

        formatted: list[str] = []
        for doc, score in results:
            formatted.append(f"- [{score:.2f}] {doc.page_content}")
        return "Relevant memories:\n" + "\n".join(formatted)
    except Exception:
        logger.exception("recall_memory failed")
        return "Unable to search memory at this time."


@tool
def save_memory(content: str, category: str = "general") -> str:
    """Save important information to long-term memory.

    Use this when the user shares facts about themselves, their preferences,
    or any information worth remembering for future conversations.

    Args:
        content: The information to remember.
        category: Category tag – e.g. 'preference', 'fact', 'context'.
    """
    manager = _get_manager()
    try:
        _run_async(
            manager.save_memory(
                content=content,
                metadata={
                    "category": category,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        return f"Saved to memory under '{category}'."
    except Exception:
        logger.exception("save_memory failed")
        return "Failed to save memory."


@tool
def update_memory(old_info: str, new_info: str, category: str = "general") -> str:
    """Update a piece of information in long-term memory.

    Use this when the user corrects something you previously remembered.
    This will find and delete the old memory, then save the corrected version.

    For example, if the user previously said their favourite colour is blue
    but now says it's green, call this with:
        old_info="favourite colour is blue"
        new_info="The user's favourite colour is green"

    Args:
        old_info: A description of the outdated/incorrect memory to remove.
        new_info: The corrected information to save in its place.
        category: Category tag – e.g. 'preference', 'fact', 'context'.
    """
    manager = _get_manager()
    try:
        # Step 1: Delete old matching memories (SYNC call - delete_memory is not async)
        deleted = manager.delete_memory(query=old_info, k=2)
        logger.info(
            "update_memory: deleted %d old memories matching '%s'",
            deleted,
            old_info[:80],
        )

        # Step 2: Save the corrected information
        _run_async(
            manager.save_memory(
                content=new_info,
                metadata={
                    "category": category,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "updated_from": old_info[:200],
                },
            )
        )
        if deleted > 0:
            return f"Updated memory: removed {deleted} old entry/entries and saved the corrected information."
        else:
            return (
                "No matching old memory found to remove, but saved the new information."
            )
    except Exception:
        logger.exception("update_memory failed")
        return "Failed to update memory."


@tool
def delete_memory(query: str) -> str:
    """Delete information from long-term memory.

    Use this when the user explicitly asks you to forget something,
    or when stored information is no longer relevant.

    Args:
        query: A description of the memory to find and delete.
    """
    manager = _get_manager()
    try:
        deleted = manager.delete_memory(query=query, k=2)  # SYNC call
        if deleted > 0:
            return f"Deleted {deleted} matching memory entry/entries."
        else:
            return "No matching memories found to delete."
    except Exception:
        logger.exception("delete_memory failed")
        return "Failed to delete memory."


@tool
def recall_preferences(topic: str = "general") -> str:
    """Recall learned user preferences.

    Use this to personalise responses based on what has been learned about
    the user's likes, dislikes, and communication style.

    Args:
        topic: The preference topic to look up (e.g. 'communication_style',
               'interests', or 'general' for all).
    """
    manager = _get_manager()
    try:
        results = _run_async(
            manager.recall_memory_with_scores(
                query=f"user preference: {topic}",
                k=5,
                collection_name="user_preferences",
            )
        )

        if not results:
            return "No preferences found for this topic."

        lines = [f"- {doc.page_content}" for doc, _score in results]
        return "Known preferences:\n" + "\n".join(lines)
    except Exception:
        logger.exception("recall_preferences failed")
        return "Unable to retrieve preferences."


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC.

    Useful when the user asks about the current time or date.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %B %d, %Y at %I:%M %p UTC")


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or real-time data.

    Use this tool when you need to find:
    - Current events, breaking news, or recent developments
    - Weather forecasts or current conditions
    - Sports scores, stock prices, or other live data
    - Facts or information that may have changed recently
    - Information not in your training data or long-term memory

    DO NOT use this for:
    - Information the user has already shared (use recall_memory instead)
    - General knowledge questions you can answer directly
    - Personal information about the user

    Args:
        query: A clear, specific search query (e.g., "weather in San Francisco today",
               "latest news about AI", "current Bitcoin price").

    Returns:
        Search results with relevant excerpts and sources.
    """
    if not settings.tavily_api_key:
        return "Web search is not available - API key not configured."

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)

        # Use basic search depth to save credits
        # max_results=3 keeps responses concise
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
        )

        # Format the results for the LLM
        results = []

        # Include the AI-generated answer summary if available
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}\n")

        # Add individual search results
        if response.get("results"):
            results.append("Sources:")
            for i, result in enumerate(response["results"], 1):
                title = result.get("title", "Untitled")
                content = result.get("content", "")
                url = result.get("url", "")

                results.append(f"\n{i}. {title}")
                results.append(
                    f"   {content[:200]}..." if len(content) > 200 else f"   {content}"
                )
                results.append(f"   Source: {url}")

        if not results:
            return "No search results found."

        return "\n".join(results)

    except Exception as e:
        logger.exception("web_search failed for query: %s", query)
        return f"Web search encountered an error: {str(e)}"
