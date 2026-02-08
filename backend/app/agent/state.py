"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState:
    """Typed dictionary for the ReAct agent state.

    Uses LangGraph's ``add_messages`` reducer so that new messages are
    *appended* to the existing list rather than replacing it.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
