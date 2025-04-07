"""Common data structures used within the app."""

from enum import StrEnum
from typing import Any, Protocol

import reflex as rx
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from pydantic import BaseModel


class InputState(BaseModel):
    """State required to run the graph."""

    question: str
    conversation_id: str | None = None


class UpdateTypes(StrEnum):
    """THe types of update that are sent back for the frontend to display."""

    preprocess = "preprocess"
    graph_start = "graph-start"
    ai_message_start = "ai-message-start"
    ai_stream = "ai-delta"
    ai_stream_tool_call = "ai-tool-call-delta"
    ai_message_end = "ai-message-end"
    tools_start = "tools-start"
    tool_end = "tool-end"
    graph_end = "graph-end"
    value_update = "value-update"


class GraphUpdate(Protocol):
    """General protocol for updates."""

    type_: UpdateTypes


class GeneralUpdate(rx.Base):
    """General update for the graph."""

    type_: UpdateTypes
    data: Any | None = None


class GraphMetadata(rx.Base):
    # There are other attributes, but this is the only one needed for now.
    node: str


class AIStartUpdate(rx.Base):
    """Update for start of AI messages."""

    type_ = UpdateTypes.ai_message_start
    m_id: str
    metadata: GraphMetadata


class AIStreamUpdate(rx.Base):
    """Update for streaming AI messages."""

    type_ = UpdateTypes.ai_stream
    m_id: str
    delta: str


class AIEndUpdate(rx.Base):
    """Update for end of AI messages."""

    type_ = UpdateTypes.ai_message_end
    m_id: str
    response: AIMessage


class ToolCallInfo(rx.Base):
    """Info for each tool call."""

    name: str
    args: dict[str, Any]
    id: str | None


class ToolsStartUpdate(rx.Base):
    type_ = UpdateTypes.tools_start
    calls: list[ToolCallInfo]


class ToolEndUpdate(rx.Base):
    type_ = UpdateTypes.tool_end
    tool_response: ToolMessage


class ToolsUse(rx.Base):
    tool_calls: list[ToolCallInfo]


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    tool_uses: list[ToolsUse] = []
    answer: str


class ToolInfo(rx.Base):
    """Info for each MCP tool."""

    name: str
    description: str


class McpServerInfo(rx.Base):
    """Information about an MCP server."""

    name: str
    tools: list[ToolInfo]
