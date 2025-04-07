"""The application logic and state management.

I.e. the dynamic behavior of the app.
"""

import logging
import uuid
from typing import Any, AsyncIterator, Mapping, Sequence

import reflex as rx
from dependency_injector.wiring import Provide, inject
from langchain_core.tools import BaseTool
from reflex.event import EventType

from host_app.containers import Application
from host_app.graph import GraphRunAdapter, make_functional_graph, make_standard_graph
from host_app.mcp_client import MultiMCPClient

from .models import (
    QA,
    AIEndUpdate,
    AIStartUpdate,
    AIStreamUpdate,
    GeneralUpdate,
    GraphUpdate,
    InputState,
    McpServerInfo,
    ToolEndUpdate,
    ToolInfo,
    ToolsStartUpdate,
    ToolsUse,
    UpdateTypes,
)

DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    chats: dict[str, list[QA]] = DEFAULT_CHATS
    """A dict from the chat name to the list of questions and answers."""

    current_chat = "Intros"
    """The current chat name."""

    question: str
    """The current question."""

    processing: bool = False
    """Whether we are processing the question."""

    current_status: str = ""
    """The current status."""

    new_chat_name: str = ""
    """The name of the new chat."""

    modal_open: bool = False
    """The new chat modal open state."""

    mcp_servers: list[McpServerInfo] = []
    """The connected MCP servers."""

    graph_mode: str = "functional"  # functional or standard
    model_name: str = ""

    @rx.event
    def set_new_chat_name(self, name: str) -> None:
        """Set the name of the new chat.

        Args:
            form_data: A dict with the new chat name.
        """
        self.new_chat_name = name

    @rx.event
    def toggle_modal(self) -> None:
        """Toggle the modal."""
        self.modal_open = not self.modal_open

    @rx.event
    def create_chat(self) -> None:
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    @rx.event
    def delete_chat(self) -> None:
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    @rx.event
    def set_chat(self, chat_name: str) -> None:
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.event
    def set_model(self, model_name: str) -> None:
        """Set the model name.

        Args:
            model_name: The name of the model.
        """
        self.model_name = model_name

    @rx.event
    def set_graph_mode(self, graph_mode: str) -> None:
        """Set the graph mode.

        Args:
            graph_mode: The graph mode.
        """
        assert graph_mode in ["functional", "standard"]
        self.graph_mode = graph_mode  # type: ignore[reportAttributeAccessIssue]

    @rx.event
    @inject
    async def on_load(self, mcp_client: MultiMCPClient = Provide[Application.mcp_client]) -> None:
        """Load the state."""
        server_tools: Mapping[str, Sequence[BaseTool]] = await mcp_client.get_tools_by_server()
        self.mcp_servers = []
        for server_name, tools in server_tools.items():
            tool_infos = [ToolInfo(name=tool.name, description=tool.description) for tool in tools]
            self.mcp_servers.append(McpServerInfo(name=server_name, tools=tool_infos))

    @rx.var(cache=True)
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    @rx.event
    async def handle_send_click(self, form_data: dict[str, Any]) -> EventType | None:
        """Handle user clicking the send button."""
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if not question:
            return

        # Initialize new QA object
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)
        self.processing = True
        self.current_status = "Starting..."
        self.question = question
        # Switch to background task because it could take a while to run
        return State.run_request_in_background

    @rx.event(background=True)
    async def run_request_in_background(self) -> AsyncIterator[EventType | None]:
        """Background task to run the langgraph graph.

        Required to use a background task because this could take a while to run.
        Note: Use `async with self:` in order to update the state in the background task.
        """
        question = self.question

        # Build the functional or standard graph to run
        graph = (
            await make_functional_graph()
            if self.graph_mode == "functional"
            else await make_standard_graph()
        )

        # (since we get updates per tool, we can only check that all tools are done when we
        #  get the next AI message)
        tool_ended = False

        # Run the graph via the adapter, handling updates.
        async for update in GraphRunAdapter(graph).astream_updates(
            input=InputState(question=question, conversation_id=self.current_chat),
            thread_id=str(uuid.uuid4()),
            llm_model=self.model_name if self.model_name else None,
        ):
            update: GraphUpdate
            match update.type_:
                case UpdateTypes.graph_start:
                    logging.debug("Graph start update")
                    assert isinstance(update, GeneralUpdate)
                    pass
                case UpdateTypes.ai_message_start:
                    logging.debug("AI start update")
                    assert isinstance(update, AIStartUpdate)
                    if tool_ended:
                        # Must have just finished getting tool responses
                        async with self:
                            self.chats[self.current_chat][
                                -1
                            ].answer += "\n\nFinished calling tool.\n\n---\n\n"
                            self.current_status = "Finished calling tools."
                        tool_ended = False
                case UpdateTypes.ai_stream:
                    logging.debug("AI delta update")
                    assert isinstance(update, AIStreamUpdate)
                    async with self:
                        self.chats[self.current_chat][-1].answer += update.delta
                        self.chats = self.chats
                case UpdateTypes.ai_stream_tool_call:
                    pass
                case UpdateTypes.ai_message_end:
                    logging.debug("AI message end update")
                    assert isinstance(update, AIEndUpdate)
                    pass
                case UpdateTypes.tools_start:
                    logging.debug("Tools start update")
                    assert isinstance(update, ToolsStartUpdate)
                    async with self:
                        self.chats[self.current_chat][-1].answer += "\n\n---\n\nCalling tools..."
                        self.chats[self.current_chat][-1].tool_uses.append(
                            ToolsUse(tool_calls=update.calls)
                        )
                        self.chats = self.chats
                        self.current_status = (
                            f"Calling tools: {[call.name for call in update.calls]})"
                        )
                case UpdateTypes.tool_end:
                    # NOTE: Get update for *each* finished tool
                    logging.debug("Tool end update")
                    assert isinstance(update, ToolEndUpdate)
                    tool_ended = True
                case UpdateTypes.graph_end:
                    logging.debug("Graph end update")
                    assert isinstance(update, GeneralUpdate)
                    pass
                case _:
                    logging.info(f"Unknown update type: {update.type_}")
                    async with self:
                        self.current_status = f"Unknown update type: {update.type_}"
            yield

        # Reset the state after processing
        async with self:
            self.current_status = ""
            self.processing = False
