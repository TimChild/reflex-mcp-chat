"""Regular graph implementation of langgraph."""

import logging
from typing import Annotated, Literal, Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel

from host_app.containers import Application
from host_app.mcp_client import MultiMCPClient
from host_app.models import InputState


class FullGraphState(BaseModel):
    """Full state used by and returned by graph."""

    question: str
    previous_messages: list[BaseMessage] = []
    response_messages: Annotated[list[AnyMessage], add_messages]
    tools: list[BaseTool] = []
    conversation_id: str | None = None


class LoadMessagesOutput(BaseModel):
    previous_messages: list[BaseMessage] = []


async def load_previous_messages(
    state: InputState,
    store: BaseStore,
) -> LoadMessagesOutput:
    question = state.question
    logging.debug(f"Processing question: {question}")

    previous_messages: Sequence[BaseMessage] = []
    logging.debug(f"Conversation ID: {state.conversation_id}")
    if state.conversation_id:
        found = await store.aget(namespace=("messages",), key=state.conversation_id)
        logging.debug(f"Found: {found}")
        if found:
            previous_messages = messages_from_dict(found.value["messages"])
    else:
        previous_messages = []
    return LoadMessagesOutput(
        previous_messages=previous_messages,
    )


class CallLLMOutput(BaseModel):
    response_messages: list[BaseMessage] = []


@inject
async def call_llm(
    state: FullGraphState,
    config: RunnableConfig,
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
    available_models: dict[str, BaseChatModel] = Provide[Application.llm_models],
    default_model: str = Provide[Application.config.default_model],
    system_prompt: str = Provide[Application.config.system_prompt],
) -> Command[Literal["tool_node", "save_messages"]]:
    if not state.tools:
        async with mcp_client as client:
            tools = await client.get_tools()
    else:
        tools = state.tools

    chat_model = available_models[config.get("configurable", {}).get("model_name", default_model)]
    model = chat_model.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(system_prompt),
        *state.previous_messages,
        HumanMessage(state.question),
        *state.response_messages,
    ]
    response: BaseMessage = await model.ainvoke(input=messages)
    assert isinstance(response, AIMessage)
    update = CallLLMOutput(response_messages=[response])

    if response.tool_calls:
        return Command(update=update, goto="tool_node")
    return Command(update=update, goto="save_messages")


async def save_messages(
    state: FullGraphState,
    store: BaseStore,
) -> None:
    if state.conversation_id:
        logging.debug(f"Saving messages for conversation ID: {state.conversation_id}")
        await store.aput(
            namespace=("messages",),
            key=state.conversation_id,
            value={
                "messages": messages_to_dict(
                    state.previous_messages
                    + [HumanMessage(state.question)]
                    + state.response_messages
                )
            },
        )
    return


class ToolNodeInput(BaseModel):
    response_messages: list[BaseMessage]
    tools: list[BaseTool]


class ToolNodeOutput(BaseModel):
    response_messages: list[BaseMessage]


@inject
async def call_tools(
    state: ToolNodeInput,
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
) -> ToolNodeOutput:
    async with mcp_client as client:
        tools = await client.get_tools()
        logging.debug("Calling tools")
        messages_state = await ToolNode(tools=tools, name="tool_node").ainvoke(
            input={"messages": state.response_messages.copy()}
        )
    results = messages_state["messages"]
    return ToolNodeOutput(response_messages=results)


@inject
async def make_graph(
    checkpointer: BaseCheckpointSaver | None = Provide[Application.checkpointer],
    store: BaseStore | None = Provide[Application.store],
    debug_mode: bool = False,
) -> CompiledGraph:
    checkpointer = checkpointer or MemorySaver()
    store = store or InMemoryStore()

    graph = StateGraph(state_schema=FullGraphState)
    graph.add_node("load_previous_messages", load_previous_messages)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tool_node", call_tools)
    graph.add_node("save_messages", save_messages)

    graph.set_entry_point("load_previous_messages")
    graph.add_edge("load_previous_messages", "call_llm")
    graph.add_edge("tool_node", "call_llm")
    # call_llm directs to tool_node or save_messages
    graph.set_finish_point("save_messages")

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=None,
        interrupt_after=None,
        debug=debug_mode,
    )
    return compiled_graph
