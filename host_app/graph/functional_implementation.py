"""Using the new Functional API for langgraph."""

import asyncio
import logging
from typing import Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.pregel import Pregel
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from host_app.containers import Application
from host_app.mcp_client import MultiMCPClient
from host_app.models import InputState


class InitializeOutput(BaseModel):
    previous_messages: list[BaseMessage] = []


@task
async def load_previous_messages(
    conversation_id: str | None,
    store: BaseStore,
) -> list[BaseMessage]:
    previous_messages: list[BaseMessage] = []
    if conversation_id:
        loaded = await store.aget(namespace=("messages",), key=conversation_id)
        if loaded:
            logging.debug("Loaded previous messages")
            previous_messages = messages_from_dict(loaded.value["messages"])
    return previous_messages


@task
async def save_messages(
    store: BaseStore,
    conversation_id: str,
    previous_messages: Sequence[BaseMessage],
    question: str,
    responses: Sequence[BaseMessage],
) -> None:
    await store.aput(
        namespace=("messages",),
        key=conversation_id,
        value={
            "messages": messages_to_dict(
                [
                    *previous_messages,
                    HumanMessage(question),
                    *responses,
                ]
            )
        },
    )


class CallToolsInput(BaseModel):
    response_messages: list[BaseMessage]
    tools: list[BaseTool]


class CallToolsOutput(BaseModel):
    response_messages: list[BaseMessage]


@task
async def call_tools(tool_calls: list[ToolCall], tools: Sequence[BaseTool]) -> list[ToolMessage]:
    def missing_message(tool_call: ToolCall) -> ToolMessage:
        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Error: Missing ToolMessage from tool {tool_call['name']}",
        )

    logging.debug(f"Calling tools: {[tc['name'] for tc in tool_calls]}")

    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    response_tasks = []
    async with asyncio.TaskGroup() as tg:
        for tool_call in tool_calls:
            if tool_call["name"] not in tools_by_name:
                response_tasks.append(missing_message(tool_call))
            tool = tools_by_name[tool_call["name"]]
            response_tasks.append(tg.create_task(tool.ainvoke(tool_call)))

    tool_responses = await asyncio.gather(*response_tasks)
    assert all(isinstance(tool_response, ToolMessage) for tool_response in tool_responses)
    logging.debug("Returning from call_tools")
    return tool_responses


class OutputState(BaseModel):
    response_messages: Sequence[AnyMessage]


@inject
async def make_graph(
    checkpointer: BaseCheckpointSaver = Provide[Application.checkpointer],
    store: BaseStore = Provide[Application.store],
    system_prompt: str = Provide[Application.config.system_prompt],
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
    default_model: str = Provide[Application.config.default_model],
    available_models: dict[str, BaseChatModel] = Provide[Application.llm_models],
    max_iterations: int = 10,
) -> Pregel:
    """Create a graph with the given checkpointer and store."""

    @entrypoint(checkpointer=checkpointer, store=store)
    async def graph(
        inputs: InputState,
        store: BaseStore,
        config: RunnableConfig,
    ) -> OutputState:
        responses: list[AIMessage | ToolMessage] = []
        question = inputs.question
        logging.debug(f"Processing question: {question}")

        async with mcp_client as client:
            tools = await client.get_tools()
            chat_model = available_models[
                config.get("configurable", {}).get("model_name", default_model)
            ]
            model = chat_model.bind_tools(tools)

            previous_messages = await load_previous_messages(
                conversation_id=inputs.conversation_id, store=store
            )

            message_history: list[BaseMessage] = [
                SystemMessage(system_prompt),
                *previous_messages,
                HumanMessage(question),
            ]

            # Loop calling ai -> tools -> ai ... until no more tool calls or max iterations
            for i in range(max_iterations):
                logging.debug(f"Iteration {i}")

                ai_message: BaseMessage = await model.ainvoke(input=message_history)
                assert isinstance(ai_message, AIMessage)
                message_history.append(ai_message)
                responses.append(ai_message)

                if not ai_message.tool_calls:
                    break

                tool_responses: list[ToolMessage] = await call_tools(
                    ai_message.tool_calls, tools=tools
                )
                message_history.extend(tool_responses)
                responses.extend(tool_responses)

        # Save the messages to the store
        if inputs.conversation_id:
            await save_messages(
                store=store,
                conversation_id=inputs.conversation_id,
                previous_messages=previous_messages,
                question=question,
                responses=responses,
            )

        return OutputState(response_messages=responses)

    return graph
