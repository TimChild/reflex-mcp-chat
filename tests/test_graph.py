"""Tests that the graph part of the app works correctly."""

import asyncio
import uuid
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.pregel import Pregel
from langgraph.store.base import BaseStore, Item
from langgraph.store.memory import InMemoryStore

from host_app.containers import Application, config_option_to_connections
from host_app.graph import GraphRunAdapter, make_functional_graph, make_standard_graph
from host_app.graph.functional_implementation import OutputState
from host_app.mcp_client import MultiMCPClient
from host_app.models import GraphUpdate, InputState, UpdateTypes

EXAMPLE_SERVER_CONFIG = {
    "command": "uv",
    "args": ["run", "tests/example_server.py"],
}
MISSING_STDIO_SERVER_CONFIG = {
    "command": "uv",
    "args": ["run", "non-existent-server.py"],
}
MISSING_SSE_SERVER_CONFIG = {
    "url": "https://missing-server.com",
}


class FakeChatModel(FakeMessagesListChatModel):
    # Note: store list of list to record bound tools each time it is bound
    tools_bound: list[list[BaseTool]] = []

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Optional[Union[str, Literal["any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        _, _ = tool_choice, kwargs
        base_tools: list[BaseTool] = []
        for tool in tools:
            assert isinstance(tool, BaseTool)
            base_tools.append(tool)
        self.tools_bound.append(base_tools)
        return self


@pytest.fixture
def fake_chat_model() -> FakeChatModel:
    return FakeChatModel(
        responses=[
            AIMessage("First response"),
            AIMessage("Second response"),
            AIMessage("Third response"),
        ]
    )


class NotSetModel:
    pass


@pytest.fixture(autouse=True, scope="session")
def container() -> Iterator[Application]:
    container = Application()
    container.wire()
    container.config.from_yaml("config.yml")
    with container.config.mcp_servers.override({"example_server": EXAMPLE_SERVER_CONFIG}):
        with container.llm_models.override({container.config.default_model(): NotSetModel()}):
            with container.store.override(InMemoryStore()):
                with container.checkpointer.override(MemorySaver()):
                    yield container
    container.unwire()


def test_container(container: Application):
    """Check the container is set up correctly."""
    conf = container.config.mcp_servers()["example_server"]
    assert isinstance(conf, dict)
    assert conf["command"] == "uv"
    assert conf["args"] == ["run", "tests/example_server.py"]
    connections = container.mcp_client().connections
    assert "example_server" in connections

    # NOTE: connections is a dict[name, SSEConnection | StdioConnection]
    assert "tests/example_server.py" in str(connections["example_server"]), (
        "Should include the path somewhere"
    )

    assert isinstance(container.store(), InMemoryStore)
    assert isinstance(container.checkpointer(), MemorySaver)

    assert len(container.config()["secrets"]) > 0


@pytest.fixture
def mock_chat_model(
    container: Application, fake_chat_model: FakeChatModel
) -> Iterator[FakeChatModel]:
    with container.llm_models.override({container.config.default_model(): fake_chat_model}):
        yield fake_chat_model


@pytest.fixture
def basic_runnable_config() -> RunnableConfig:
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
    }


async def test_compile_graph():
    graph = await make_standard_graph()
    assert isinstance(graph, CompiledGraph)
    assert isinstance(graph.checkpointer, BaseCheckpointSaver)


@pytest.fixture(params=["standard", "functional"])
async def graph(request: pytest.FixtureRequest, mock_chat_model: FakeChatModel) -> Pregel:
    _ = mock_chat_model
    if request.param == "functional":
        return await make_functional_graph()
    else:
        return await make_standard_graph()


async def test_invoke_graph(graph: Pregel, basic_runnable_config: RunnableConfig):
    result: dict[str, Any] = await graph.ainvoke(
        input=InputState(question="Hello"), config=basic_runnable_config
    )
    output: OutputState = OutputState.model_validate(result)
    assert output.response_messages[0].content == "First response"


def test_init_graph_adapter(graph: Pregel):
    runner = GraphRunAdapter(graph)
    assert isinstance(runner, GraphRunAdapter)
    connections = runner.mcp_client.connections
    assert isinstance(connections, dict)
    assert list(connections.keys()) == ["example_server"]


@pytest.fixture
def graph_adapter(graph: Pregel) -> GraphRunAdapter:
    return GraphRunAdapter(graph)


@pytest.mark.usefixtures("mock_chat_model")
async def test_astream_graph_adapter(graph_adapter: GraphRunAdapter):
    updates: list[GraphUpdate] = []
    async for update in graph_adapter.astream_updates(input=InputState(question="Hello")):
        assert hasattr(update, "type_")
        updates.append(update)

    assert len(updates) > 0
    assert updates[0].type_ == UpdateTypes.graph_start
    assert updates[-1].type_ == UpdateTypes.graph_end


async def test_memory_store_standalone(container: Application):
    store = container.store()
    before = await store.aget(namespace=("testing",), key="test")
    assert ("testing",) in store.list_namespaces(), "Should have been created on attempted access"
    assert before is None

    # Test retrieving with same `store`
    await store.aput(namespace=("testing",), key="test", value={"value": "value"})
    after = await store.aget(namespace=("testing",), key="test")
    assert isinstance(after, Item)
    assert after.value == {"value": "value"}

    # Check that `store` is a singleton
    store2 = container.store()
    after2 = await store2.aget(namespace=("testing",), key="test")
    assert isinstance(after2, Item)
    assert after2.value == {"value": "value"}


async def test_graph_binds_tools(graph_adapter: GraphRunAdapter, mock_chat_model: FakeChatModel):
    _ = await graph_adapter.ainvoke(input=InputState(question="Hello"))

    assert len(mock_chat_model.tools_bound) == 1, "Should bind tools once"
    assert mock_chat_model.tools_bound[0][0].name == "test-tool", (
        "Should bind test-tool during tests"
    )


@pytest.mark.usefixtures("mock_chat_model")
async def test_graph_has_memory(graph_adapter: GraphRunAdapter, container: Application):
    _ = await graph_adapter.ainvoke(
        input=InputState(question="Hello", conversation_id="test-conv-id"),
    )

    store: BaseStore = container.store()
    assert ("messages",) in store.list_namespaces()
    value = store.get(namespace=("messages",), key="test-conv-id")
    assert value is not None


async def test_mcp_client_with_missing_server(container: Application):
    with container.config.mcp_servers.override(
        {
            "example_server": EXAMPLE_SERVER_CONFIG,
            "missing_server": MISSING_STDIO_SERVER_CONFIG,
        }
    ):
        # mcp_client = container.adapters.mcp_client()
        conns = config_option_to_connections(container.config.mcp_servers())
        mcp_client = MultiMCPClient(connections=conns)
        mcp_client.set_connection_timeout(0.5)

        async def func() -> None:
            async with mcp_client:
                pass

        try:
            await asyncio.wait_for(func(), timeout=1)
        except TimeoutError:
            pytest.fail("Should not hang on missing server")


@pytest.mark.usefixtures("mock_chat_model")
async def test_graph_runs_with_missing_mcp_server(
    graph_adapter: GraphRunAdapter, container: Application
):
    """Should still be able to run the graph even if one of the servers is down."""
    with container.config.mcp_servers.override(
        {
            "example_server": EXAMPLE_SERVER_CONFIG,
            "missing_server": MISSING_SSE_SERVER_CONFIG,
        }
    ):
        mcp_client = container.mcp_client()
        mcp_client.set_connection_timeout(0.5)
        response: OutputState = await graph_adapter.ainvoke(input=InputState(question="Hello"))
        assert response.response_messages[0].content == "First response"


class TestWithToolCalls:
    async def test_single_call(
        self, graph_adapter: GraphRunAdapter, mock_chat_model: FakeChatModel
    ):
        tool_call = ToolCall(id="test-call-id", name="test-tool", args={})
        mock_chat_model.responses = [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage("Response after tool call"),
        ]
        response: OutputState = await graph_adapter.ainvoke(input=InputState(question="Hello"))

        first_message = response.response_messages[0]
        assert isinstance(first_message, AIMessage)
        assert first_message.content == ""
        assert len(first_message.tool_calls) == 1

        second_message = response.response_messages[1]
        assert isinstance(second_message, ToolMessage)

        third_message = response.response_messages[2]
        assert isinstance(third_message, AIMessage)
        assert third_message.content == "Response after tool call"

    async def test_sequential_calls(
        self, graph_adapter: GraphRunAdapter, mock_chat_model: FakeChatModel
    ):
        tool_call1 = ToolCall(id="test-call-id1", name="test-tool", args={})
        tool_call2 = ToolCall(id="test-call-id2", name="test-tool", args={})
        mock_chat_model.responses = [
            AIMessage(content="", tool_calls=[tool_call1]),
            AIMessage(content="", tool_calls=[tool_call2]),
            AIMessage("Response after tool calls"),
        ]
        response: OutputState = await graph_adapter.ainvoke(input=InputState(question="Hello"))

        first_message = response.response_messages[0]
        assert isinstance(first_message, AIMessage)
        assert first_message.content == ""
        assert len(first_message.tool_calls) == 1

        second_message = response.response_messages[1]
        assert isinstance(second_message, ToolMessage)

        third_message = response.response_messages[2]
        assert isinstance(third_message, AIMessage)
        assert third_message.content == ""
        assert len(third_message.tool_calls) == 1

        fourth_message = response.response_messages[3]
        assert isinstance(fourth_message, ToolMessage)

        fifth_message = response.response_messages[4]
        assert isinstance(fifth_message, AIMessage)
        assert fifth_message.content == "Response after tool calls"
