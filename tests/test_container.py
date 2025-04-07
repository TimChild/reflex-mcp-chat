"""Tests that the graph part of the app works correctly."""

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from host_app.containers import Application


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

    assert isinstance(container.llm_models(), dict)
    models = container.llm_models()
    assert len(models) == 1, "only expecting the default fake model for tests"
    assert container.config.default_model() in models
    assert not isinstance(models[container.config.default_model()], BaseChatModel), (
        "The default container model should NOT actually be a chat model (want it to raise errors if used)"
    )
