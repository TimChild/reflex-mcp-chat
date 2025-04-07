import os
from typing import Iterator

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from host_app.containers import Application


@pytest.fixture(scope="session")
def example_server_config() -> dict:
    return {
        "command": "uv",
        "args": ["run", "tests/example_server.py"],
    }


@pytest.fixture(scope="session")
def missing_stdio_server_config() -> dict:
    return {
        "command": "uv",
        "args": ["run", "non-existent-server.py"],
    }


@pytest.fixture(scope="session")
def missing_sse_server_config() -> dict:
    return {
        "url": "https://missing-server.com",
    }


@pytest.fixture(scope="session")
def with_fake_env_vars() -> Iterator[None]:
    """Set fake environment variables for testing.

    App requires them not to be blank, but for tests we don't actually use them.
    """
    env_vars: list[str] = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GITHUB_PERSONAL_ACCESS_TOKEN",
    ]
    before: dict[str, str] = {k: os.environ[k] for k in env_vars if k in os.environ}

    for k in env_vars:
        os.environ[k] = "test_" + k.lower().replace("_", "-")
    yield

    for k in env_vars:
        if k in before:
            os.environ[k] = before[k]
        else:
            del os.environ[k]


@pytest.fixture(scope="session")
def container(example_server_config: dict, with_fake_env_vars: None) -> Iterator[Application]:
    _ = with_fake_env_vars

    class NotSetModel:
        pass

    container = Application()
    container.wire()
    container.config.from_yaml("config.yml")
    with container.config.mcp_servers.override({"example_server": example_server_config}):
        with container.llm_models.override({container.config.default_model(): NotSetModel()}):
            with container.store.override(InMemoryStore()):
                with container.checkpointer.override(MemorySaver()):
                    yield container
    container.unwire()
