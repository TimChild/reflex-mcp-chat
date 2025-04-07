"""Containers for dependency injection.

Any depenendencies of the application should be defined here.

This makes it easier to override dependencies for testing, and for centralizing application
configuration in a single config file.
"""

import logging.config
from typing import Any

# import aiosqlite
# import psycopg
# import psycopg.rows
from dependency_injector import containers, providers
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver, AsyncShallowPostgresSaver
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.store.postgres import AsyncPostgresStore
from host_app.mcp_client import MultiMCPClient, SSEConnection, StdioConnection

# Load .env file into environment variables (so they can be used in config.yml)
load_dotenv()


def config_option_to_connections(
    simple_config_dict: dict[str, dict[str, Any]],
) -> dict[str, StdioConnection | SSEConnection]:
    """Convert the mcp config dict from yaml to Connection instances.

    Basically just determine which type of connection, then check that the args are specified
    correctly for that connection type.

    Args:
        - The full mcp connections config as loaded from config.yaml

    Returns:
        - Sanitized connections dict utilizing the StdioConnection and SSEConnection TypedDict classes.
    """
    connections: dict[str, StdioConnection | SSEConnection] = {}
    assert isinstance(simple_config_dict, dict)
    for name, conf in simple_config_dict.items():
        if url := conf.get("url"):
            connections[name] = SSEConnection(transport="sse", url=url)
        elif command := conf.get("command"):
            args = conf.get("args", [])
            assert isinstance(args, list)
            assert all(isinstance(arg, str) for arg in args)
            connections[name] = StdioConnection(
                transport="stdio",
                command=command,
                args=args,
                env=None,
                encoding="utf-8",
                encoding_error_handler="strict",
            )
        else:
            raise ValueError(f"Invalid connection configuration: {conf}")
    return connections


## For Sqlite checkpointer (no store)
# class AsyncSqliteConn(resources.AsyncResource):
#     async def init(self, conn_string: str) -> aiosqlite.Connection:
#         """Initialize the checkpointer with the given connection string."""
#         conn = aiosqlite.connect(conn_string)
#         return await conn
#
#     async def shutdown(self, resource: Any) -> None:  # noqa: ANN401
#         assert isinstance(resource, aiosqlite.Connection)
#         await resource.close()


## For Postgres checkpointer and store
# class AsyncPostgresConn(resources.AsyncResource):
#     async def init(self, conn_string: str) -> psycopg.AsyncConnection:
#         """Initialize the checkpointer with the given connection string."""
#         conn = psycopg.AsyncConnection.connect(
#             conn_string,
#             autocommit=True,
#             prepare_threshold=0,
#             row_factory=psycopg.rows.dict_row,  # pyright: ignore[reportArgumentType]
#         )
#         return await conn
#
#     async def shutdown(self, resource: Any) -> None:  # noqa: ANN401
#         assert isinstance(resource, psycopg.AsyncConnection)
#         await resource.close()


class Application(containers.DeclarativeContainer):
    """The main application container for dependency injection."""

    config = providers.Configuration(yaml_files=["config.yml"], strict=True)
    "Load application configuration from config.yml"

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )
    "Initialize logging from config (validating the parameters)"

    mcp_client = providers.Factory(
        MultiMCPClient,
        connections=providers.Singleton(
            config_option_to_connections,
            config.mcp_servers,
        ),
    )
    "Single interface for working with multiple MCP clients"

    llm_models = providers.Dict(
        openai_gpt4o=providers.Factory(
            ChatOpenAI,
            model="gpt-4o",
            api_key=config.secrets.OPENAI_API_KEY,
        ),
        anthropic_claude_sonnet=providers.Factory(
            ChatAnthropic,
            model="claude-3-7-sonnet-latest",
            api_key=config.secrets.ANTHROPIC_API_KEY,
        ),
    )
    "The main LLM model to use for completions"

    ## For longer persistence a database is required
    # conn = providers.Resource(AsyncSqliteConn, config.checkpoint_db)
    # conn = providers.Resource(AsyncPostgresConn, config.checkpoint_db)
    "Connection to the database for persistent storage"

    ## CHECKPOINTERS -- For persistence of langgraph runs
    checkpointer = providers.Resource(MemorySaver)
    # checkpointer = providers.Resource(AsyncSqliteSaver, conn=conn)
    # checkpointer = providers.Resource(AsyncPostgresSaver, conn=conn)
    # checkpointer = providers.Resource(AsyncShallowPostgresSaver, conn=conn)  # No timetravel
    "Persistence provider for langgraph runs (e.g. enables interrupt/resume)"

    store = providers.Resource(InMemoryStore)
    # No sqlite equivalent
    # store = providers.Resource(AsyncPostgresStore, conn=conn)
    "Persistence provider for langgraph data (e.g. enables persisting data between runs)"

    wiring_config = containers.WiringConfiguration(
        modules=[
            ".host_app",
            ".state",
            ".components.navbar",
        ],
        packages=[".graph"],
        auto_wire=False,
    )
    """Configures the dependency injection wiring for the application. I.e., which modules/packages
    require injections to be performed."""
