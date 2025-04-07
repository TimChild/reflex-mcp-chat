import asyncio
import json
import logging
import uuid
from typing import Any, AsyncContextManager, cast

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    SSEConnection,
    StdioConnection,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, InitializeResult, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client


class MCPServerConnectionError(Exception):
    pass


class LCClientPatch(MultiServerMCPClient):
    initialize_timeout_s: float = 5

    async def __aenter__(self) -> "LCClientPatch":
        """Connect to all servers during context."""
        await super().__aenter__()
        return self

    # added timeout on intiaializing a session
    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        try:
            # raise Exception
            await asyncio.wait_for(session.initialize(), timeout=self.initialize_timeout_s)
            # NOTE: The problem is that this may only get the timeout error.
            #  The actual error ends up only getting caught in the exit stack
            #  but there I can't know which server it was for. (my PR to mcp may help with this)
        except Exception as e:
            logging.error(f"Failed to initialize session for {server_name}: {e}")
            return

        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools


ErroredServers = dict[str, tuple[SSEConnection | StdioConnection, Exception]]


class MultiMCPClient:
    def __init__(self, connections: dict[str, SSEConnection | StdioConnection]) -> None:
        """Initializes an adapter for multiple mcp clients.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be either a StdioConnection or SSEConnection.
        """
        self.connections = connections
        self.lc_client: LCClientPatch = LCClientPatch(connections=connections)
        self._context_depth = 0
        self.timeout = 1
        self.errored_servers: ErroredServers = {}

    async def ping_servers(self) -> dict[str, Exception]:
        async def send_ping(
            client_context_manager: AsyncContextManager,
        ) -> InitializeResult:
            async with client_context_manager as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    await session.send_ping()
                    return init

        errors: dict[str, Exception] = {}
        for server_name, connection in self.connections.items():
            try:
                if connection["transport"] == "stdio":
                    params = StdioServerParameters(
                        command=connection["command"], args=connection["args"]
                    )
                    await asyncio.wait_for(
                        send_ping(stdio_client(params)),
                        timeout=self.timeout,
                    )
                if connection["transport"] == "sse":
                    await asyncio.wait_for(
                        send_ping(sse_client(url=connection["url"])),
                        timeout=self.timeout,
                    )
            except Exception as e:
                errors[server_name] = e
        return errors

    async def check_connections(self) -> None:
        """Simple short lived connection to check servers are accessible.

        Returns:
            dict[str, Exception]: A dictionary mapping server names to any exceptions raised during connection.
        """
        errors = await self.ping_servers()
        if errors:
            for server_name, error in errors.items():
                logging.error(
                    f"Failed to connect to {server_name}: {error} -- Removing from connections"
                )
                conn = self.connections.pop(server_name)
                self.errored_servers[server_name] = (conn, error)

    async def __aenter__(self) -> "MultiMCPClient":
        """Connects to all servers during context."""
        if self._context_depth < 0:
            raise RuntimeError("Context manager has already exited")
        if self._context_depth == 0:
            await self.check_connections()
            self.lc_client = await self.lc_client.__aenter__()
        self._context_depth += 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Closes all server connections."""
        if self._context_depth <= 0:
            raise RuntimeError("Context manager has already exited")
        self._context_depth -= 1
        if self._context_depth == 0:
            await self.lc_client.__aexit__(exc_type, exc_value, traceback)

    async def get_tools(self) -> list[StructuredTool]:
        """Get all tools available from all connected servers."""
        # NOTE: lc loads on initial connection, so don't need to await here (in general it would be awaited though)
        async with self:
            tools = self.lc_client.get_tools()
        assert all(isinstance(tool, StructuredTool) for tool in tools)
        return cast(list[StructuredTool], tools)

    async def get_tools_by_server(self) -> dict[str, list[StructuredTool]]:
        """Get tools as dict of server name to tools."""
        async with self:
            for all_tools in self.lc_client.server_name_to_tools.values():
                assert all(isinstance(tool, StructuredTool) for tool in all_tools)
            return cast(dict[str, list[StructuredTool]], self.lc_client.server_name_to_tools)

    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:  # noqa: ANN401, ANN003
        """Manually call a tool on a specific server.

        Typically, the tool call will be made via the StructuredTool.func/coroutine methods (assuming the
        tool is used within the same mcp client session as when they were loaded).

        Returns whatever the tool returns.
        """
        if server_name not in self.lc_client.server_name_to_tools:
            if server_name in self.errored_servers:
                raise MCPServerConnectionError(
                    f"Server {server_name} failed to connect {self.errored_servers[server_name]}"
                )
            raise ValueError(f"Server {server_name} not in connected servers")
        async with self:
            server_tools = self.lc_client.server_name_to_tools[server_name]
            tool = next(t for t in server_tools if t.name == tool_name)
            assert isinstance(tool, StructuredTool)
            assert tool.coroutine is not None
            tool_call = ToolCall(
                name=tool_name,
                args=kwargs,
                id=str(uuid.uuid4()),
            )
            tool_content = await tool.ainvoke(tool_call)
            try:
                return json.loads(tool_content)
            except json.JSONDecodeError:
                return tool_content

    def set_connection_timeout(self, timeout_s: float) -> None:
        """Set the timeout for initializing a session."""
        self.lc_client.initialize_timeout_s = timeout_s
