"""Tests that the graph part of the app works correctly."""

import asyncio

import pytest

from host_app.containers import Application, config_option_to_connections
from host_app.mcp_client import MultiMCPClient


async def test_mcp_client_with_missing_server(
    container: Application, example_server_config: dict, missing_stdio_server_config: dict
):
    """Check that client doesn't hang on missing server."""
    with container.config.mcp_servers.override(
        {
            "example_server": example_server_config,
            "missing_server": missing_stdio_server_config,
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
