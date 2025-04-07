"""
Basic testing that the checkpointer is being provided correctly by the container.

Easy to get this wrong when switching from sync MemorySaver to async sqlite/postgres savers.
"""

from typing import AsyncIterator

import pytest

# from aiosqlite import Connection
from dependency_injector.wiring import Provide, inject
from langgraph.checkpoint.base import BaseCheckpointSaver

from host_app.containers import Application

# @inject
# async def get_conn(conn: Connection = Provide[Application.conn]) -> Connection:
#     return conn


@inject
async def checkpoint_getter(
    checkpointer: BaseCheckpointSaver = Provide[Application.checkpointer],
) -> BaseCheckpointSaver:
    assert isinstance(checkpointer, BaseCheckpointSaver)
    return checkpointer


@pytest.fixture(scope="session")
async def application() -> AsyncIterator[Application]:
    container = Application()
    container.wire(modules=[__name__])
    coro_or_none = container.init_resources()
    if coro_or_none:
        await coro_or_none
    yield container
    coro_or_none = container.shutdown_resources()
    if coro_or_none:
        await coro_or_none
    container.unwire()


# @pytest.mark.usefixtures("application")
# async def test_get_conn():
#     """Test that the connection is available in the container."""
#     conn = await get_conn()
#     assert isinstance(conn, Connection)


@pytest.mark.usefixtures("application")
async def test_get_checkpointer():
    """Test that the checkpointer is available in the container."""
    checkpointer = await checkpoint_getter()
    assert isinstance(checkpointer, BaseCheckpointSaver)
