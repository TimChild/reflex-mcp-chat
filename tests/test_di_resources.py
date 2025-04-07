"""Basic tests to understand how the dependency injection works with async resources."""

import asyncio
from typing import AsyncIterator, Iterator

import pytest
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject


def sync_resource() -> str:
    print("getting sync resource")
    return "some sync str"


async def async_resource() -> str:
    print("getting async resource")
    return "some async str"


def sync_g() -> Iterator[str]:
    print("getting sync g")
    yield "some sync g str"
    print("stopping sync g")


async def async_g() -> AsyncIterator[str]:
    print("getting async g")
    await asyncio.sleep(0)
    yield "some async g str"
    print("stopping async g")


def sync_using_async(value: str) -> str:
    print("sync_using_async")
    return f"sync_using_async:{value}"


class Container(containers.DeclarativeContainer):
    sync_f = providers.Resource(sync_resource)
    async_f = providers.Resource(async_resource)

    sync_g = providers.Resource(sync_g)
    async_g = providers.Resource(async_g)

    sync_depends_on_async = providers.Resource(sync_using_async, value=async_f)


@inject
def use_sync_resources(
    sync_f: str = Provide[Container.sync_f],
    sync_g: Iterator[str] = Provide[Container.sync_g],
) -> str:
    print("using sync resources")
    return f"{sync_f}:{sync_g}"


@inject
async def use_async_resources(
    async_f: str = Provide[Container.async_f],
    async_g: AsyncIterator[str] = Provide[Container.async_g],
) -> str:
    print("using async resources")
    return f"{async_f}:{async_g}"


@pytest.fixture(scope="session")
async def container() -> AsyncIterator[Container]:
    """Fixture to provide a container instance."""
    container = Container()
    container.wire(modules=[__name__])
    await container.init_resources()  # pyright: ignore[reportGeneralTypeIssues]
    yield container
    await container.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
    container.unwire()


@pytest.mark.usefixtures("container")
def test_sync_resources() -> None:
    """Test the sync resource and generator."""
    result = use_sync_resources()
    assert result == "some sync str:some sync g str"


@pytest.mark.usefixtures("container")
async def test_async_resources() -> None:
    """Test the async resource and generator."""
    result = await use_async_resources()
    assert result == "some async str:some async g str"


async def test_get_async_resource_value(container: Container) -> None:
    # NOTE: still requires await because of the async dependency
    v = await container.sync_depends_on_async()  # pyright: ignore[reportGeneralTypeIssues]
    assert v == "sync_using_async:some async str"
