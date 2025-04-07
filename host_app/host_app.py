"""Initialize the Reflex app and dependecy injector container."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import reflex as rx
import reflex_chakra as rc
from reflex.config import environment

from host_app.components import chat, navbar
from host_app.state import State

from .containers import Application

logging.basicConfig(level=logging.DEBUG)

rx.Cookie


def index() -> rx.Component:
    """The main app."""
    return rc.vstack(
        navbar(),
        chat.chat(),
        chat.action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )


def check_secrets_not_null(secrets: dict[str, str]) -> None:
    """Check that all secrets specified contain a value."""
    for k, v in secrets.items():
        if not v:
            raise ValueError(f"Missing secret: {k}")


@asynccontextmanager
async def lifespan(container: Application) -> AsyncIterator[None]:
    """Lifespan function for the app.

    Before yield runs as app starts up, after yield runs as app shuts down.
    """
    coro_or_none = container.init_resources()
    if coro_or_none:
        await coro_or_none
    yield
    coro_or_none = container.shutdown_resources()
    if coro_or_none:
        await coro_or_none


def make_app() -> rx.App:
    # Add state and page to the app.
    container = Application()
    container.wire()
    check_secrets_not_null(container.config.secrets())

    app = rx.App(
        theme=rx.theme(
            appearance="dark",
            accent_color="indigo",
        ),
    )
    app.add_page(index, on_load=[State.on_load])
    app.register_lifespan_task(lifespan, container=container)
    return app


if env := environment.REFLEX_ENV_MODE.get():
    compile_context = environment.REFLEX_COMPILE_CONTEXT.get()
    print(f"Running in {env} mode. Compile context: {compile_context}")
    app = make_app()
