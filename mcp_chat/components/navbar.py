from typing import Any

import reflex as rx
from dependency_injector.wiring import Provide, inject
from reflex_github_button import github_button

from mcp_chat.containers import Application
from mcp_chat.models import McpServerInfo, ToolInfo
from mcp_chat.state import State


def sidebar_chat(chat: str) -> rx.Component:
    """A sidebar chat item.

    Args:
        chat: The chat item.
    """
    return rx.drawer.close(
        rx.hstack(
            rx.button(
                chat,
                on_click=lambda: State.set_chat(chat),
                width="80%",
                variant="surface",
            ),
            rx.button(
                rx.icon(
                    tag="trash",
                    on_click=lambda: State.delete_chat(chat),
                    stroke_width=1,
                ),
                width="20%",
                variant="surface",
                color_scheme="red",
            ),
            width="100%",
        )
    )


def chat_history_sidebar() -> rx.Component:
    """The sidebar component."""
    trigger = rx.button(
        rx.icon(
            tag="messages-square",
            color=rx.color("mauve", 12),
        ),
        background_color=rx.color("mauve", 6),
    )
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Chats", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.foreach(State.chat_titles, lambda chat: sidebar_chat(chat)),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def new_chat_dialog() -> rx.Component:
    """A modal to create a new chat."""
    trigger = rx.button("+ New chat")

    return rx.dialog.root(
        rx.dialog.trigger(trigger),
        rx.dialog.content(
            rx.hstack(
                rx.input(
                    placeholder="Type something...",
                    on_blur=State.set_new_chat_name,
                    width=["15em", "20em", "30em", "30em", "30em", "30em"],
                ),
                rx.dialog.close(
                    rx.button(
                        "Create chat",
                        on_click=State.create_chat,
                    ),
                ),
                # background_color=rx.color("mauve", 1),
                spacing="2",
                width="100%",
            ),
        ),
    )


def options_dialog() -> rx.Component:
    trigger = rx.button(
        rx.icon(
            tag="sliders-horizontal",
            color=rx.color("mauve", 12),
        ),
        background_color=rx.color("mauve", 6),
    )
    return rx.dialog.root(
        rx.dialog.trigger(trigger),
        rx.dialog.content(
            rx.hstack(
                model_selection(),
                graph_mode_selection(),
                align="center",
            ),
        ),
    )


def connected_mcp_server_infos() -> rx.Component:
    def render_tool_info(tool_info: ToolInfo) -> rx.Component:
        return rx.hstack(
            rx.tooltip(
                rx.badge(tool_info.name),
                content=tool_info.description,
            ),
            spacing="1",
        )

    def render_mcp_server_info(server_info: McpServerInfo) -> rx.Component:
        return rx.card(
            rx.inset(rx.heading(server_info.name), padding="1em", side="top"),
            rx.data_list.root(
                rx.data_list.item(
                    rx.data_list.label("Tools"),
                    rx.data_list.value(
                        rx.flex(
                            rx.foreach(server_info.tools, render_tool_info),
                            wrap="wrap",
                            spacing="1",
                        )
                    ),
                )
            ),
        )

    return rx.dialog.root(
        rx.dialog.trigger(rx.button("MCP Servers")),
        rx.dialog.content(
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(State.mcp_servers, render_mcp_server_info),
                ),
                max_height="80vh",
                overflow_y="auto",
            ),
        ),
    )


def graph_mode_selection() -> rx.Component:
    return rx.hstack(
        "Graph mode:",
        rx.select(
            ["functional", "standard"],
            default_value=State.graph_mode,
            on_change=State.set_graph_mode,
            placeholder="Graph mode",
        ),
        align="center",
    )


@inject
def model_selection(
    llm_models: dict[str, Any] = Provide[Application.llm_models],
    default: str = Provide[Application.config.default_model],
) -> rx.Component:
    return rx.hstack(
        "Model:",
        rx.select(
            list(llm_models.keys()),
            default_value=default,
            on_change=State.set_model,
        ),
        align="center",
    )


def navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.link(
                    rx.avatar(fallback="TC", variant="solid"),
                    href="https://github.com/TimChild/",
                ),
                rx.heading("Reflex MCP Chat"),
                rx.desktop_only(
                    rx.badge(
                        State.current_chat,
                        rx.tooltip(
                            rx.icon("info", size=14),
                            content="The current selected chat.",
                        ),
                        variant="soft",
                    )
                ),
                align="center",
            ),
            rx.hstack(
                github_button("star", "TimChild", "reflex-mcp-chat", show_count=True),
                align="center",
            ),
            rx.hstack(
                connected_mcp_server_infos(),
                new_chat_dialog(),
                chat_history_sidebar(),
                options_dialog(),
                align="center",
            ),
            justify="between",
            align="center",
        ),
        padding="12px",
        border_bottom=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        width="100",
        # position="sticky",
        # top="0",
        align="center",
    )
