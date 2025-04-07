import reflex as rx
import reflex_chakra as rc

from host_app.components import loading_icon
from host_app.models import ToolCallInfo, ToolsUse
from host_app.state import QA, State

message_style = dict(
    display="inline-block",
    padding="1em",
    border_radius="8px",
    max_width=["30em", "30em", "50em", "50em", "50em", "50em"],
)


def message(qa: QA) -> rx.Component:
    """A single question/answer message.

    Args:
        qa: The question/answer pair.

    Returns:
        A component displaying the question/answer pair.
    """

    def render_tool_use(tool_use: ToolsUse) -> rx.Component:
        def render_tool_call(tc: ToolCallInfo) -> rx.Component:
            return rx.tooltip(
                rx.badge(tc.name),
                content=f"Args: {rx.Var.create(tc.args).to_string()}",
            )

        return rx.box(
            rx.hstack(
                "Tools Called:",
                rx.foreach(tool_use.tool_calls, render_tool_call),
                align="center",
                wrap="wrap",
            ),
            padding="1em",
            border_radius="8px",
            text_align="left",
            color=rx.color("yellow", 12),
            background_color=rx.color("yellow", 4),
            margin_x="auto",
            margin_top="1em",
            width="90%",
        )

    return rx.box(
        rx.box(
            rx.markdown(
                qa.question,
                background_color=rx.color("mauve", 4),
                color=rx.color("mauve", 12),
                style=rx.Style(message_style),
            ),
            text_align="right",
            margin_top="1em",
        ),
        rx.cond(
            qa.tool_uses,
            rx.foreach(
                qa.tool_uses,
                render_tool_use,
            ),
        ),
        rx.box(
            rx.markdown(
                qa.answer,
                background_color=rx.color("accent", 4),
                color=rx.color("accent", 12),
                style=rx.Style(message_style),
            ),
            text_align="left",
            padding_top="1em",
        ),
        width="100%",
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.vstack(
        rx.box(rx.foreach(State.chats[State.current_chat], message), width="100%"),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.center(
        rx.vstack(
            rc.form(
                rc.form_control(
                    rx.hstack(
                        rx.vstack(
                            rx.cond(State.current_status, rx.text(State.current_status)),
                            rx.input(
                                rx.input.slot(
                                    rx.tooltip(
                                        rx.icon("info", size=18),
                                        content="Enter a question to get a response.",
                                    )
                                ),
                                placeholder="Type something...",
                                id="question",
                                width=["15em", "20em", "45em", "50em", "50em", "50em"],
                            ),
                            align="center",
                        ),
                        rx.button(
                            rx.cond(
                                State.processing,
                                loading_icon(height="1em"),
                                rx.text("Send"),
                            ),
                            type="submit",
                        ),
                        align_items="center",
                    ),
                    is_disabled=State.processing,
                ),
                on_submit=State.handle_send_click,
                reset_on_submit=True,
            ),
            rx.text(
                "ReflexGPT may return factually incorrect or misleading responses. Use discretion.",
                text_align="center",
                font_size=".75em",
                color=rx.color("mauve", 10),
            ),
            rx.logo(margin_top="-1em", margin_bottom="-1em"),
            align_items="center",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )
