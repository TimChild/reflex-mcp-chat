import reflex as rx
import reflex_chakra as rc

from mcp_chat.components import loading_icon
from mcp_chat.models import ToolCallInfo, ToolsUse
from mcp_chat.state import QA, State

message_style = dict(
    display="inline-block",
    padding="1em",
    border_radius="8px",
    max_width=["30em", "30em", "50em", "50em", "50em", "50em"],
)


def render_qa(qa: QA) -> rx.Component:
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
            max_width="95%",
            margin_left="auto",
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
            max_width="95%",
            margin_right="auto",
        ),
        width="100%",
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.auto_scroll(
        rx.center(
            rx.box(
                rx.foreach(State.chats[State.current_chat], render_qa),
                max_width="50em",
            ),
            width="100%",
        ),
        flex="1",
        width="100%",
        padding_x="4px",
        padding_bottom="1em",
        align_self="center",
        overflow_y="auto",
        height="100%",
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
                            rx.text_area(
                                enter_key_submit=True,
                                id="question",
                                placeholder="Type something...",
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
                "Made with ❤️ by ",
                rx.link(
                    "Tim",
                    href="https://adventuresoftim.com",
                    is_external=True,
                    color=rx.color("mauve", 12),
                ),
                text_align="center",
                font_size=".75em",
                color=rx.color("mauve", 10),
            ),
            align_items="center",
        ),
        # position="sticky",
        # bottom="0",
        # left="0",
        padding_y="16px",
        border_top=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        align_items="stretch",
        width="100%",
    )
