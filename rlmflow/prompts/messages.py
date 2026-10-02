"""Current-node message rendering and the live tool list.

Keeps message shaping out of ``flow.py``. Canonical history projection lives on
``Node``; this module adds the system prompt and request-only child status.
"""

from __future__ import annotations

from typing import Any

from rlmflow.graph.nodes import Node, TruncationSummary
from rlmflow.tools.tools import format_tool_docs


def render_tools(flow: Any, node: Node) -> str:
    """The ``### Tools`` list for exactly the callables ``node``'s REPL binds."""
    return format_tool_docs(flow.build_tools(node))


def _append_to_last_user(
    current_messages: list[dict[str, str]],
    content: str,
) -> list[dict[str, str]]:
    """Add request-only context without changing a node's durable snapshot."""
    if not content:
        return current_messages
    current_messages = [dict(message) for message in current_messages]
    user_index = next(
        (
            index
            for index in range(len(current_messages) - 1, -1, -1)
            if current_messages[index]["role"] == "user"
        ),
        None,
    )
    if user_index is None:
        current_messages.append({"role": "user", "content": content})
    else:
        existing = current_messages[user_index]["content"].rstrip()
        current_messages[user_index]["content"] = (
            f"{existing}\n\n{content}" if existing else content
        )
    return current_messages


def build_messages(flow: Any, node: Node) -> list[dict[str, str]]:
    """The prompt as of ``node``: system message, then that agent's turns."""
    agent = node.parent_agent
    system_prompt = flow.build_system_prompt(node)
    if node.messages is None:
        user_prompt = flow.default_user_prompt(node)
        node.snapshot_messages(node.build_messages(user_prompt, flow))
    background = [
        f"- `{child.config.name}` ({child.config.path}): "
        + ("result ready" if child.terminal else "running")
        for child in agent.sub_agents
    ]
    current_messages = _append_to_last_user(
        node.message_dicts(),
        "\n".join(["Background subagents:", *background]) if background else "",
    )
    keep = agent.config.keep_n_messages
    if keep is not None:
        current_messages = current_messages[-keep:] if keep > 0 else []
    previous_keep = None if keep is None else max(keep - len(current_messages), 0)
    previous_messages = [] if node.prev is None else node.prev.project(keep=previous_keep)
    if keep is not None:
        summary = next(
            (item for item in node.iter_backwards() if isinstance(item, TruncationSummary)),
            None,
        )
        if summary is not None:
            notice = summary.message_dicts()
            visible = [*previous_messages, *current_messages]
            if any(message not in visible for message in notice):
                previous_messages = [*notice, *previous_messages]
    system = system_prompt.render()
    return [
        {"role": "system", "content": system},
        *previous_messages,
        *current_messages,
    ]


__all__ = [
    "build_messages",
    "render_tools",
]
