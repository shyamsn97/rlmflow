"""Current-node rendering and input manifests.

Keeps message shaping out of ``flow.py``. Plan, final-answer, truncation, and a
dead REPL are node types; this module re-exports their default strings and holds
dynamic manifests. Canonical history projection lives on ``Node``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    FINAL_ANSWER_ACTION,
    TRUNCATION_SUMMARY,
    AgentStart,
    Node,
    TruncationSummary,
    TurnMode,
    UserQuery,
)
from rlmflow.tools.tools import format_tool_docs

if TYPE_CHECKING:
    from rlmflow.runtime import Runtime

RenderFn = Callable[["Runtime", Node], list[dict[str, str]]]


def build_background_agents_manifest(agent: AgentStart) -> str:
    """Render direct children and current result readiness from the graph."""
    children = agent.sub_agents
    if not children:
        return ""

    lines = ["Background subagents:"]
    for child in children:
        state = "result ready" if child.terminal else "running"
        lines.append(f"- `{child.config.name}` ({child.config.path}): {state}")
    return "\n".join(lines)


def default_render(_runtime: Runtime, node: Node) -> list[dict[str, str]]:
    """Render the frontier and add live background-agent status."""
    messages = node.render()
    agent = node.parent_agent
    if agent is None:
        return messages
    manifest = build_background_agents_manifest(agent)
    if not manifest:
        return messages
    return [
        {"role": "user", "content": manifest},
        *messages,
    ]


def build_inputs_manifest(inputs: dict[str, str]) -> str:
    """List inputs by name and size without copying their values into the prompt.

    ``INPUTS`` is the RLM context boundary: the model should inspect values
    programmatically in the persistent REPL, not receive a second copy in its
    context window. Returns ``""`` when there are no inputs.
    """
    if not inputs:
        return ""
    lines = [f"- {name}: str, {len(value)} chars" for name, value in inputs.items()]
    total = sum(len(value) for value in inputs.values())
    return (
        "Your REPL `INPUTS` contain important information related to the query:\n"
        + "\n".join(lines)
        + f"\nTotal input chars: {total}."
    )


def format_transition_footer(
    options: list[tuple[str, str]],
    *,
    finish_description: str = "Submit your final answer.",
    final: bool = False,
) -> str:
    """Render continuation, optional behavior choices, and completion."""
    if final:
        return f"End with finish(answer) — {finish_description}"
    lines = ["End the REPL block normally to continue."]
    if options:
        lines.append("Or transition immediately (stdout from that block is not observed):")
        lines.extend(f'- transition("{name}") — {description}' for name, description in options)
    lines.append(f"- finish(answer) — {finish_description}")
    return "\n".join(lines)


def transition_footer(flow: Any, node: Node) -> str:
    behavior = flow.transitions.current_behavior(node)
    owner = (
        node
        if node.turn_mode is TurnMode.FINAL
        else behavior or (node if isinstance(node, UserQuery) else None)
    )
    finish_description = getattr(
        owner,
        "finish_description",
        UserQuery.finish_description,
    )
    if node.turn_mode is TurnMode.FINAL:
        return format_transition_footer(
            [],
            finish_description=finish_description,
            final=True,
        )
    if node.turn_mode is not TurnMode.ACTION:
        return ""
    options = [(option.name, option.description) for option in flow.transitions.available(node)]
    return format_transition_footer(
        options,
        finish_description=finish_description,
    )


def render_tools(flow: Any, node: Node) -> str:
    """Live ``Available in the REPL`` list: gated builtins plus host tools."""
    from rlmflow.prompts.prompts import as_agent

    parts: list[str] = []
    docs = format_tool_docs(
        flow.toolsets,
        flow.tools,
        flow=flow,
        node=node,
    ).strip()
    if docs:
        parts.append(docs)
    agent = as_agent(node)
    clients = getattr(flow, "_llm_clients", {})
    if (
        node.turn_mode is not TurnMode.FINAL
        and clients
        and (can_spawn(agent) or flow.use_llm_query)
    ):
        lines = ["`model=` on `launch_subagent` / `llm_query` is required:"]
        for key in sorted(clients):
            tags: list[str] = []
            if agent is not None and key == agent.config.model:
                tags.append("current model")
            if key == flow.delegation_model:
                tags.append("default for delegated work")
            suffix = f" — {'; '.join(tags)}" if tags else ""
            lines.append(f"- `{key}`{suffix}")
        parts.append("\n".join(lines))
    profiles = flow.prompt_profiles
    if (
        node.turn_mode is not TurnMode.FINAL
        and profiles
        and flow.prompt_router is None
        and can_spawn(agent)
    ):
        lines = ["Available prompt profiles (pass `prompt_profile` to `launch_subagent`):"]
        for name in sorted(profiles):
            desc = getattr(profiles[name], "description", "")
            lines.append(f"- `{name}`" + (f" — {desc}" if desc else ""))
        parts.append("\n".join(lines))
    return "\n\n".join(part for part in parts if part)


def build_messages(flow: Any, node: Node) -> list[dict[str, str]]:
    """The prompt as of ``node``: system message, then that agent's turns."""
    agent = node.parent_agent
    profile = flow.profile(agent)
    render_fn = profile.render_fn or flow.render_fn
    current_messages = render_fn(flow.runtime, node)
    keep = agent.config.keep_n_messages
    if keep is not None:
        current_messages = current_messages[-keep:] if keep > 0 else []
    footer = flow.transition_footer(node)
    if footer:
        current_messages = list(current_messages)
        user_index = next(
            (
                index
                for index in range(len(current_messages) - 1, -1, -1)
                if current_messages[index]["role"] == "user"
            ),
            None,
        )
        if user_index is None:
            current_messages.append({"role": "user", "content": footer})
        else:
            message = dict(current_messages[user_index])
            content = message.get("content", "").rstrip()
            message["content"] = f"{content}\n\n{footer}" if content else footer
            current_messages[user_index] = message
    previous_keep = None if keep is None else max(keep - len(current_messages), 0)
    previous_messages = [] if node.prev is None else node.prev.project(keep=previous_keep)
    if keep is not None:
        summary = next(
            (item for item in node.iter_backwards() if isinstance(item, TruncationSummary)),
            None,
        )
        if summary is not None:
            notice = summary.render()
            visible = [*previous_messages, *current_messages]
            if any(message not in visible for message in notice):
                previous_messages = [*notice, *previous_messages]
    system = (
        node.build_system_prompt(flow)
        if isinstance(node, UserQuery)
        else flow.build_system_prompt(node)
    )
    return [
        {"role": "system", "content": system},
        *previous_messages,
        *current_messages,
    ]


__all__ = [
    "COLD_REPL_NOTE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "RenderFn",
    "build_background_agents_manifest",
    "build_inputs_manifest",
    "build_messages",
    "format_transition_footer",
    "default_render",
    "render_tools",
    "transition_footer",
]
