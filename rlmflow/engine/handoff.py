"""Deliver a finished child's result to the agent that launched it.

The mailbox is live-only. On restore, rebuild it from terminal children whose
launcher transcript does not yet contain the matching ``[handoff {path}]``
marker. Flush only after that parent lands ``ExecOutput``, or after
``DoneOutput`` when ``resume_on_child_done`` is on. Never splice into an
in-flight step.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlmflow.graph.nodes import AgentStart, DoneOutput, ExecOutput, Node, UserQuery

HANDOFF_RESULT_MAX = 4000


@dataclass(frozen=True)
class Handoff:
    path: str
    name: str
    result: Any


def marker(path: str) -> str:
    return f"[handoff {path}]"


def supervisor(child: AgentStart) -> AgentStart | None:
    """The agent that launched ``child``, or ``None`` for a root."""
    parent_node = child.parent
    if parent_node is None:
        return None
    return parent_node.parent_agent


def delivered(agent: AgentStart, child: AgentStart) -> bool:
    token = marker(child.config.path)
    return any(type(node) is UserQuery and token in node.content for node in agent.transcript())


def enqueue(parent: AgentStart, child: AgentStart) -> None:
    """Stash one finished child until the parent is ready to observe it."""
    path = child.config.path
    if any(item.path == path for item in parent.pending_handoffs):
        return
    if delivered(parent, child):
        return
    parent.pending_handoffs.append(
        Handoff(path=path, name=child.config.name, result=child.result())
    )


def dependency_routed(child: AgentStart) -> bool:
    """Whether a handle-valued input routes this result to a consumer."""
    root = child.root
    if root is None:
        return False
    return any(
        source_id == child.id
        for agent in root.iter_agents()
        for _input_name, source_id in agent.input_references
    )


def flush(agent: AgentStart) -> bool:
    """Append one batched ``UserQuery`` when the parent's turn has landed.

    Returns True if a query was appended.
    """
    if not agent.pending_handoffs or agent.in_flight is not None:
        return False
    frontier = agent.frontier
    wake = isinstance(frontier, DoneOutput) and agent.config.resume_on_child_done
    if not isinstance(frontier, ExecOutput) and not wake:
        return False
    limit = min(HANDOFF_RESULT_MAX, agent.config.max_output_length)
    query = UserQuery(content=_render(agent.pending_handoffs, limit))
    agent.pending_handoffs.clear()
    frontier.append(query)
    return True


def apply(node: Node) -> list[Node]:
    """Enqueue a just-finished child and flush any parent that is ready.

    Returns the ``UserQuery`` leaves that should be submitted instead of the
    landed node, if the frontier moved.

    A child ``DoneOutput`` only flushes when the parent is already terminal
    (``resume_on_child_done``). Flushing an ``ExecOutput`` from that path
    would race the driver, which may already have submitted that observation.
    """
    extra: list[Node] = []
    if isinstance(node, DoneOutput):
        parent = supervisor(node.parent_agent)
        if parent is not None and not dependency_routed(node.parent_agent):
            enqueue(parent, node.parent_agent)
            if isinstance(parent.frontier, DoneOutput) and flush(parent):
                extra.append(parent.frontier)
    if isinstance(node, ExecOutput) and flush(node.parent_agent):
        extra.append(node.parent_agent.frontier)
    return extra


def restore(root: AgentStart) -> None:
    """Rebuild mailboxes from the tree and flush any parent already sitting idle."""
    for agent in root.iter_agents():
        agent.pending_handoffs.clear()
        for child in agent.sub_agents:
            if child.terminal and not dependency_routed(child):
                enqueue(agent, child)
        flush(agent)


def _render(items: list[Handoff], limit: int) -> str:
    parts = ["Child results are ready. Each block is that child's finish() value."]
    for item in items:
        body = _format_result(item.result, limit)
        parts.append(f"{marker(item.path)}\nname: {item.name}\n{body}")
    return "\n\n".join(parts)


def _format_result(value: Any, limit: int) -> str:
    text = _result_body(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _result_body(value: Any) -> str:
    """A mapping is its fields; anything else is JSON, labeled ``result``."""
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return "result: {}"
        lines: list[str] = []
        for key, item in value.items():
            rendered = _jsonish(item)
            label = str(key)
            if "\n" in rendered:
                lines.append(f"{label}:\n{textwrap.indent(rendered, '  ')}")
            else:
                lines.append(f"{label}: {rendered}")
        return "\n".join(lines)
    return f"result: {_jsonish(value)}"


def _jsonish(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    return json.dumps(value, default=str, ensure_ascii=False)


__all__ = [
    "HANDOFF_RESULT_MAX",
    "Handoff",
    "apply",
    "delivered",
    "dependency_routed",
    "enqueue",
    "flush",
    "marker",
    "restore",
    "supervisor",
]
