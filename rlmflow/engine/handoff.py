"""Deliver finished agents' results to the agents they report to.

An agent reports to the agent that launched it and to every agent it has a
``reports_to`` edge to, so one rule covers a recursive tree and a set of peers.
A ``DoneOutput`` is delivered once per recipient: the handoff message carries a
``from`` edge to it. Delivery waits until the recipient's turn has landed an
``ExecOutput``. A recipient that already finished is woken only by a result
that landed after it, and only when ``resume_on_child_done`` is on.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlmflow.graph import ops, reads
from rlmflow.graph.graph import DEPENDS_ON, FROM, REPORTS_TO, Graph
from rlmflow.graph.nodes import AgentStart, DoneOutput, ExecOutput, UserQuery

HANDOFF_RESULT_MAX = 4000


@dataclass(frozen=True)
class Handoff:
    path: str
    name: str
    result: Any


def marker(path: str) -> str:
    return f"[handoff {path}]"


def reporters(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """Agents whose results go to ``agent``: its children, then ``reports_to`` peers."""
    found = {child.id: child for child in reads.children(g, agent)}
    for edge in g.edges(kind=REPORTS_TO, target=agent.id):
        found.setdefault(edge.source, g.node(edge.source))
    return list(found.values())


def recipients(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """Agents that ``agent``'s results go to: its launcher, then ``reports_to`` targets."""
    launcher = reads.launcher(g, agent)
    found = {launcher.id: launcher} if launcher is not None else {}
    for edge in g.edges(kind=REPORTS_TO, source=agent.id):
        found.setdefault(edge.target, g.node(edge.target))
    return list(found.values())


def pending(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """Finished reporters whose result has not been sent to ``agent``."""
    return [
        reporter
        for reporter in reporters(g, agent)
        if isinstance(done := reads.frontier(g, reporter), DoneOutput)
        and done.agent_id == reporter.id
        and not _delivered(g, done, agent)
        and not g.edges(kind=DEPENDS_ON, target=reporter.id)
    ]


def _delivered(g: Graph, done: DoneOutput, agent: AgentStart) -> bool:
    return any(
        g.node(edge.source).agent_id == agent.id for edge in g.edges(kind=FROM, target=done.id)
    )


def react(g: Graph, agent: AgentStart) -> UserQuery | None:
    """Send one batched handoff message when ``agent``'s turn has landed.

    The caller guarantees ``agent`` has no step in flight.
    """
    frontier = reads.frontier(g, agent)
    if not isinstance(frontier, (ExecOutput, DoneOutput)):
        return None
    ready = pending(g, agent)
    if not ready:
        return None
    if isinstance(frontier, DoneOutput) and not (
        agent.config.resume_on_child_done
        and any(reads.frontier(g, reporter).seq > frontier.seq for reporter in ready)
    ):
        return None
    limit = min(HANDOFF_RESULT_MAX, agent.config.max_output_length)
    items = [
        Handoff(
            path=reads.path(g, reporter),
            name=reporter.config.name,
            result=reads.result(g, reporter),
        )
        for reporter in ready
    ]
    children = {child.id for child in reads.children(g, agent)}
    return ops.send(
        g,
        agent,
        _render(items, limit, children=all(reporter.id in children for reporter in ready)),
        sender=[reads.frontier(g, reporter) for reporter in ready],
    )


def _render(items: list[Handoff], limit: int, *, children: bool = True) -> str:
    parts = [
        "Child results are ready. Each block is that child's submit() value."
        if children
        else "Agent results are ready. Each block is that agent's submit() value."
    ]
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


__all__ = ["Handoff", "marker", "pending", "react", "recipients", "reporters"]
