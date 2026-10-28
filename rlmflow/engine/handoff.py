"""Deliver finished agents' results to the agents they report to.

An agent reports to the agent that created it and to every agent it has a
``reports_to`` edge to, so one rule covers a recursive tree and a set of peers.
A ``done`` node is delivered once per recipient: the handoff ``message`` carries
a ``from`` edge to it. Rules by the recipient's frontier:

- ``waiting``: one batched message once every agent in ``on`` is done.
- ``done``: nothing; done is final. An explicit ``send`` still reopens it.
- ``llm_output``: nothing yet; its cell hasn't run.
- otherwise: the results that are ready, except ones the agent already
  awaited inside a cell.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping
from typing import Any

from rlmflow.graph.graph import DEPENDS_ON, FROM, REPORTS_TO, Graph
from rlmflow.graph.nodes import CODE, DONE, LLM_OUTPUT, WAITING, Node

HANDOFF_RESULT_MAX = 4000
#: ``code`` data listing the agents whose results the cell awaited.
AWAITED_KEY = "awaited"


def marker(path: str) -> str:
    return f"[handoff {path}]"


def reporters(g: Graph, agent: Node) -> list[Node]:
    """Agents whose results go to ``agent``: those it launched, then ``reports_to`` peers."""
    found = {child.id: child for child in g.launched(agent)}
    for edge in g.edges(kind=REPORTS_TO, target=agent.id):
        found.setdefault(edge.source, g.node(edge.source))
    return list(found.values())


def recipients(g: Graph, agent: Node) -> list[Node]:
    """Agents that ``agent``'s results go to: its launcher and the forks whose history
    launched it too, then ``reports_to`` targets."""
    found: dict[str, Node] = {}
    launches = g.edges(kind=FROM, source=agent.id)
    if launches:
        launcher = g.node(g.node(launches[0].target).agent_id)
        found[launcher.id] = launcher
        for fork in g.agents():
            if fork.id in found or g.prev(fork) is None:
                continue
            if any(child.id == agent.id for child in g.launched(fork)):
                found[fork.id] = fork
    for edge in g.edges(kind=REPORTS_TO, source=agent.id):
        found.setdefault(edge.target, g.node(edge.target))
    return list(found.values())


def pending(g: Graph, agent: Node) -> list[Node]:
    """Finished reporters whose result has not been sent to, or awaited by, ``agent``."""
    awaited = {
        agent_id
        for node in g.nodes(CODE, agent=agent.id)
        for agent_id in node.data.get(AWAITED_KEY, ())
    }
    return [
        reporter
        for reporter in reporters(g, agent)
        if (done := g.frontier(reporter)).type == DONE
        and done.agent_id == reporter.id
        and reporter.id not in awaited
        and not _delivered(g, done, agent)
        and not g.edges(kind=DEPENDS_ON, target=reporter.id)
    ]


def _delivered(g: Graph, done: Node, agent: Node) -> bool:
    return any(
        g.node(edge.source).agent_id == agent.id for edge in g.edges(kind=FROM, target=done.id)
    )


def react(g: Graph, agent: Node) -> Node | None:
    """Send one batched handoff message if ``agent`` can take results now.

    The caller guarantees ``agent`` has no step in flight.
    """
    frontier = g.frontier(agent)
    if frontier.type in (DONE, LLM_OUTPUT):
        return None
    if frontier.type == WAITING and not all(
        g.terminal(g.node(agent_id)) for agent_id in frontier.data.get("on", ())
    ):
        return None
    ready = pending(g, agent)
    if not ready:
        return None
    limit = min(HANDOFF_RESULT_MAX, agent.config.max_output_length)
    children = {child.id for child in g.launched(agent)}
    text = render(
        [(g.path(r), r.config_name or "root", g.result(r)) for r in ready],
        limit,
        children=all(reporter.id in children for reporter in ready),
    )
    return g.send(agent, text, sender=[g.frontier(r) for r in ready])


def render(items: list[tuple[str, str, Any]], limit: int, *, children: bool = True) -> str:
    """``(path, config name, result)`` per finished agent, as one message."""
    parts = [
        "Agents you started have finished. Each block is that agent's result."
        if children
        else "Agent results are ready. Each block is that agent's result."
    ]
    for path, config_name, result in items:
        parts.append(f"{marker(path)}\nconfig: {config_name}\n{_format_result(result, limit)}")
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
    if isinstance(value, Mapping):
        if not value:
            return "result: {}"
        lines: list[str] = []
        for key, item in value.items():
            rendered = _jsonish(item)
            if "\n" in rendered:
                lines.append(f"{key}:\n{textwrap.indent(rendered, '  ')}")
            else:
                lines.append(f"{key}: {rendered}")
        return "\n".join(lines)
    return f"result: {_jsonish(value)}"


def _jsonish(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    return json.dumps(value, default=str, ensure_ascii=False)


__all__ = ["AWAITED_KEY", "marker", "pending", "react", "recipients", "render", "reporters"]
