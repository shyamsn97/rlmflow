"""The agent forest as plain text: one status line per agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlmflow.graph.graph import FROM, Graph
from rlmflow.graph.nodes import AGENT_START, CODE, DONE, LLM_OUTPUT, MESSAGE, WAITING, Node


def top_level(graph: Graph) -> list[Node]:
    """Agents nobody launched, in creation order."""
    return [agent for agent in graph.agents() if graph.launcher(agent) is None]


def launched_by(graph: Graph, node: Node) -> list[Node]:
    """Agents whose ``from`` edge points at ``node``."""
    launched = (graph.node(edge.source) for edge in graph.edges(kind=FROM, target=node.id))
    return [agent for agent in launched if agent.type == AGENT_START]


def render_forest(graphs: Iterable[Graph]) -> str:
    """Render one or more graphs in first-seen order."""
    return "\n\n".join(render_tree(graph) for graph in graphs)


def render_tree(graph: Graph) -> str:
    """Render each top-level agent's task and recursive status tree."""
    blocks = []
    for top in top_level(graph):
        lines = [_clip(top.task) or "task"]
        _append_agent(graph, lines, top, prefix="", last=True)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _append_agent(graph: Graph, lines: list[str], agent: Node, *, prefix: str, last: bool) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph, agent)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = graph.children(agent)
    for index, child in enumerate(children):
        _append_agent(graph, lines, child, prefix=child_prefix, last=index == len(children) - 1)


def _status(graph: Graph, agent: Node) -> str:
    frontier = graph.frontier(agent)
    turns = sum(node.type == LLM_OUTPUT for node in graph.own_history(agent))
    suffix = f" ({turns} turns)" if turns else ""
    path = graph.path(agent)

    if frontier.type == DONE:
        return f"{path}: done {_clip(frontier.data.get('result'), limit=64)}{suffix}"
    if frontier.type == WAITING:
        on = frontier.data.get("on") or []
        active = sum(not graph.terminal(graph.node(a)) for a in on if a in graph)
        return f"{path}: waiting on {active}/{len(on)}{suffix}"
    children = [c for c in graph.children(agent) if not graph.terminal(c)]
    if children:
        return f"{path}: children running {len(children)}{suffix}"
    if frontier.type == LLM_OUTPUT:
        return f"{path}: running code{suffix}"
    if frontier.type == CODE:
        return f"{path}: planning{suffix}"
    if frontier.type in (AGENT_START, MESSAGE):
        return f"{path}: working{suffix}"
    return f"{path}: pending{suffix}"


def _clip(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value if value is not None else "").strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["launched_by", "render_forest", "render_tree", "top_level"]
