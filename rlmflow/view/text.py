"""The agent forest as plain text: one status line per agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import CODE, DONE, IDLE, LLM, Agent


def render_forest(graphs: Iterable[Graph]) -> str:
    """Render one or more graphs in first-seen order."""
    return "\n\n".join(render_tree(graph) for graph in graphs)


def render_tree(graph: Graph) -> str:
    """Render each top-level agent's task and recursive status tree."""
    blocks = []
    for top in graph.roots():
        lines = [_clip(top.task) or "task"]
        _append_agent(graph, lines, top, prefix="", last=True)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _append_agent(graph: Graph, lines: list[str], agent: Agent, *, prefix: str, last: bool) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph, agent)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = graph.children(agent)
    for index, child in enumerate(children):
        _append_agent(graph, lines, child, prefix=child_prefix, last=index == len(children) - 1)


def _status(graph: Graph, agent: Agent) -> str:
    history = graph.history(agent)
    frontier = history[-1] if history else None
    turns = sum(node.type == LLM for node in agent.nodes)
    suffix = f" ({turns} turns)" if turns else ""
    path = graph.path(agent)

    if graph.status(agent) == "done" and frontier is not None and frontier.type == DONE:
        return f"{path}: done {_clip(frontier.data.get('result'), limit=64)}{suffix}"
    if graph.status(agent) == IDLE:
        return f"{path}: idle{suffix}"
    children = [c for c in graph.children(agent) if graph.status(c) != "done"]
    if children:
        return f"{path}: children running {len(children)}{suffix}"
    if frontier is not None and frontier.type == LLM:
        return f"{path}: running code{suffix}"
    if frontier is not None and frontier.type == CODE:
        return f"{path}: planning{suffix}"
    return f"{path}: working{suffix}"


def _clip(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value if value is not None else "").strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["render_forest", "render_tree"]
