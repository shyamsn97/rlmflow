"""The agent forest as plain text: one status line per agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import DONE, Agent, Node


def kind(node: Node) -> str:
    """What a node is, for display: ``agent``, ``message``, ``done``, ``idle``, ``model``, ``code`` or ``reply``."""
    if isinstance(node, Agent):
        return "agent"
    if node.role == "user":
        return "message"
    if node.data.get("terminal"):
        return "done"
    if node.data.get("idle"):
        return "idle"
    if "model" in node.data:
        return "model"
    if node.runs_code:
        return "code"
    return "reply"


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


def _append_agent(graph: Graph, lines: list[str], agent: Node, *, prefix: str, last: bool) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph, agent)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = agent.children()
    for index, child in enumerate(children):
        _append_agent(graph, lines, child, prefix=child_prefix, last=index == len(children) - 1)


def _status(graph: Graph, agent: Node) -> str:
    frontier = agent.nodes()[-1]
    turns = sum("seen" in node.data for node in agent.nodes())
    suffix = f" ({turns} turns)" if turns else ""
    path = agent.path()
    status = agent.status()
    if status == "done":
        return f"{path}: done {_clip(agent.final_result(), limit=64)}{suffix}"
    if status == "idle":
        return f"{path}: idle{suffix}"
    children = [c for c in agent.children() if c.status() != "done"]
    if children:
        return f"{path}: children running {len(children)}{suffix}"
    if frontier.state != DONE and frontier.runs_code:
        return f"{path}: running code{suffix}"
    if frontier.runs_code:
        return f"{path}: planning{suffix}"
    return f"{path}: working{suffix}"


def _clip(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value if value is not None else "").strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["kind", "render_forest", "render_tree"]
