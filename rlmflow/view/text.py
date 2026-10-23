"""The agent forest as plain text: one status line per agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rlmflow.graph import reads
from rlmflow.graph.graph import FROM, Graph
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    Node,
    UserQuery,
)


def top_level(graph: Graph) -> list[AgentStart]:
    """Agents nobody launched, in creation order."""
    return [agent for agent in graph.nodes(AgentStart) if reads.launcher(graph, agent) is None]


def launched_by(graph: Graph, node: Node) -> list[AgentStart]:
    """Agents whose ``from`` edge points at ``node``."""
    launched = (graph.node(edge.source) for edge in graph.edges(kind=FROM, target=node.id))
    return [agent for agent in launched if isinstance(agent, AgentStart)]


def render_forest(graphs: Iterable[Graph]) -> str:
    """Render one or more graphs in first-seen order."""
    return "\n\n".join(render_tree(graph) for graph in graphs)


def render_tree(graph: Graph) -> str:
    """Render each top-level agent's query and recursive status tree."""
    blocks = []
    for top in top_level(graph):
        lines = [_clip(top.content) or "query"]
        _append_agent(graph, lines, top, prefix="", last=True)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _append_agent(
    graph: Graph,
    lines: list[str],
    agent: AgentStart,
    *,
    prefix: str,
    last: bool,
) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph, agent)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = reads.children(graph, agent)
    for index, child in enumerate(children):
        _append_agent(graph, lines, child, prefix=child_prefix, last=index == len(children) - 1)


def _status(graph: Graph, agent: AgentStart) -> str:
    frontier = reads.frontier(graph, agent)
    turns = sum(isinstance(node, LLMOutput) for node in reads.history(graph, agent))
    suffix = f" ({turns} turns)" if turns else ""
    path = reads.path(graph, agent)

    if isinstance(frontier, DoneOutput):
        return f"{path}: done {_clip(frontier.result, limit=64)}{suffix}"
    if isinstance(frontier, ErrorOutput):
        return f"{path}: error {_clip(frontier.content, limit=64)}{suffix}"
    if isinstance(frontier, ExecAction):
        children = launched_by(graph, frontier)
        if children:
            active = sum(not reads.terminal(graph, child) for child in children)
            return f"{path}: children running {active}/{len(children)}{suffix}"
        return f"{path}: running code{suffix}"
    if isinstance(frontier, LLMOutput):
        return f"{path}: planning{suffix}"
    if isinstance(frontier, ExecOutput):
        return f"{path}: observed output{suffix}"
    if isinstance(frontier, UserQuery):
        return f"{path}: working{suffix}"
    return f"{path}: pending{suffix}"


def _clip(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["launched_by", "render_forest", "render_tree", "top_level"]
