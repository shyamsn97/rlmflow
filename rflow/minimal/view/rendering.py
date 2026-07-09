"""Event-stream consumers for minimal rflow terminal output.

The graph stays the durable source of truth. Renderers consume graph events and
read the current graph snapshot, matching Tau's "core emits, UI consumes" split.
"""

from __future__ import annotations

from rflow.minimal.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rflow.minimal.graph.events import Event


class LiveTreeRenderer:
    """Redraw a compact graph tree in response to graph events."""

    def __init__(self, *, clear: bool = True) -> None:
        self.clear = clear

    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is None:
            return
        if self.clear:
            print("\033[2J\033[H", end="")
        print(render_tree(graph))


def render_tree(graph: Graph) -> str:
    lines = [_clip(graph.query) or "root query"]
    _append_agent(lines, graph, prefix="", last=True)
    return "\n".join(lines)


def _append_agent(lines: list[str], graph: Graph, *, prefix: str, last: bool) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = list(graph.children.values())
    for i, child in enumerate(children):
        _append_agent(lines, child, prefix=child_prefix, last=i == len(children) - 1)


def _status(graph: Graph) -> str:
    current = graph.current()
    turns = sum(isinstance(node, LLMOutput) for node in graph.nodes)
    suffix = f" ({turns} turns)" if turns else ""
    if isinstance(current, DoneOutput):
        return f"{graph.agent_id}: done {_clip(current.result, limit=64)}{suffix}"
    if isinstance(current, ErrorOutput):
        return f"{graph.agent_id}: error {_clip(current.error or current.output, limit=64)}{suffix}"
    if isinstance(current, SupervisingOutput):
        return (
            f"{graph.agent_id}: waiting on {len(current.waiting_on)} children{suffix}"
        )
    if isinstance(current, ResumeAction):
        return f"{graph.agent_id}: resumed{suffix}"
    if isinstance(current, ExecAction):
        return f"{graph.agent_id}: running code{suffix}"
    if isinstance(current, LLMOutput):
        return f"{graph.agent_id}: planning{suffix}"
    if isinstance(current, ExecOutput):
        return f"{graph.agent_id}: observed output{suffix}"
    if isinstance(current, UserQuery):
        return f"{graph.agent_id}: pending"
    return f"{graph.agent_id}: pending"


def _clip(text: str, *, limit: int = 80) -> str:
    text = " ".join(text.strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


__all__ = [
    "LiveTreeRenderer",
    "render_tree",
]
