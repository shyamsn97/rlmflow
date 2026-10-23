"""Read a finished graph back as the ordered run it was.

The graph log is the order a run happened in, so the timeline is its nodes in
``seq`` order. ``replay`` yields the graph as it stood after each node landed;
each snapshot is a real :class:`Graph`, so anything that reads a graph works on
a replay without knowing it is one. Snapshots copy nodes shallowly, so they can
be collected, compared, or rendered out of order without touching the run they
came from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rlmflow.graph import reads
from rlmflow.graph.config import NATIVE_BACKEND
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    AgentStart,
    BackendRun,
    Node,
    TransitionDecision,
    VerificationFailure,
)
from rlmflow.view.text import render_tree

DETAIL_LIMIT = 1600


@dataclass
class Step:
    """One node, with what it was and where it sits in the run."""

    index: int
    node: Node
    agent: str
    kind: str
    summary: str
    detail: str
    elapsed: float

    @property
    def title(self) -> str:
        return f"{self.agent} · {self.kind}"


def timeline(g: Graph) -> list[Node]:
    """Every node in the graph, in the order it was added."""
    return g.nodes()


def summarize(node: Node, limit: int = 96) -> str:
    """A single line describing what this node did."""
    if isinstance(node, AgentStart):
        executor = (
            f"backend {node.config.backend}"
            if node.config.backend != NATIVE_BACKEND
            else f"model {node.config.model}"
        )
        text = f"start {node.config.name} · {executor}"
    elif isinstance(node, BackendRun):
        text = f"invoke {node.backend} · {node.run_id}"
    elif isinstance(node, TransitionDecision):
        choice = (
            node.candidates[node.selected_rule_id]
            if node.selected_rule_id is not None
            else "continue"
        )
        text = f"selected {choice}"
    elif isinstance(node, VerificationFailure):
        text = f"{node.kind} rejected: {node.feedback}"
    elif node.type == "done_output":
        text = f"done {_stringify(getattr(node, 'result', None) or node.content)}"
    elif node.type == "error_output":
        text = f"{getattr(node, 'error', 'error')}: {node.content}"
    else:
        text = getattr(node, "code", "") or node.content or ""
    text = " ".join(str(text).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def node_detail(node: Node, limit: int = DETAIL_LIMIT) -> str:
    """The content worth reading for this node, trimmed."""
    blocks: list[str] = []
    if isinstance(node, AgentStart):
        blocks.append(f"query:\n{node.content}")
        if node.args:
            blocks.append(f"args: {', '.join(node.args)}")
    code = getattr(node, "code", "")
    if code:
        blocks.append(f"code:\n{code}")
    if node.content and not isinstance(node, AgentStart):
        blocks.append(f"content:\n{node.content}")
    if isinstance(node, TransitionDecision):
        choices = "\n".join(
            f"- {rule_id}: {condition}" for rule_id, condition in node.candidates.items()
        )
        blocks.append(
            f"choices:\n{choices}\n"
            f"selected: {node.selected_rule_id or 'continue'}\n"
            f"read set: {', '.join(node.read_set_node_ids)}"
        )
    if isinstance(node, VerificationFailure):
        blocks.append(
            f"kind: {node.kind}\ncandidate: {_stringify(node.candidate)}\nfeedback: {node.feedback}"
        )
    result = getattr(node, "result", None)
    if result is not None:
        blocks.append(f"result:\n{_stringify(result)}")
    error = getattr(node, "error", "")
    if error:
        blocks.append(f"error: {error}")
    if node.metadata:
        blocks.append(f"metadata:\n{json.dumps(node.metadata, indent=2, default=str)}")
    text = "\n\n".join(blocks).strip()
    return text[:limit] + "\n… (truncated)" if len(text) > limit else text


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else repr(value)


def steps(g: Graph) -> list[Step]:
    """The run as a list of steps, ready to render."""
    ordered = timeline(g)
    start = ordered[0].created_at if ordered else 0.0
    return [
        Step(
            index=index,
            node=node,
            agent=reads.path(g, g.node(node.agent_id)),
            kind=node.type,
            summary=summarize(node),
            detail=node_detail(node),
            elapsed=node.created_at - start,
        )
        for index, node in enumerate(ordered)
    ]


def as_graph(source: Graph | str | Path) -> Graph:
    """A graph, or the graph saved in a run directory."""
    if isinstance(source, Graph):
        return source
    return Graph.load(source)


def snapshot(g: Graph, count: int) -> Graph:
    """A copy of ``g`` holding its first ``count`` nodes and the edges they need."""
    starts = [item.seq for item in g.items() if isinstance(item, Node)]
    return g.as_of(starts[count] - 1 if count < len(starts) else len(g) - 1)


def replay(source: Graph | str | Path) -> Iterator[Graph]:
    """The run as a graph per step, from the first node to the whole thing.

    ::

        for snap in replay("runs/coding/graph"):
            print(render_tree(snap))
    """
    g = as_graph(source)
    for count in range(1, len(g.nodes()) + 1):
        yield snapshot(g, count)


def render_steps(source: Graph | str | Path) -> Iterator[str]:
    """Each step as the ASCII tree, ready to print."""
    for snap in replay(source):
        yield render_tree(snap)


__all__ = [
    "Step",
    "as_graph",
    "node_detail",
    "render_steps",
    "replay",
    "snapshot",
    "steps",
    "summarize",
    "timeline",
]
