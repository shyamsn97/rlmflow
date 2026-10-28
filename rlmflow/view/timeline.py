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

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node
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
    data = node.data
    match node.type:
        case "agent_start":
            config = node.metadata.get("config") or {}
            name = node.config_name or "root"
            text = f"start {node.name} · {name} · {config.get('harness', '?')} · model {config.get('model', '?')}"
        case "message":
            text = f"from {data.get('sender', 'user')}: {data.get('text', '')}"
        case "llm_output":
            text = data.get("text", "")
        case "code":
            text = data.get("code") or data.get("output", "")
        case "waiting":
            text = f"waiting on {len(data.get('on') or [])} agents"
        case "done":
            text = f"done {_stringify(data.get('result'))}"
        case _:
            text = ""
    text = " ".join(str(text).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def node_detail(node: Node, limit: int = DETAIL_LIMIT) -> str:
    """The content worth reading for this node, trimmed."""
    data = node.data
    blocks: list[str] = []
    if node.type == "agent_start":
        blocks.append(f"task:\n{node.task}")
        if node.args:
            blocks.append(f"args: {', '.join(node.args)}")
    for key in ("text", "code", "output"):
        if data.get(key):
            blocks.append(f"{key}:\n{data[key]}")
    if node.type == "waiting" and data.get("on"):
        blocks.append(f"on: {', '.join(data['on'])}")
    if data.get("result") is not None:
        blocks.append(f"result:\n{_stringify(data['result'])}")
    if data.get("error") and isinstance(data["error"], str):
        blocks.append(f"error: {data['error']}")
    elif data.get("error"):
        blocks.append("error: yes")
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
            agent=g.path(g.node(node.agent_id)),
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
