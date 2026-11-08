"""Read a finished graph back as the ordered run it was.

Every node (an agent is its root node) carries the ``seq`` it was added at,
so the timeline is the nodes in that order. ``replay`` yields the graph as it stood after each one; each
snapshot is a real :class:`Graph`, so anything that reads a graph works on a
replay without knowing it is one. Snapshots are rebuilt from the saved records,
so they can be collected or rendered out of order without touching the run.
Edits are applied in place, so a snapshot of an edited run shows the edited
graph cut at that point, not what it held before the edit.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node
from rlmflow.view.text import kind, render_tree

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
    """Every node in the graph, in the order they were added."""
    return g.nodes()


def summarize(item: Node, limit: int = 96) -> str:
    """A single line describing what this node did."""
    root = item.root_data
    if root is not None:
        config = root._config
        text = (
            f"start {root.name} · {root.config_name or 'root'} · "
            f"{getattr(config, 'harness', '?')} · model {getattr(config, 'model', '?')}"
        )
        return _clip(text, limit)
    match kind(item):
        case "message":
            text = f"from {item.producer}: {item.text}"
        case "done":
            text = f"done {_stringify(item.result)}"
        case "idle":
            text = "idle, waiting for messages"
        case "code":
            text = item.code or item.output
        case _:
            text = item.text or item.output
    return _clip(text, limit)


def node_detail(item: Node, limit: int = DETAIL_LIMIT) -> str:
    """The content worth reading for this node, trimmed."""
    blocks: list[str] = []
    if item.root_data is not None:
        blocks.append(f"task:\n{item.root_data.task}")
        args = [name for name in item.input if name != "task"]
        if args:
            blocks.append(f"args: {', '.join(args)}")
    else:
        if item.text:
            blocks.append(f"text:\n{item.text}")
        if item.input:
            blocks.append(f"input: {', '.join(item.input)}")
        if item.runs_code:
            blocks.append(f"code:\n{item.code}")
    if item.output:
        blocks.append(f"output:\n{item.output}")
    if item.result is not None:
        blocks.append(f"result:\n{_stringify(item.result)}")
    if item.error:
        blocks.append("error: yes")
    if item.metadata:
        blocks.append(f"metadata:\n{json.dumps(item.metadata, indent=2, default=str)}")
    text = "\n\n".join(blocks).strip()
    return text[:limit] + "\n… (truncated)" if len(text) > limit else text


def _clip(text: object, limit: int) -> str:
    text = " ".join(str(text).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


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
            node=item,
            agent=g.path(g.owner(item)),
            kind=kind(item),
            summary=summarize(item),
            detail=node_detail(item),
            elapsed=item.created_at - start,
        )
        for index, item in enumerate(ordered)
    ]


def as_graph(source: Graph | str | Path) -> Graph:
    """A graph, or the graph saved in a run directory."""
    if isinstance(source, Graph):
        return source
    return Graph.load(source)


def snapshot(g: Graph, count: int) -> Graph:
    """A copy of ``g`` holding its first ``count`` nodes, and the edges between them."""
    from rlmflow.graph.persistence import from_records, to_records

    ordered = timeline(g)
    if count >= len(ordered):
        return g.copy()
    cutoff = ordered[count].seq
    document = to_records(g)
    nodes = [record for record in document["nodes"] if record["seq"] < cutoff]
    kept = {record["id"] for record in nodes}
    for record in nodes:
        if "root" in record:
            record["root"]["history"] = [i for i in record["root"]["history"] if i in kept]
    edges = [
        record
        for record in document["edges"]
        if record["seq"] < cutoff and record["source"] in kept and record["target"] in kept
    ]
    return from_records({**document, "nodes": nodes, "edges": edges})


def replay(source: Graph | str | Path) -> Iterator[Graph]:
    """The run as a graph per step, from the first node to the whole thing.

    ::

        for snap in replay("runs/coding/graph"):
            print(render_tree(snap))
    """
    g = as_graph(source)
    for count in range(1, len(timeline(g)) + 1):
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
