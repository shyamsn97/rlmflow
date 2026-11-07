"""Read a finished graph back as the ordered run it was.

Every agent and node carries the ``seq`` it was added at, so the timeline is
both, in that order. ``replay`` yields the graph as it stood after each one; each
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
from rlmflow.graph.nodes import AGENT, Agent, Node
from rlmflow.view.text import render_tree

DETAIL_LIMIT = 1600


@dataclass
class Step:
    """One agent or node, with what it was and where it sits in the run."""

    index: int
    node: Agent | Node
    agent: str
    kind: str
    summary: str
    detail: str
    elapsed: float

    @property
    def title(self) -> str:
        return f"{self.agent} · {self.kind}"


def timeline(g: Graph) -> list[Agent | Node]:
    """Every agent and node in the graph, in the order they were added."""
    return sorted([*g.agents(), *g.nodes()], key=lambda item: item.seq)


def summarize(item: Agent | Node, limit: int = 96) -> str:
    """A single line describing what this agent or node did."""
    if isinstance(item, Agent):
        config = getattr(item, "config", None)
        text = (
            f"start {item.name} · {item.config_name or 'root'} · "
            f"{getattr(config, 'harness', '?')} · model {getattr(config, 'model', '?')}"
        )
        return _clip(text, limit)
    data = item.data
    match item.type:
        case "message":
            text = f"from {data.get('sender', 'user')}: {data.get('text', '')}"
        case "llm":
            text = data.get("text", "")
        case "code":
            text = data.get("code") or data.get("output", "")
        case "idle":
            text = "idle, waiting for messages"
        case "done":
            text = f"done {_stringify(data.get('result'))}"
        case _:
            text = ""
    return _clip(text, limit)


def node_detail(item: Agent | Node, limit: int = DETAIL_LIMIT) -> str:
    """The content worth reading for this agent or node, trimmed."""
    blocks: list[str] = []
    if isinstance(item, Agent):
        blocks.append(f"task:\n{item.task}")
        if item.args:
            blocks.append(f"args: {', '.join(item.args)}")
        data: dict = {}
    else:
        data = item.data
    for key in ("text", "code", "output"):
        if data.get(key):
            blocks.append(f"{key}:\n{data[key]}")
    if data.get("result") is not None:
        blocks.append(f"result:\n{_stringify(data['result'])}")
    if data.get("error") and isinstance(data["error"], str):
        blocks.append(f"error: {data['error']}")
    elif data.get("error"):
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
            agent=g.path(item if isinstance(item, Agent) else g.owner(item)),
            kind=AGENT if isinstance(item, Agent) else item.type,
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
    """A copy of ``g`` holding its first ``count`` agents and nodes, and the edges between them."""
    from rlmflow.graph.persistence import from_records, to_records

    ordered = timeline(g)
    if count >= len(ordered):
        return g.copy()
    cutoff = ordered[count].seq
    document = to_records(g)
    nodes = [record for record in document["nodes"] if record["seq"] < cutoff]
    kept = {record["id"] for record in nodes}
    agents = []
    for record in document["agents"]:
        if record["seq"] >= cutoff:
            continue
        record["nodes"] = [i for i in record["nodes"] if i in kept]
        record["inbox"] = [i for i in record["inbox"] if i in kept]
        if record.get("base") not in (None, *kept):
            record.pop("base")
        agents.append(record)
    ids = kept | {record["id"] for record in agents}
    edges = [
        record for record in document["edges"] if record["seq"] < cutoff and record["target"] in ids
    ]
    return from_records({**document, "agents": agents, "nodes": nodes, "edges": edges, "edits": []})


def replay(source: Graph | str | Path) -> Iterator[Graph]:
    """The run as a graph per step, from the first agent to the whole thing.

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
