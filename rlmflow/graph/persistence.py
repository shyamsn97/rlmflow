"""Save a graph as ``graph.json`` (its agents, nodes, edges and edits) plus argument blobs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlmflow.graph.config import RECURSE, transform_argument
from rlmflow.graph.graph import Graph, GraphError
from rlmflow.graph.nodes import NODE_TYPES, Agent, Edge, Node, strict_json

#: Bumped on any incompatible change; older formats are not read.
FORMAT_VERSION = 5
ARG_BLOB_THRESHOLD_BYTES = 64 * 1024
ARG_BLOB_REF_KEY = "$rlmflow_arg_blob"


def node_record(node: Node) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": node.id,
        "type": node.type,
        "agent": node.agent_id,
        "seq": node.seq,
        "created_at": isoformat(node.created_at),
        "data": strict_json(node.data, f"node {node.id!r} data"),
    }
    if node.metadata:
        record["metadata"] = strict_json(node.metadata, f"node {node.id!r} metadata")
    if node.started_at is not None and node.finished_at is not None:
        record["timing"] = {
            "started_at": isoformat(node.started_at),
            "finished_at": isoformat(node.finished_at),
            "duration_ms": round((node.finished_at - node.started_at) * 1000, 3),
        }
    return record


def agent_record(agent: Agent) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": agent.id,
        "name": agent.name,
        "task": agent.task,
        "args": strict_json(agent.args, f"agent {agent.name!r} args"),
        "config": agent.config_name,
        "seq": agent.seq,
        "created_at": isoformat(agent.created_at),
        "nodes": [node.id for node in agent.nodes],
        "inbox": [node.id for node in agent.inbox],
    }
    if agent.base is not None:
        record["base"] = agent.base.id
    if agent.metadata:
        record["metadata"] = strict_json(agent.metadata, f"agent {agent.name!r} metadata")
    return record


def edge_record(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "kind": edge.kind,
        "source": edge.source,
        "target": edge.target,
        "data": dict(edge.data),
        "seq": edge.seq,
    }


def to_records(graph: Graph) -> dict[str, Any]:
    """The graph as one JSON document."""
    with graph._lock:
        return {
            "format": FORMAT_VERSION,
            "id": graph.id,
            "agents": [agent_record(agent) for agent in graph.agents()],
            "nodes": [node_record(node) for node in graph.nodes()],
            "edges": [edge_record(edge) for edge in graph.edges()],
            "edits": strict_json(graph.edits, "edits"),
        }


def from_records(document: dict[str, Any]) -> Graph:
    """Rebuild a graph from ``to_records``."""
    graph = Graph(_text(document.get("id"), "graph id"))
    nodes: dict[str, Node] = {}
    for raw in _list(document, "nodes"):
        node = Node(_node_type(raw), **_dict(raw.get("data"), "node data"))
        node.id = _text(raw.get("id"), "node id")
        node.agent_id = _text(raw.get("agent"), f"node {node.id} agent")
        node.seq = int(raw.get("seq", -1))
        node.metadata = dict(raw.get("metadata") or {})
        node.created_at = _epoch(raw.get("created_at")) or node.created_at
        timing = raw.get("timing") or {}
        node.started_at = _epoch(timing.get("started_at"))
        node.finished_at = _epoch(timing.get("finished_at"))
        nodes[node.id] = node
    for raw in _list(document, "agents"):
        agent = Agent(
            str(raw.get("task") or ""),
            name=_text(raw.get("name"), "agent name"),
            args=_dict(raw.get("args"), "agent args"),
            config=raw.get("config"),
        )
        agent.id = _text(raw.get("id"), "agent id")
        agent.seq = int(raw.get("seq", -1))
        agent.metadata = dict(raw.get("metadata") or {})
        agent.created_at = _epoch(raw.get("created_at")) or agent.created_at
        for key, target in (("nodes", agent.nodes), ("inbox", agent.inbox)):
            for node_id in raw.get(key) or []:
                node = nodes.get(node_id)
                if node is None or node.agent_id != agent.id:
                    raise GraphError(f"agent {agent.id!r} lists unknown node {node_id!r}")
                target.append(node)
        if raw.get("base"):
            agent.base = nodes.get(raw["base"])
            if agent.base is None:
                raise GraphError(f"fork {agent.id!r} has unknown base {raw['base']!r}")
            graph._forks[agent.base.id].append(agent.id)
        graph._agents[agent.id] = agent
    placed = {node.id for agent in graph._agents.values() for node in (*agent.nodes, *agent.inbox)}
    if placed != set(nodes):
        raise GraphError("every node must be in exactly one agent's history or inbox")
    graph._nodes = {node.id: node for node in sorted(nodes.values(), key=lambda n: n.seq)}
    for raw in _list(document, "edges"):
        edge = Edge(
            _text(raw.get("kind"), "edge kind"),
            _text(raw.get("source"), "edge source"),
            _text(raw.get("target"), "edge target"),
            dict(raw.get("data") or {}),
            _text(raw.get("id"), "edge id"),
            int(raw.get("seq", -1)),
        )
        for end in (edge.source, edge.target):
            if end not in graph:
                raise GraphError(f"edge {edge.id!r} points at missing {end!r}")
        graph._index(edge)
    graph.edits = list(document.get("edits") or [])
    graph._seq = 1 + max(
        [-1, *(a.seq for a in graph._agents.values()), *(n.seq for n in nodes.values())]
        + [e.seq for e in graph._edges.values()]
    )
    return graph


class _ArgBlobStore:
    """Content-address large agent arguments once per saved graph."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def encode(self, value: Any) -> Any:
        def encode_string(_path: str, item: Any) -> Any:
            if not isinstance(item, str):
                return RECURSE
            encoded = item.encode("utf-8")
            if len(encoded) < ARG_BLOB_THRESHOLD_BYTES:
                return item
            digest = hashlib.sha256(encoded).hexdigest()
            self.path.mkdir(parents=True, exist_ok=True)
            blob_path = self.path / f"{digest}.txt"
            if not blob_path.exists():
                _write_text(blob_path, item)
            return {ARG_BLOB_REF_KEY: digest, "encoding": "utf-8", "bytes": len(encoded)}

        return transform_argument(value, encode_string, path="")


def _resolve_arg(value: Any, blob_path: Path) -> Any:
    def resolve_blob(_path: str, item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {ARG_BLOB_REF_KEY, "encoding", "bytes"}:
            digest = item.get(ARG_BLOB_REF_KEY)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid argument blob digest {digest!r}")
            path = blob_path / f"{digest}.txt"
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise ValueError(f"missing argument blob {digest}") from exc
            encoded = text.encode("utf-8")
            if len(encoded) != item.get("bytes") or hashlib.sha256(encoded).hexdigest() != digest:
                raise ValueError(f"argument blob {digest} failed integrity validation")
            return text
        return RECURSE

    return transform_argument(value, resolve_blob, path="")


def save_graph(graph: Graph, path: str | Path) -> Path:
    """Atomically write ``graph.json``, moving large arguments into ``blobs/``."""
    run = Path(path)
    run.mkdir(parents=True, exist_ok=True)
    document = to_records(graph)
    blobs = _ArgBlobStore(run / "blobs")
    for record in document["agents"]:
        record["args"] = blobs.encode(record["args"])
    _write_text(run / "graph.json", json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return run


def load_graph(path: str | Path) -> Graph:
    """Read a run directory written by ``save_graph``."""
    run = Path(path)
    try:
        document = json.loads((run / "graph.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{run / 'graph.json'} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{run / 'graph.json'} must contain a JSON object")
    version = document.get("format")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{run} was saved in run format {version or 1}, and this rlmflow reads only "
            f"format {FORMAT_VERSION}; rerun it with this version."
        )
    for record in _list(document, "agents"):
        record["args"] = {
            name: _resolve_arg(value, run / "blobs")
            for name, value in _dict(record.get("args"), "agent args").items()
        }
    return from_records(document)


def _node_type(raw: dict[str, Any]) -> str:
    kind = raw.get("type")
    if kind not in NODE_TYPES:
        raise ValueError(f"unknown node type {kind!r}")
    return kind


def _list(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = document.get(key)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"graph field {key!r} must be a list of objects")
    return items


def _dict(value: Any, what: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{what} must be an object")
    return dict(value)


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp).timestamp() if stamp else None


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


__all__ = [
    "ARG_BLOB_THRESHOLD_BYTES",
    "FORMAT_VERSION",
    "agent_record",
    "edge_record",
    "from_records",
    "load_graph",
    "node_record",
    "save_graph",
    "to_records",
]
