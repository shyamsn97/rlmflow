"""Save a graph as ``graph.json`` (its nodes and edges) plus blobs for large inputs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlmflow.graph.config import RECURSE, transform_argument
from rlmflow.graph.graph import Graph, GraphError
from rlmflow.graph.nodes import DONE, READY, RUNNING, Agent, Edge, Node, strict_json

#: Bumped on any incompatible change. Format 6 is read too: its agents' ``history``
#: lists are ignored (``owner`` and ``seq`` give the same order).
FORMAT_VERSION = 7
READS = (6, 7)
BLOB_THRESHOLD_BYTES = 64 * 1024
BLOB_REF_KEY = "$rlmflow_blob"


def node_record(node: Node) -> dict[str, Any]:
    what = f"node {node.id!r}"
    record: dict[str, Any] = {
        "id": node.id,
        "owner": node.owner,
        "seq": node.seq,
        "version": node.version,
        "role": node.role,
        "producer": node.producer,
        "text": node.text,
        "input": strict_json(node.input, f"{what} input"),
        "code": node.code,
        "state": node.state,
        "created_at": isoformat(node.created_at),
    }
    if node.calls:
        record["calls"] = strict_json(node.calls, f"{what} calls")
    if node.output:
        record["output"] = node.output
    if node.result is not None:
        record["result"] = strict_json(node.result, f"{what} result")
    if node.error:
        record["error"] = True
    record["data"] = strict_json(node.data, f"{what} data")
    if node.metadata:
        record["metadata"] = strict_json(node.metadata, f"{what} metadata")
    if node.started_at is not None and node.finished_at is not None:
        record["timing"] = {
            "started_at": isoformat(node.started_at),
            "finished_at": isoformat(node.finished_at),
            "duration_ms": round((node.finished_at - node.started_at) * 1000, 3),
        }
    if isinstance(node, Agent):
        record["agent"] = {
            "name": node.name,
            "task": node.task,
            "config": node.config_name,
            "arg_descriptions": dict(node.arg_descriptions),
            "repl": node.repl_id,
        }
        if node.config_record is not None:
            record["agent"]["config_record"] = strict_json(node.config_record, "config record")
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
    return {
        "format": FORMAT_VERSION,
        "id": graph.id,
        "metadata": strict_json(graph.metadata, "graph metadata"),
        "nodes": [node_record(node) for node in graph.nodes()],
        "edges": [edge_record(edge) for edge in graph.edges()],
    }


def from_records(document: dict[str, Any]) -> Graph:
    """Rebuild a graph from ``to_records``; a node that was running is ready to run again."""
    graph = Graph(_text(document.get("id"), "graph id"))
    graph.metadata = _dict(document.get("metadata"), "graph metadata")
    nodes: dict[str, Node] = {}
    for raw in _list(document, "nodes"):
        node_id = _text(raw.get("id"), "node id")
        state = raw.get("state", DONE)
        if state not in (READY, RUNNING, DONE):
            raise ValueError(f"node {node_id} has unknown state {state!r}")
        timing = raw.get("timing") or {}
        data = _dict(raw.get("data"), "node data")
        fields: dict[str, Any] = dict(
            role=str(raw.get("role") or "user"),
            text=str(raw.get("text") or ""),
            input=_dict(raw.get("input"), "node input"),
            code=str(raw.get("code") or "pass"),
            output=str(raw.get("output") or ""),
            result=raw.get("result"),
            error=bool(raw.get("error")),
            state=READY if state == RUNNING else state,
            calls=list(raw.get("calls") or []),
            data=data,
            producer=str(raw.get("producer") or data.pop("producer", "")),
            metadata=_dict(raw.get("metadata"), "node metadata"),
            id=node_id,
            owner=_text(raw.get("owner"), f"node {node_id} owner"),
            seq=int(raw.get("seq", -1)),
            version=int(raw.get("version", 0)),
            started_at=_epoch(timing.get("started_at")),
            finished_at=_epoch(timing.get("finished_at")),
            graph=graph,
        )
        if created := _epoch(raw.get("created_at")):
            fields["created_at"] = created
        agent = raw.get("agent", raw.get("root"))
        if agent is None:
            node: Node = Node(**fields)
        else:
            agent = _dict(agent, "agent")
            # Format 6 kept the agent's own metadata, and its config record, apart.
            older = _dict(agent.get("metadata"), "agent metadata")
            recorded = agent.get("config_record", older.pop("config", None))
            fields["metadata"] = {**older, **fields["metadata"]}
            node = Agent(
                **fields,
                name=_text(agent.get("name"), "agent name"),
                task=str(agent.get("task") or ""),
                config_name=agent.get("config"),
                arg_descriptions=_dict(agent.get("arg_descriptions"), "argument descriptions"),
                repl_id=str(agent.get("repl") or node_id),
                config_record=recorded,
            )
        if nodes.setdefault(node.id, node) is not node:
            raise GraphError(f"node {node.id!r} appears twice")
    ordered = sorted(nodes.values(), key=lambda n: n.seq)
    for node in ordered:
        if isinstance(node, Agent):
            if node.owner != node.id:
                raise GraphError(f"agent {node.id!r} must own itself")
            graph._roots[node.id] = node
            graph._own[node.id] = []
    for node in ordered:
        own = graph._own.get(node.owner)
        if own is None:
            raise GraphError(f"node {node.id!r} names {node.owner!r}, which is no agent here")
        if not own and node.owner != node.id:
            raise GraphError(f"node {node.id!r} comes before its agent {node.owner!r}")
        own.append(node)
    graph._nodes = {node.id: node for node in ordered}
    for raw in _list(document, "edges"):
        if raw.get("kind") == "fork":
            raise GraphError(
                "this run has a fork edge, saved before forks became copies of the graph; "
                "an older rlmflow can still read it"
            )
        edge = Edge(
            _text(raw.get("kind"), "edge kind"),
            _text(raw.get("source"), "edge source"),
            _text(raw.get("target"), "edge target"),
            dict(raw.get("data") or {}),
            _text(raw.get("id"), "edge id"),
            int(raw.get("seq", -1)),
        )
        for end in (edge.source, edge.target):
            if end not in graph._nodes:
                raise GraphError(f"edge {edge.id!r} points at missing {end!r}")
        graph._index(edge)
    graph._seq = 1 + max(
        [-1, *(n.seq for n in nodes.values()), *(e.seq for e in graph._edges.values())]
    )
    return graph


class _BlobStore:
    """Content-address large input strings once per saved graph."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def encode(self, value: Any) -> Any:
        def encode_string(_path: str, item: Any) -> Any:
            if not isinstance(item, str):
                return RECURSE
            encoded = item.encode("utf-8")
            if len(encoded) < BLOB_THRESHOLD_BYTES:
                return item
            digest = hashlib.sha256(encoded).hexdigest()
            self.path.mkdir(parents=True, exist_ok=True)
            blob_path = self.path / f"{digest}.txt"
            if not blob_path.exists():
                _write_text(blob_path, item)
            return {BLOB_REF_KEY: digest, "encoding": "utf-8", "bytes": len(encoded)}

        return transform_argument(value, encode_string, path="")


def _resolve_blobs(value: Any, blob_path: Path) -> Any:
    def resolve_blob(_path: str, item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {BLOB_REF_KEY, "encoding", "bytes"}:
            digest = item.get(BLOB_REF_KEY)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid blob digest {digest!r}")
            path = blob_path / f"{digest}.txt"
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise ValueError(f"missing blob {digest}") from exc
            encoded = text.encode("utf-8")
            if len(encoded) != item.get("bytes") or hashlib.sha256(encoded).hexdigest() != digest:
                raise ValueError(f"blob {digest} failed integrity validation")
            return text
        return RECURSE

    return transform_argument(value, resolve_blob, path="")


def save_graph(graph: Graph, path: str | Path) -> Path:
    """Atomically write ``graph.json``, moving large input strings into ``blobs/``."""
    run = Path(path)
    run.mkdir(parents=True, exist_ok=True)
    document = to_records(graph)
    blobs = _BlobStore(run / "blobs")
    for record in document["nodes"]:
        record["input"] = blobs.encode(record["input"])
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
    if version not in READS:
        raise ValueError(
            f"{run} was saved in run format {version or 1}, and this rlmflow reads only "
            f"formats {', '.join(map(str, READS))}; rerun it with this version."
        )
    for record in _list(document, "nodes"):
        record["input"] = _resolve_blobs(_dict(record.get("input"), "node input"), run / "blobs")
    return from_records(document)


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
    "BLOB_THRESHOLD_BYTES",
    "FORMAT_VERSION",
    "edge_record",
    "from_records",
    "load_graph",
    "node_record",
    "save_graph",
    "to_records",
]
