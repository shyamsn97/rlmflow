"""Persist a graph as its ordered item log plus content-addressed argument blobs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from rlmflow.graph.config import RECURSE, AgentConfig, transform_argument
from rlmflow.graph.graph import Edge, Graph, InvariantError, check_invariants
from rlmflow.graph.nodes import AGENT_START, NODE_TYPES, Node, isoformat, strict_json

#: Bumped on any incompatible change; older runs are refused, not converted.
FORMAT_VERSION = 2
ARG_BLOB_THRESHOLD_BYTES = 64 * 1024
ARG_BLOB_REF_KEY = "$rlmflow_arg_blob"


class _ArgBlobStore:
    """Content-address large agent arguments once per persisted graph."""

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


def node_record(node: Node) -> dict[str, Any]:
    """One node as a JSON log record."""
    record: dict[str, Any] = {
        "id": node.id,
        "type": node.type,
        "agent_id": node.agent_id,
        "seq": node.seq,
        "created_at": isoformat(node.created_at),
        "data": strict_json(node.data, f"node {node.id!r} data"),
    }
    if node.metadata:
        record["metadata"] = strict_json(node.metadata, f"node {node.id!r} metadata")
    timing = node.timing()
    if timing:
        record["timing"] = timing
    return record


def edge_record(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "kind": edge.kind,
        "source": edge.source,
        "target": edge.target,
        "data": dict(edge.data),
        "seq": edge.seq,
        "removed_seq": edge.removed_seq,
    }


def save_graph(graph: Graph, path: str | Path) -> Path:
    """Atomically write the graph log, moving large arguments into ``blobs/``."""
    run = Path(path)
    run.mkdir(parents=True, exist_ok=True)
    blobs = _ArgBlobStore(run / "blobs")
    records = []
    for item in graph.items():
        if isinstance(item, Edge):
            records.append(edge_record(item))
            continue
        record = node_record(item)
        if item.type == AGENT_START:
            record["data"]["args"] = blobs.encode(record["data"].get("args", {}))
        records.append(record)
    document = {
        "format": FORMAT_VERSION,
        "id": graph.id,
        "configs": {name: config.to_dict() for name, config in graph.configs.items()},
        "items": records,
    }
    _write_json(run / "graph.json", document)
    return run


def load_graph(path: str | Path) -> Graph:
    """Read a run directory: index its log, then check the whole graph once."""
    run = Path(path)
    data = _read_json(run / "graph.json")
    version = data.get("format")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{run} was saved in run format {version or 1}, and this rlmflow reads only "
            f"format {FORMAT_VERSION}. Runs from before the harness refactor can't be "
            "loaded; rerun them with this version."
        )
    records = data.get("items")
    if not isinstance(records, list):
        raise ValueError("graph field 'items' must be a list")
    graph_id = data.get("id")
    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph has an invalid id")
    graph = Graph(graph_id)
    for name, config in (data.get("configs") or {}).items():
        graph.configs[name] = AgentConfig.from_dict(config)
    try:
        for position, raw in enumerate(records):
            if not isinstance(raw, dict):
                raise TypeError(f"graph item {position} must be an object")
            if raw.get("seq") != position and raw.get("removed_seq") != position:
                raise ValueError(f"graph item {position} is out of log order")
            removing = "kind" in raw and raw.get("removed_seq") is not None
            graph._append_unchecked(_log_item(graph, raw, run), removing=removing)
        check_invariants(graph)
    except InvariantError as exc:
        raise InvariantError(f"invalid graph log: {exc}") from exc
    return graph


def _log_item(graph: Graph, raw: dict[str, Any], run: Path) -> Node | Edge:
    if "kind" in raw:
        if raw.get("removed_seq") is not None:
            current = graph._edges.get(raw["id"])
            if current is None or current.structural or not current.active:
                raise InvariantError(f"cannot remove edge {raw['id']!r}")
            return current
        return Edge(
            kind=raw["kind"],
            source=raw["source"],
            target=raw["target"],
            data=dict(raw.get("data") or {}),
            id=raw["id"],
        )
    node_type = raw.get("type")
    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown node type {node_type!r}")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"node {raw.get('id')!r} data must be an object")
    node = Node(node_type, **data)
    node.id = raw["id"]
    agent_id = raw.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(f"node {raw.get('id')!r} has an invalid agent_id")
    node.agent_id = agent_id
    node.metadata = dict(raw.get("metadata") or {})
    if raw.get("created_at"):
        node.created_at = _epoch(raw["created_at"])
    timing = raw.get("timing") or {}
    node.started_at = _epoch(timing.get("started_at"))
    node.finished_at = _epoch(timing.get("finished_at"))
    if node_type == AGENT_START:
        AgentConfig.from_dict(data.get("config") or {})
        node.data["args"] = {
            name: _resolve_arg(value, run / "blobs") for name, value in node.args.items()
        }
    return node


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp).timestamp() if stamp else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    _write_text(path, text + "\n")


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


__all__ = [
    "ARG_BLOB_THRESHOLD_BYTES",
    "FORMAT_VERSION",
    "edge_record",
    "load_graph",
    "node_record",
    "save_graph",
]
