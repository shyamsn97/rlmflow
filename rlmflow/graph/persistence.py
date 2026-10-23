"""Persist a graph as its ordered item log plus content-addressed argument blobs."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from rlmflow.graph.config import RECURSE, AgentConfig, input_parts, transform_argument
from rlmflow.graph.graph import (
    DEPENDS_ON,
    FROM,
    NEXT,
    Edge,
    Graph,
    InvariantError,
    check_invariants,
)
from rlmflow.graph.nodes import (
    AgentStart,
    ExecAction,
    LLMOutput,
    LLMUsage,
    Node,
    TransitionDecision,
    _isoformat,
    agent_ref_id,
    message_snapshot,
)
from rlmflow.structured import as_finish_value

ARG_BLOB_THRESHOLD_BYTES = 64 * 1024
ARG_BLOB_REF_KEY = "$rlmflow_arg_blob"

_PERSISTED_CONFIG_FIELDS = (
    "backend",
    "model",
    "prompt_profile",
    "output_schema",
    "child_config",
    "verifier_id",
    "verifier_hash",
    "reuse_worker",
    "max_depth",
    "max_iters",
    "max_budget",
    "keep_n_messages",
    "max_output_length",
    "max_query_chars",
    "resume_on_child_done",
)
_BOOLEAN_CONFIG_FIELDS = frozenset({"reuse_worker", "resume_on_child_done"})


def _config_values(payload: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Config fields from a saved record; ``child_output_schema`` predates ``child_config``."""
    defaults = AgentConfig()
    values = {name: payload.get(name, getattr(defaults, name)) for name in names}
    for name in _BOOLEAN_CONFIG_FIELDS:
        values[name] = bool(values[name])
    values["child_config"] = dict(values["child_config"] or {})
    if (schema := payload.get("child_output_schema")) is not None:
        values["child_config"].setdefault("output_schema", schema)
    return values


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return value


def _node_types() -> dict[str, type[Node]]:
    """Every defined Node class that declares its own ``type``."""
    found: dict[str, type[Node]] = {}
    stack: list[type[Node]] = [Node]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "type" not in vars(cls):
            continue
        existing = found.setdefault(cls.type, cls)
        if existing is not cls:
            raise ValueError(
                f"node type {cls.type!r} is defined by both "
                f"{existing.__qualname__} and {cls.__qualname__}"
            )
    return found


class _ArgBlobStore:
    """Content-address large agent arguments once per persisted graph."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def encode_args(self, args: dict[str, Any]) -> dict[str, Any]:
        return {name: self.encode(value) for name, value in args.items()}

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
            return {
                ARG_BLOB_REF_KEY: digest,
                "encoding": "utf-8",
                "bytes": len(encoded),
            }

        return transform_argument(value, encode_string, path="")


def _resolve_document(data: dict[str, Any], blob_path: Path) -> dict[str, Any]:
    for record in data.get("nodes", []):
        if record.get("type") != AgentStart.type:
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
            continue
        payload["args"] = {
            name: _resolve_arg(value, blob_path) for name, value in payload["args"].items()
        }
    return data


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


def _new_node(data: dict[str, Any], *, config: AgentConfig | None = None) -> Node:
    raw_payload = data.get("payload") or {}
    raw_metadata = data.get("metadata") or {}
    if not isinstance(raw_payload, dict):
        raise TypeError(f"node {data.get('id')!r} payload must be an object")
    if not isinstance(raw_metadata, dict):
        raise TypeError(f"node {data.get('id')!r} metadata must be an object")
    payload = dict(raw_payload)
    metadata = raw_metadata
    extensions = metadata.get("extensions") or {}
    if not isinstance(extensions, dict):
        raise TypeError(f"node {data.get('id')!r} metadata extensions must be an object")
    if "messages" not in payload:
        raise ValueError(f"node {data.get('id')!r} has no message snapshot field")
    raw_messages = payload.pop("messages")
    snapshot = None if raw_messages is None else message_snapshot(raw_messages)
    type_name = data.get("type")
    cls = _node_types().get(type_name)
    if cls is None:
        raise ValueError(f"unknown node type {type_name!r}; import its class before loading")
    timing = metadata.get("timing") or {}
    if not isinstance(timing, dict):
        raise TypeError(f"node {data.get('id')!r} timing must be an object")
    ran: dict[str, Any] = {
        "started_at": _epoch(timing.get("started_at")),
        "finished_at": _epoch(timing.get("finished_at")),
    }
    created = _epoch(metadata.get("created_at"))
    if created is not None:
        ran["created_at"] = created
    if cls is AgentStart:
        if config is None:
            agent_id = data.get("agent_id")
            if not isinstance(agent_id, str) or not agent_id:
                raise ValueError(f"agent {data.get('id')!r} has an invalid agent_id")
            config = _v1_config(agent_id, payload)
        return AgentStart(
            id=data["id"],
            content=payload.get("content", ""),
            args=dict(payload.get("args") or {}),
            _messages=snapshot,
            config=config,
            system_prompts=dict(payload.get("system_prompts") or {}),
            metadata=deepcopy(extensions),
            **ran,
        )
    values = {name: value for name, value in payload.items() if name in cls.__dataclass_fields__}
    values["metadata"] = deepcopy(extensions)
    if issubclass(cls, ExecAction) and "saved_tool_calls" in values:
        saved = values["saved_tool_calls"]
        if not isinstance(saved, dict):
            raise TypeError(f"node {data.get('id')!r} saved tool calls must be an object")
        values["saved_tool_calls"] = {}
        for tool_name, calls in saved.items():
            if not isinstance(calls, dict):
                raise TypeError(
                    f"node {data.get('id')!r} saved {tool_name!r} calls must be an object"
                )
            try:
                values["saved_tool_calls"][tool_name] = {
                    int(call_id): result for call_id, result in calls.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"node {data.get('id')!r} has an invalid saved call id") from exc
    if cls in {LLMOutput, TransitionDecision}:
        values["usage"] = LLMUsage(**(metadata.get("usage") or {}))
    if cls is LLMOutput:
        values["prompt_id"] = metadata.get("system_prompt", "")
    return cls(id=data["id"], _messages=snapshot, **values, **ran)


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp).timestamp() if stamp else None


def _v1_config(path: str, payload: dict[str, Any]) -> AgentConfig:
    """The agent configuration of a tree-format record, named by its path."""
    values = _config_values(payload, _PERSISTED_CONFIG_FIELDS)
    return AgentConfig(name=path.rpartition(".")[2], **values)


# -- graph log format ------------------------------------------------------

_LOG_CONFIG_FIELDS = ("name", *_PERSISTED_CONFIG_FIELDS, "context_policy")
_ENVELOPE_FIELDS = frozenset(
    {"id", "agent_id", "metadata", "seq", "created_at", "started_at", "finished_at", "_messages"}
)


def node_record(node: Node, agent: AgentStart | None = None) -> dict[str, Any]:
    """One node as a log record; ``agent`` supplies per-turn model metadata."""
    metadata: dict[str, Any] = {"created_at": _isoformat(node.created_at)}
    if node.metadata:
        try:
            metadata["extensions"] = json.loads(json.dumps(node.metadata, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"node {node.id!r} metadata must be strict JSON") from exc
    timing = node.timing()
    if timing:
        metadata["timing"] = timing
    messages = node.messages
    payload: dict[str, Any] = {
        "content": node.content,
        "messages": None if messages is None else [message.as_dict() for message in messages],
    }
    if isinstance(node, AgentStart):
        payload["args"] = dict(node.args)
        payload["system_prompts"] = dict(node.system_prompts)
        payload["config"] = {name: getattr(node.config, name) for name in _LOG_CONFIG_FIELDS}
    else:
        payload |= {
            item.name: _plain(getattr(node, item.name))
            for item in fields(node)
            if item.name not in _ENVELOPE_FIELDS
        }
        if isinstance(node, LLMOutput):
            del payload["usage"], payload["prompt_id"]
            turn: dict[str, Any] = {
                "usage": asdict(node.usage),
                "system_prompt": node.prompt_id,
            }
            if agent is not None:
                turn["model"] = agent.config.model
                if agent.config.keep_n_messages is not None:
                    turn["keep_n_messages"] = agent.config.keep_n_messages
            metadata = {**turn, **metadata}
        elif isinstance(node, TransitionDecision):
            del payload["usage"]
            metadata["usage"] = asdict(node.usage)
    return {
        "id": node.id,
        "type": node.type,
        "agent_id": node.agent_id,
        "seq": node.seq,
        "metadata": metadata,
        "payload": payload,
    }


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
    _validate_saved_calls(graph.nodes(ExecAction))
    records = []
    for item in graph.items():
        if isinstance(item, Edge):
            records.append(edge_record(item))
            continue
        owner = graph._nodes.get(item.agent_id)
        record = node_record(item, owner if isinstance(owner, AgentStart) else None)
        if isinstance(item, AgentStart):
            record["payload"]["args"] = blobs.encode_args(record["payload"]["args"])
        records.append(record)
    _write_json(run / "graph.json", {"id": graph.id, "items": records})
    return run


def load_graph(path: str | Path) -> Graph:
    """Read a run directory: index its log, then check the whole graph once."""
    run = Path(path)
    data = _read_json(run / "graph.json")
    if "items" not in data:
        return upgrade_v1(run)
    records = data["items"]
    if not isinstance(records, list):
        raise ValueError("graph field 'items' must be a list")
    graph_id = data.get("id")
    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph has an invalid id")
    graph = Graph(graph_id)
    try:
        for position, raw in enumerate(records):
            if not isinstance(raw, dict):
                raise TypeError(f"graph item {position} must be an object")
            if raw.get("seq") != position and raw.get("removed_seq") != position:
                raise ValueError(f"graph item {position} is out of log order")
            graph._append_unchecked(*_log_item(graph, raw, run), removing=_is_removal(raw))
        check_invariants(graph)
    except InvariantError as exc:
        raise InvariantError(f"invalid graph log: {exc}") from exc
    _validate_saved_calls(graph.nodes(ExecAction))
    return graph


def _is_removal(raw: dict[str, Any]) -> bool:
    return "kind" in raw and raw.get("removed_seq") is not None


def _log_item(graph: Graph, raw: dict[str, Any], run: Path) -> tuple[Node | Edge]:
    if _is_removal(raw):
        current = graph._edges.get(raw["id"])
        if current is None or current.structural or not current.active:
            raise InvariantError(f"cannot remove edge {raw['id']!r}")
        return (current,)
    if "kind" in raw:
        edge = Edge(
            kind=raw["kind"],
            source=raw["source"],
            target=raw["target"],
            data=dict(raw.get("data") or {}),
            id=raw["id"],
        )
        return (edge,)
    node = _log_node(raw)
    if isinstance(node, AgentStart):
        node.args = {name: _resolve_arg(value, run / "blobs") for name, value in node.args.items()}
    return (node,)


def _log_node(data: dict[str, Any]) -> Node:
    payload = dict(data.get("payload") or {})
    if data.get("type") == AgentStart.type:
        config = payload.pop("config", None)
        if not isinstance(config, dict):
            raise TypeError(f"agent {data.get('id')!r} config must be an object")
        values = _config_values(config, _LOG_CONFIG_FIELDS)
        record = {**data, "payload": payload}
        node = _new_node(record, config=AgentConfig(**values))
    else:
        node = _new_node(data)
    agent_id = data.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError(f"node {data.get('id')!r} has an invalid agent_id")
    node.agent_id = agent_id
    return node


def _validate_saved_calls(actions: list[ExecAction]) -> None:
    """Saved tool results are strict JSON, keyed by non-negative call index."""
    for node in actions:
        if not isinstance(node.saved_tool_calls, dict):
            raise TypeError(f"action {node.id!r} saved tool calls must be an object")
        for tool_name, calls in node.saved_tool_calls.items():
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError(f"action {node.id!r} has an invalid saved tool name")
            if not isinstance(calls, dict):
                raise TypeError(f"action {node.id!r} saved {tool_name!r} calls must be an object")
            for call_id, result in calls.items():
                if not isinstance(call_id, int) or call_id < 0:
                    raise ValueError(
                        f"action {node.id!r} has invalid {tool_name!r} call id {call_id!r}"
                    )
                try:
                    as_finish_value(result)
                except TypeError as exc:
                    raise TypeError(
                        f"action {node.id!r} saved {tool_name!r} call {call_id} must be strict JSON"
                    ) from exc


def upgrade_v1(path: str | Path) -> Graph:
    """Convert a tree-format run directory into a graph.

    ``parent_id`` becomes a ``next`` edge, or a ``from`` edge on a child
    ``AgentStart``. Handle arguments become ``depends_on`` edges with ``as``,
    carrying any ``Input`` description.
    """
    run = Path(path)
    data = _resolve_document(_read_json(run / "graph.json"), run / "arg_blobs")
    records = data.get("nodes")
    if not isinstance(records, list):
        raise ValueError("run document field 'nodes' must be a list")
    graph = Graph(data.get("graph_id") or None)
    agents_by_path: dict[str, str] = {}
    items: list[Node | Edge] = []
    for raw in records:
        record = dict(raw)
        parent_id = record.get("parent_id")
        node = _new_node(record)
        if isinstance(node, AgentStart):
            agents_by_path[record["agent_id"]] = node.id
            node.agent_id = node.id
            items.append(node)
            if parent_id is not None:
                items.append(Edge(kind=FROM, source=node.id, target=parent_id))
            for name, value in list(node.args.items()):
                wrapped = input_parts(value)
                producer = agent_ref_id(wrapped[0] if wrapped is not None else value)
                if producer is None:
                    continue
                del node.args[name]
                edge_data = {"as": name}
                if wrapped is not None:
                    edge_data["description"] = wrapped[1]
                items.append(Edge(kind=DEPENDS_ON, source=node.id, target=producer, data=edge_data))
            continue
        owner = agents_by_path.get(record.get("agent_id"))
        if owner is None or parent_id is None:
            raise ValueError(f"node {node.id!r} belongs to no known agent")
        node.agent_id = owner
        items.append(node)
        items.append(Edge(kind=NEXT, source=parent_id, target=node.id))
    try:
        graph._append_unchecked(*items)
        check_invariants(graph)
    except InvariantError as exc:
        raise InvariantError(f"invalid graph log: {exc}") from exc
    return graph


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
    "edge_record",
    "load_graph",
    "node_record",
    "save_graph",
    "upgrade_v1",
]
