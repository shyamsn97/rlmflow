"""Persist a run as flat, parent-linked node records."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from rlmflow.graph.config import AGENT_REF_KEY, AgentConfig, agent_ref_id
from rlmflow.graph.nodes import (
    ActionNode,
    AgentStart,
    AppendChild,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    LLMOutput,
    LLMUsage,
    Node,
    OutputNode,
    PlanQuery,
    ReplDead,
    TruncationSummary,
    UserQuery,
    _isoformat,
    new_agent_id,
    new_node_id,
)

VERSION = 5
INPUT_BLOB_THRESHOLD_BYTES = 64 * 1024
INPUT_BLOB_REF_KEY = "$rlmflow_input_blob"
NodeT = TypeVar("NodeT", bound=Node)


class UnsupportedGraphVersion(ValueError):
    """Raised when a run uses a persistence format this runtime does not read."""

    def __init__(self, found: object) -> None:
        super().__init__(f"unsupported graph version {found!r}; this runtime requires v{VERSION}")
        self.found = found
        self.supported = VERSION


_NODE_TYPES: dict[str, type[Node]] = {}


def register_node_type(cls: type[NodeT]) -> type[NodeT]:
    """Register one unique Node type for persistence decoding."""
    name = cls.type
    existing = _NODE_TYPES.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"node type {name!r} is already registered to {existing.__name__}")
    _NODE_TYPES[name] = cls
    return cls


for _node_type in (
    Node,
    AgentStart,
    UserQuery,
    PlanQuery,
    FinalQuery,
    TruncationSummary,
    LLMOutput,
    ActionNode,
    ExecAction,
    AppendChild,
    OutputNode,
    ExecOutput,
    ErrorOutput,
    ReplDead,
    DoneOutput,
):
    register_node_type(_node_type)


def to_document(root: AgentStart) -> dict[str, Any]:
    """Return one complete v5 run document."""
    if root.root is not root or root.parent is not None:
        raise ValueError("a persisted run must be a standalone root")
    _validate_graph(root)
    records = [node.to_record() for node in root.walk()]
    return {
        "version": VERSION,
        "graph_id": root.id,
        "root_id": root.id,
        "node_count": len(records),
        "metadata": {},
        "nodes": records,
    }


def from_document(data: dict[str, Any]) -> AgentStart:
    """Strictly rebuild a v5 run in one parent-before-child pass."""
    version = data.get("version")
    if version != VERSION:
        raise UnsupportedGraphVersion(version)

    records = data.get("nodes")
    if not isinstance(records, list):
        raise ValueError("run document field 'nodes' must be a list")
    if data.get("node_count") != len(records):
        raise ValueError(
            f"node_count {data.get('node_count')!r} does not match {len(records)} records"
        )

    created: dict[str, Node] = {}
    root: AgentStart | None = None
    for position, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise TypeError(f"node record {position} must be an object")
        record = dict(raw)
        node_id = record.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"node record {position} has an invalid id")
        if node_id in created:
            raise ValueError(f"duplicate node id {node_id!r}")

        parent_id = record.get("parent_id")
        node = _new_node(record)
        if parent_id is None:
            if root is not None:
                raise ValueError("run has more than one root")
            if position != 0:
                raise ValueError("the root record must be first")
            if not isinstance(node, AgentStart):
                raise TypeError(f"a run must start with an agent, not {node.type!r}")
            root = node
        else:
            if not isinstance(parent_id, str):
                raise ValueError(f"node {node_id!r} has an invalid parent_id")
            parent = created.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"node {node_id!r} references missing or later parent {parent_id!r}"
                )
            try:
                parent.append(node)
            except (RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid append for node {node_id!r}: {exc}") from exc

        if node.seq != record.get("order"):
            raise ValueError(
                f"node {node_id!r} has order {record.get('order')!r}; expected {node.seq}"
            )
        expected_agent = node.parent_agent.config.path if node.parent_agent is not None else None
        if record.get("agent_id") != expected_agent:
            raise ValueError(
                f"node {node_id!r} belongs to {record.get('agent_id')!r}; "
                f"expected {expected_agent!r}"
            )
        created[node_id] = node

    if root is None:
        raise ValueError("run has no root")
    root_id = data.get("root_id")
    if root_id != root.id:
        raise ValueError(f"root_id {root_id!r} does not match root record {root.id!r}")
    graph_id = data.get("graph_id")
    if graph_id != root.id:
        raise ValueError(f"graph_id {graph_id!r} does not match root record {root.id!r}")
    _validate_graph(root)
    return root


def _validate_graph(root: AgentStart) -> None:
    """Reject dangling launch receipts and invalid handle-valued inputs."""
    edges: dict[str, tuple[str, ...]] = {}
    for agent in root.iter_agents():
        sources: list[str] = []
        source_ids: set[str] = set()
        for input_name, value in agent.config.inputs.items():
            if not isinstance(input_name, str) or not input_name:
                raise ValueError(
                    f"agent {agent.config.path!r} has invalid input name {input_name!r}"
                )
            if isinstance(value, str):
                continue
            source_id = agent_ref_id(value)
            if source_id is None:
                raise TypeError(
                    f"agent {agent.config.path!r} input {input_name!r} must be "
                    "a string or exact agent reference"
                )
            source = root.find_agent(source_id)
            if source is None:
                raise ValueError(
                    f"agent {agent.config.path!r} input {input_name!r} references "
                    f"missing agent {source_id!r}"
                )
            if source is agent:
                raise ValueError(f"agent {agent.config.path!r} cannot depend on itself")
            consumer_launch = root._index.launch_actions_by_agent_id.get(agent.id)
            source_launch = root._index.launch_actions_by_agent_id.get(source.id)
            if (
                consumer_launch is None
                or source_launch is None
                or consumer_launch.parent_agent is not source_launch.parent_agent
            ):
                raise ValueError(
                    f"agent {agent.config.path!r} input {input_name!r} references "
                    "an agent outside its launch scope"
                )
            if source.id in source_ids:
                raise ValueError(
                    f"agent {agent.config.path!r} has duplicate agent input source "
                    f"{source.config.path!r}"
                )
            source_ids.add(source.id)
            sources.append(source.id)
        edges[agent.id] = tuple(sources)

    state: dict[str, int] = {}
    for start_id in edges:
        if state.get(start_id) == 2:
            continue
        stack = [(start_id, False)]
        while stack:
            agent_id, exiting = stack.pop()
            if exiting:
                state[agent_id] = 2
                continue
            if state.get(agent_id) == 2:
                continue
            if state.get(agent_id) == 1:
                agent = root.find_agent(agent_id)
                label = agent.config.path if agent is not None else agent_id
                raise ValueError(f"dependency cycle contains {label!r}")
            state[agent_id] = 1
            stack.append((agent_id, True))
            for source_id in reversed(edges.get(agent_id, ())):
                if state.get(source_id) == 1:
                    agent = root.find_agent(source_id)
                    label = agent.config.path if agent is not None else source_id
                    raise ValueError(f"dependency cycle contains {label!r}")
                if state.get(source_id) != 2:
                    stack.append((source_id, False))

    receipts: dict[str, tuple[str, int]] = {}
    for node in root.walk():
        if not isinstance(node, ExecAction):
            continue
        for call_id, agent_id in node.launched_agent_ids.items():
            if not isinstance(call_id, int) or call_id < 0:
                raise ValueError(f"action {node.id!r} has invalid launch call id {call_id!r}")
            if root.find_agent(agent_id) is None:
                raise ValueError(
                    f"action {node.id!r} launch receipt points to missing agent {agent_id!r}"
                )
            existing = receipts.get(agent_id)
            if existing is not None and existing[0] != node.id:
                raise ValueError(
                    f"agent {agent_id!r} has launch receipts from {existing[0]!r} and {node.id!r}"
                )
            receipts[agent_id] = (node.id, call_id)


def save(root: AgentStart, path: str | Path) -> Path:
    """Write derived views, then atomically commit the authoritative graph."""
    run = Path(path)
    agents = run / "agents"
    blobs = _InputBlobStore(run / "input_blobs")
    agents.mkdir(parents=True, exist_ok=True)
    _write_agents(root, agents, blobs)
    _write_json(run / "latest.json", _summary_document(root))
    _write_json(run / "graph.json", _encode_document(to_document(root), blobs))
    return run


def load(path: str | Path) -> AgentStart:
    """Load a strict v5 run directory."""
    run = Path(path)
    graph_path = run / "graph.json"
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{graph_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{graph_path} must contain a JSON object")
    return from_document(_resolve_document(data, run / "input_blobs"))


def clone_until(stop: Node) -> AgentStart:
    """Clone a complete run with fresh IDs, pruning descendants of ``stop``."""
    root = stop.root
    if root is None or stop.parent_agent is None:
        raise RuntimeError("node is detached")

    source: list[dict[str, Any]] = []
    stack = [root]
    found = False
    while stack:
        node = stack.pop()
        source.append(deepcopy(node.to_record()))
        if node is stop:
            found = True
            continue
        stack.extend(reversed(node.children))
    if not found:
        raise ValueError(f"node {stop.id!r} does not belong to its recorded root")

    id_map = {
        record["id"]: (new_agent_id() if record["type"] == AgentStart.type else new_node_id())
        for record in source
    }
    stamp = time.time()
    for offset, record in enumerate(source):
        old_id = record["id"]
        parent_id = record["parent_id"]
        record["id"] = id_map[old_id]
        record["parent_id"] = id_map[parent_id] if parent_id is not None else None
        payload = record.get("payload")
        if isinstance(payload, dict):
            inputs = payload.get("inputs")
            if isinstance(inputs, dict):
                remapped_inputs: dict[str, Any] = {}
                for name, value in inputs.items():
                    source_id = agent_ref_id(value)
                    if source_id is None:
                        remapped_inputs[name] = value
                        continue
                    if source_id not in id_map:
                        raise ValueError(
                            f"fork at {stop.id!r} is not closed over input dependency {source_id!r}"
                        )
                    remapped_inputs[name] = {AGENT_REF_KEY: id_map[source_id]}
                payload["inputs"] = remapped_inputs
            launched = payload.get("launched_agent_ids")
            if isinstance(launched, dict):
                payload["launched_agent_ids"] = {
                    str(call_id): id_map[agent_id]
                    for call_id, agent_id in launched.items()
                    if agent_id in id_map
                }
            retrieved = payload.get("retrieved_agent_ids")
            if isinstance(retrieved, list):
                payload["retrieved_agent_ids"] = [
                    id_map[agent_id] for agent_id in retrieved if agent_id in id_map
                ]
        record.setdefault("metadata", {})["created_at"] = _isoformat(stamp + offset * 1e-6)

    new_root_id = source[0]["id"]
    return from_document(
        {
            "version": VERSION,
            "graph_id": new_root_id,
            "root_id": new_root_id,
            "node_count": len(source),
            "metadata": {},
            "nodes": source,
        }
    )


def _agent_document(agent: AgentStart) -> dict[str, Any]:
    config = agent.config
    return {
        "agent_id": config.path,
        "parent_agent_id": config.path.rpartition(".")[0] or None,
        "depth": config.depth,
        "query": agent.content,
        "inputs": dict(config.inputs),
        "model": config.model,
        "prompt_profile": config.prompt_profile,
        "output_schema": config.output_schema,
        "system_prompts": dict(agent.system_prompts),
    }


def _summary_document(root: AgentStart) -> dict[str, Any]:
    stats = root.stats
    return {
        "version": VERSION,
        "graph_id": root.id,
        "root_id": root.id,
        "root_agent_id": root.config.path,
        "node_count": stats.node_count,
        "agent_ids": [agent.config.path for agent in root.iter_agents()],
        "finished": root.terminal,
        "result": root.result(),
    }


def _write_agents(
    root: AgentStart,
    agents_path: Path,
    blobs: _InputBlobStore,
) -> None:
    pending = [(root, agents_path / root.config.name)]
    while pending:
        agent, path = pending.pop()
        path.mkdir(parents=True, exist_ok=True)
        agent_document = _agent_document(agent)
        agent_document["inputs"] = blobs.encode_inputs(agent_document["inputs"])
        _write_json(path / "agent.json", agent_document)
        _write_json(path / "latest.json", _encode_record(agent.frontier.to_record(), blobs))
        lines = [
            json.dumps(_encode_record(node.to_record(), blobs), default=str, ensure_ascii=False)
            for node in agent.transcript()
        ]
        _write_text(path / "session.jsonl", "".join(line + "\n" for line in lines))
        _prune(path, {sub.config.name for sub in agent.sub_agents})
        pending.extend((sub, path / sub.config.name) for sub in reversed(agent.sub_agents))
    _prune(agents_path, {root.config.name})


class _InputBlobStore:
    """Content-address large agent inputs once per persisted graph."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def encode_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {name: self.encode(value) for name, value in inputs.items()}

    def encode(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        encoded = value.encode("utf-8")
        if len(encoded) < INPUT_BLOB_THRESHOLD_BYTES:
            return value
        digest = hashlib.sha256(encoded).hexdigest()
        self.path.mkdir(parents=True, exist_ok=True)
        blob_path = self.path / f"{digest}.txt"
        if not blob_path.exists():
            _write_text(blob_path, value)
        return {
            INPUT_BLOB_REF_KEY: digest,
            "encoding": "utf-8",
            "bytes": len(encoded),
        }


def _encode_document(data: dict[str, Any], blobs: _InputBlobStore) -> dict[str, Any]:
    for record in data.get("nodes", []):
        _encode_record(record, blobs)
    return data


def _encode_record(record: dict[str, Any], blobs: _InputBlobStore) -> dict[str, Any]:
    if record.get("type") != AgentStart.type:
        return record
    payload = record.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("inputs"), dict):
        payload["inputs"] = blobs.encode_inputs(payload["inputs"])
    return record


def _resolve_document(data: dict[str, Any], blob_path: Path) -> dict[str, Any]:
    for record in data.get("nodes", []):
        if record.get("type") != AgentStart.type:
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), dict):
            continue
        payload["inputs"] = {
            name: _resolve_input(value, blob_path) for name, value in payload["inputs"].items()
        }
    return data


def _resolve_input(value: Any, blob_path: Path) -> Any:
    if not isinstance(value, dict) or set(value) != {
        INPUT_BLOB_REF_KEY,
        "encoding",
        "bytes",
    }:
        return value
    digest = value.get(INPUT_BLOB_REF_KEY)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"invalid input blob digest {digest!r}")
    path = blob_path / f"{digest}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"missing input blob {digest}") from exc
    encoded = text.encode("utf-8")
    if len(encoded) != value.get("bytes") or hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError(f"input blob {digest} failed integrity validation")
    return text


def _prune(path: Path, keep: set[str]) -> None:
    """Drop agent directories left behind by an earlier, larger save."""
    for entry in path.iterdir():
        if entry.is_dir() and entry.name not in keep and (entry / "agent.json").is_file():
            shutil.rmtree(entry)


def _new_node(data: dict[str, Any]) -> Node:
    raw_payload = data.get("payload") or {}
    raw_metadata = data.get("metadata") or {}
    if not isinstance(raw_payload, dict):
        raise TypeError(f"node {data.get('id')!r} payload must be an object")
    if not isinstance(raw_metadata, dict):
        raise TypeError(f"node {data.get('id')!r} metadata must be an object")
    payload = dict(raw_payload)
    metadata = raw_metadata
    type_name = data.get("type")
    cls = _NODE_TYPES.get(type_name)
    if cls is None:
        raise ValueError(f"unknown node type {type_name!r}; register it before loading")
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
        agent_id = data.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError(f"agent {data.get('id')!r} has an invalid agent_id")
        return AgentStart(
            id=data["id"],
            content=payload.get("content", ""),
            config=_config(agent_id, payload),
            system_prompts=dict(payload.get("system_prompts") or {}),
            **ran,
        )
    fields = {name: value for name, value in payload.items() if name in cls.__dataclass_fields__}
    if issubclass(cls, ExecAction) and "launched_agent_ids" in fields:
        launched = fields["launched_agent_ids"]
        if not isinstance(launched, dict):
            raise TypeError(f"node {data.get('id')!r} launch receipts must be an object")
        fields["launched_agent_ids"] = {
            int(call_id): agent_id for call_id, agent_id in launched.items()
        }
    if cls is LLMOutput:
        fields["usage"] = LLMUsage(**(metadata.get("usage") or {}))
        fields["prompt_id"] = metadata.get("system_prompt", "")
    return cls(id=data["id"], **fields, **ran)


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp).timestamp() if stamp else None


def _config(path: str, payload: dict[str, Any]) -> AgentConfig:
    """Reconstruct the complete persisted agent configuration."""
    defaults = AgentConfig()
    return AgentConfig(
        name=path.rpartition(".")[2],
        path=path,
        depth=path.count("."),
        inputs=dict(payload.get("inputs") or {}),
        model=payload.get("model", "default"),
        prompt_profile=payload.get("prompt_profile", "default"),
        output_schema=payload.get("output_schema"),
        child_output_schema=payload.get("child_output_schema"),
        reuse_repl=bool(payload.get("reuse_repl", False)),
        max_depth=payload.get("max_depth", 1),
        max_iters=payload.get("max_iters", defaults.max_iters),
        max_budget=payload.get("max_budget", defaults.max_budget),
        keep_n_messages=payload.get("keep_n_messages"),
        max_output_length=payload.get("max_output_length", defaults.max_output_length),
        max_query_chars=payload.get("max_query_chars", defaults.max_query_chars),
        resume_on_child_done=bool(
            payload.get("resume_on_child_done", defaults.resume_on_child_done)
        ),
    )


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
    "INPUT_BLOB_THRESHOLD_BYTES",
    "VERSION",
    "UnsupportedGraphVersion",
    "from_document",
    "load",
    "register_node_type",
    "save",
    "to_document",
]
