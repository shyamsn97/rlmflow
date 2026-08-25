"""Persist a run as flat, parent-linked node records."""

from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    AppendChild,
    ContinueQuery,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    InspectQuery,
    LLMOutput,
    LLMUsage,
    Node,
    PlanQuery,
    ReplDead,
    TruncationSummary,
    UserQuery,
    new_agent_id,
    new_node_id,
)

VERSION = 3
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
    InspectQuery,
    PlanQuery,
    FinalQuery,
    ContinueQuery,
    TruncationSummary,
    LLMOutput,
    ExecAction,
    AppendChild,
    ExecOutput,
    ErrorOutput,
    ReplDead,
    DoneOutput,
):
    register_node_type(_node_type)


def to_document(root: AgentStart) -> dict[str, Any]:
    """Return one complete v3 run document."""
    if root.root is not root or root.parent is not None:
        raise ValueError("a persisted run must be a standalone root")
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
    """Strictly rebuild a v3 run in one parent-before-child pass."""
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
    return root


def save(root: AgentStart, path: str | Path) -> Path:
    """Write derived views, then atomically commit the authoritative graph."""
    run = Path(path)
    agents = run / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    _write_agents(root, agents)
    _write_json(run / "latest.json", _summary_document(root))
    _write_json(run / "graph.json", to_document(root))
    return run


def load(path: str | Path) -> AgentStart:
    """Load a strict v3 run directory."""
    graph_path = Path(path) / "graph.json"
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{graph_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{graph_path} must contain a JSON object")
    return from_document(data)


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


def _write_agents(root: AgentStart, agents_path: Path) -> None:
    pending = [(root, agents_path / root.config.name)]
    while pending:
        agent, path = pending.pop()
        path.mkdir(parents=True, exist_ok=True)
        _write_json(path / "agent.json", _agent_document(agent))
        _write_json(path / "latest.json", agent.frontier.to_record())
        lines = [
            json.dumps(node.to_record(), default=str, ensure_ascii=False)
            for node in agent.transcript()
        ]
        _write_text(path / "session.jsonl", "".join(line + "\n" for line in lines))
        _prune(path, {sub.config.name for sub in agent.sub_agents})
        pending.extend((sub, path / sub.config.name) for sub in reversed(agent.sub_agents))
    _prune(agents_path, {root.config.name})


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
    if cls is LLMOutput:
        fields["usage"] = LLMUsage(**(metadata.get("usage") or {}))
        fields["prompt_id"] = metadata.get("system_prompt", "")
    return cls(id=data["id"], **fields, **ran)


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp).timestamp() if stamp else None


def _isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


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
        reuse_repl=bool(payload.get("reuse_repl", False)),
        launch_call_id=payload.get("launch_call_id"),
        max_depth=payload.get("max_depth", 1),
        max_iters=payload.get("max_iters", defaults.max_iters),
        child_max_iters=payload.get("child_max_iters"),
        max_budget=payload.get("max_budget", defaults.max_budget),
        keep_n_messages=payload.get("keep_n_messages"),
        max_output_length=payload.get("max_output_length", 4_000),
        max_query_chars=payload.get("max_query_chars", 2_000),
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
    "VERSION",
    "UnsupportedGraphVersion",
    "from_document",
    "load",
    "register_node_type",
    "save",
    "to_document",
]
