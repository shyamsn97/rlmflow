"""Write an agent tree in the run format: ``graph.json`` plus a directory per agent."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    AppendChild,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    Node,
    UserQuery,
)

VERSION = 2

NODE_TYPES: dict[str, type[Node]] = {
    cls.type: cls
    for cls in (
        AgentStart,
        UserQuery,
        LLMOutput,
        ExecAction,
        AppendChild,
        ExecOutput,
        ErrorOutput,
        DoneOutput,
    )
}
#: Runs saved when a delegating turn wrote a node of its own. Children hang off the
#: action now, so the partial output it carried reads back as an ordinary output.
NODE_TYPES["supervising_output"] = ExecOutput


def to_dict(agent: AgentStart) -> dict[str, Any]:
    """The whole tree, nested from ``agent`` down."""
    return {
        "version": VERSION,
        "graph_id": agent.id,
        "node_count": sum(1 for _ in agent.walk()),
        "metadata": {},
        "root": agent.to_dict(),
    }


def agent_to_dict(agent: AgentStart) -> dict[str, Any]:
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


def summary(agent: AgentStart) -> dict[str, Any]:
    nodes = list(agent.walk())
    return {
        "version": VERSION,
        "graph_id": agent.id,
        "root_id": agent.id,
        "root_agent_id": agent.config.path,
        "node_count": len(nodes),
        "agent_ids": [n.config.path for n in nodes if isinstance(n, AgentStart)],
        "finished": agent.terminal,
        "result": agent.result(),
    }


def save(agent: AgentStart, path: str | Path) -> Path:
    """Write ``agent`` to ``path`` and return the run directory."""
    run = Path(path)
    agents = run / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    _write_json(run / "graph.json", to_dict(agent))
    _write_json(run / "latest.json", summary(agent))
    _write_agent(agent, agents / agent.config.name)
    _prune(agents, {agent.config.name})
    return run


def load(path: str | Path) -> AgentStart:
    """Rebuild the tree that :func:`save` wrote to ``path``."""
    data = json.loads((Path(path) / "graph.json").read_text(encoding="utf-8"))
    if "root" not in data:
        # The engine before the node-tree rewrite keyed agents off a flat table.
        # Nothing reads that shape any more, so say which shape this is instead of
        # failing on a missing key.
        shape = "pre-rewrite (root_agent_id/agents)" if "root_agent_id" in data else "unknown"
        raise ValueError(f"{path}/graph.json is not a run graph: {shape} format")
    return from_dict(data["root"])


def from_dict(data: dict[str, Any]) -> AgentStart:
    """Rebuild a tree from run-format data by replaying its appends."""
    root = _new_node(data)
    if not isinstance(root, AgentStart):
        raise TypeError(f"a run must start with an agent, not {data.get('type')!r}")
    _attach(root, data)
    return root


def _write_agent(agent: AgentStart, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "agent.json", agent_to_dict(agent))
    _write_json(path / "latest.json", agent.frontier.to_dict(nested=False))
    session = [
        json.dumps(node.to_dict(nested=False), default=str, ensure_ascii=False)
        for node in agent.transcript()
    ]
    (path / "session.jsonl").write_text("".join(line + "\n" for line in session), encoding="utf-8")
    for sub in agent.sub_agents:
        _write_agent(sub, path / sub.config.name)
    _prune(path, {sub.config.name for sub in agent.sub_agents})


def _prune(path: Path, keep: set[str]) -> None:
    """Drop agent directories left behind by an earlier, larger save."""
    for entry in path.iterdir():
        if entry.is_dir() and entry.name not in keep and (entry / "agent.json").is_file():
            shutil.rmtree(entry)


def _attach(node: Node, data: dict[str, Any]) -> None:
    # Sub-agents branch off before the sequel, so they have to attach while the
    # agent's frontier is still this node.
    children = sorted(data["children"], key=lambda child: child["type"] != AgentStart.type)
    for child_data in children:
        child = node.append(_new_node(child_data, data.get("agent_id")))
        _attach(child, child_data)


def _new_node(data: dict[str, Any], parent_agent: str | None = None) -> Node:
    payload = dict(data.get("payload") or {})
    metadata = data.get("metadata") or {}
    agent_id = data.get("agent_id") or "root"
    cls = NODE_TYPES.get(data.get("type", ""), Node)
    timing = metadata.get("timing") or {}
    ran = {
        "started_at": _epoch(timing.get("started_at")),
        "finished_at": _epoch(timing.get("finished_at")),
    }
    # Runs saved before nodes were stamped have no creation time; those fall back
    # to the field default so the graph is still orderable, just by load order.
    created = _epoch(metadata.get("created_at"))
    if created is not None:
        ran["created_at"] = created
    # Older runs opened an agent with a query node, so a query naming a different
    # agent than its parent is where that agent starts.
    if cls is AgentStart or (cls is UserQuery and agent_id != parent_agent):
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


def _config(path: str, payload: dict[str, Any]) -> AgentConfig:
    """What the format records; the run limits belong to the flow, not the run."""
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
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "VERSION",
    "agent_to_dict",
    "from_dict",
    "load",
    "save",
    "summary",
    "to_dict",
]
