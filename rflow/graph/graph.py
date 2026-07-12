"""Tiny recursive graph model.

This is the minimal rflow source of truth. Events are emitted from graph commits,
but callers should inspect/replay this graph for durable state.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal
from uuid import uuid4


def new_id() -> str:
    return f"n_{uuid4().hex[:8]}"


def new_graph_id() -> str:
    return f"g_{uuid4().hex}"


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Node:
    type: str
    id: str = field(default_factory=new_id)
    agent_id: str = ""
    seq: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return False


@dataclass
class ObservationNode(Node):
    content: str = ""


@dataclass
class ActionNode(Node):
    pass


@dataclass
class UserQuery(ObservationNode):
    type: Literal["user_query"] = "user_query"


@dataclass
class LLMOutput(ObservationNode):
    type: Literal["llm_output"] = "llm_output"
    code: str = ""


@dataclass
class ExecAction(ActionNode):
    type: Literal["exec_action"] = "exec_action"
    code: str = ""


@dataclass
class ExecOutput(ObservationNode):
    type: Literal["exec_output"] = "exec_output"
    output: str = ""


@dataclass
class SupervisingOutput(ObservationNode):
    type: Literal["supervising_output"] = "supervising_output"
    output: str = ""
    waiting_on: list[str] = field(default_factory=list)


@dataclass
class ResumeAction(ActionNode):
    type: Literal["resume_action"] = "resume_action"
    resumed_from: list[str] = field(default_factory=list)


@dataclass
class ErrorOutput(ObservationNode):
    type: Literal["error_output"] = "error_output"
    error: str = ""
    output: str = ""


@dataclass
class DoneOutput(ObservationNode):
    type: Literal["done_output"] = "done_output"
    result: str = ""
    output: str = ""

    @property
    def terminal(self) -> bool:
        return True


NODE_TYPES = {
    "user_query": UserQuery,
    "llm_output": LLMOutput,
    "exec_action": ExecAction,
    "exec_output": ExecOutput,
    "supervising_output": SupervisingOutput,
    "resume_action": ResumeAction,
    "error_output": ErrorOutput,
    "done_output": DoneOutput,
}


def _digest(nodes: list[Node]) -> str:
    """Fingerprint of a node prefix for checkpoint staleness detection.

    Changes if any node below the checkpoint is added, removed, replaced, or has
    its code edited. Used by ``Graph.revert`` to refuse when history was rewritten.
    """
    h = hashlib.md5(usedforsecurity=False)
    for i, node in enumerate(nodes):
        h.update(f"{i}:{node.type}:{getattr(node, 'code', '')}".encode())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class GraphCheckpoint:
    """A savepoint = an index into an agent's node stream + a staleness digest.

    ``position`` is ``len(agent.nodes)`` at checkpoint time; ``revert`` truncates
    back to it. ``digest`` fingerprints ``nodes[:position]`` so a revert refuses if
    that prefix was rewritten out from under the checkpoint.
    """

    agent_id: str
    position: int
    digest: str


@dataclass
class Graph:
    agent_id: str = "root"
    graph_id: str = field(default_factory=new_graph_id)
    query: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    model: str = "default"
    depth: int = 0
    parent_agent_id: str | None = None
    output_schema: object | None = None
    nodes: list[Node] = field(default_factory=list)
    children: dict[str, Graph] = field(default_factory=dict)

    def current(self) -> Node | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def finished(self) -> bool:
        cur = self.current()
        return bool(cur and cur.terminal) and all(
            child.finished for child in self.children.values()
        )

    def result(self) -> str:
        cur = self.current()
        return cur.result if isinstance(cur, DoneOutput) else ""

    def tokens(self, *, recursive: bool = True) -> tuple[int, int]:
        graphs = self.walk() if recursive else (self,)
        input_tokens = output_tokens = 0
        for agent in graphs:
            for node in agent.nodes:
                if isinstance(node, LLMOutput):
                    usage = node.metadata.get("usage") or {}
                    input_tokens += int(usage.get("input_tokens", 0) or 0)
                    output_tokens += int(usage.get("output_tokens", 0) or 0)
        return input_tokens, output_tokens

    def total_tokens(self, *, recursive: bool = True) -> int:
        input_tokens, output_tokens = self.tokens(recursive=recursive)
        return input_tokens + output_tokens

    def usage(self, *, recursive: bool = True) -> LLMUsage:
        input_tokens, output_tokens = self.tokens(recursive=recursive)
        return LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def commit(self, node: Node) -> Node:
        node.agent_id = self.agent_id
        node.seq = len(self.nodes)
        self.nodes.append(node)
        return node

    def agent_for(self, agent_id: str | None = None) -> Graph:
        return self if agent_id is None else self[agent_id]

    def inject_action(self, content: str, *, agent_id: str | None = None) -> Any:
        from rflow.graph.events import AppendNode

        target = self.agent_for(agent_id)
        return AppendNode(
            type="append_node",
            agent_id=target.agent_id,
            node_type="user_query",
            node=UserQuery(content=content),
        )

    def inject(self, content: str, *, agent_id: str | None = None) -> Any:
        action = self.inject_action(content, agent_id=agent_id)
        apply_graph_action(self, action)
        return action

    def fork(
        self,
        *,
        from_node_id: str | None = None,
        agent_id: str | None = None,
        keep_children: bool = True,
        session: Literal["isolated", "shared"] = "isolated",
    ) -> Graph:
        if session not in {"isolated", "shared"}:
            raise ValueError("session must be 'isolated' or 'shared'")
        forked = deepcopy(self)
        if session == "isolated":
            forked.set_graph_id(new_graph_id())
        target = forked.agent_for(agent_id)
        if from_node_id is not None:
            forked.rewind(from_node_id, agent_id=target.agent_id)
        if not keep_children:
            target.children.clear()
        return forked

    def rewind_action(self, node_id: str, *, agent_id: str | None = None) -> Any:
        from rflow.graph.events import RemoveNode

        target = self.agent_for(agent_id)
        return RemoveNode(
            type="remove_node",
            agent_id=target.agent_id,
            node_id=node_id,
            subtree=True,
        )

    def rewind(self, node_id: str, *, agent_id: str | None = None) -> Any:
        action = self.rewind_action(node_id, agent_id=agent_id)
        apply_graph_action(self, action)
        return action

    def checkpoint(self, *, agent_id: str | None = None) -> GraphCheckpoint:
        agent = self.agent_for(agent_id)
        return GraphCheckpoint(
            agent_id=agent.agent_id,
            position=len(agent.nodes),
            digest=_digest(agent.nodes),
        )

    def revert(self, checkpoint: GraphCheckpoint) -> Any:
        """Truncate an agent's node stream back to ``checkpoint.position``.

        Refuses (raises) if the checkpoint is stale (nodes truncated below it) or
        if the prefix was rewritten (digest mismatch). Reuses the ``RemoveNode``
        path, so it emits a normal graph action and keeps history visible.
        """
        agent = self.agent_for(checkpoint.agent_id)
        if checkpoint.position > len(agent.nodes):
            raise ValueError(
                "stale checkpoint: agent has fewer nodes than checkpoint.position"
            )
        if _digest(agent.nodes[: checkpoint.position]) != checkpoint.digest:
            raise ValueError(
                "checkpoint digest mismatch: history was rewritten below the checkpoint"
            )
        if checkpoint.position == len(agent.nodes):
            return None  # already at the checkpoint; nothing to truncate
        node_id = agent.nodes[checkpoint.position].id
        return self.rewind(node_id, agent_id=checkpoint.agent_id)

    def replace_node_action(
        self, node_id: str, node: Node, *, agent_id: str | None = None
    ) -> Any:
        from rflow.graph.events import ReplaceNode

        target = self.agent_for(agent_id)
        return ReplaceNode(
            type="replace_node",
            agent_id=target.agent_id,
            node_id=node_id,
            node_type=node.type,
            node=node,
        )

    def replace_node(
        self, node_id: str, node: Node, *, agent_id: str | None = None
    ) -> Any:
        action = self.replace_node_action(node_id, node, agent_id=agent_id)
        apply_graph_action(self, action)
        return action

    def remove_child_action(
        self, child_agent_id: str, *, parent_agent_id: str | None = None
    ) -> Any:
        from rflow.graph.events import RemoveChild

        parent = self.agent_for(parent_agent_id)
        return RemoveChild(
            type="remove_child",
            parent_agent_id=parent.agent_id,
            child_agent_id=child_agent_id,
        )

    def remove_child(
        self, child_agent_id: str, *, parent_agent_id: str | None = None
    ) -> Any:
        action = self.remove_child_action(
            child_agent_id,
            parent_agent_id=parent_agent_id,
        )
        apply_graph_action(self, action)
        return action

    def set_graph_id(self, graph_id: str) -> None:
        for agent in self.walk():
            agent.graph_id = graph_id

    def walk(self) -> Iterator[Graph]:
        yield self
        for child in self.children.values():
            yield from child.walk()

    @property
    def agents(self) -> dict[str, Graph]:
        return {agent.agent_id: agent for agent in self.walk()}

    def __getitem__(self, agent_id: str) -> Graph:
        return self.agents[agent_id]

    def __contains__(self, agent_id: object) -> bool:
        return isinstance(agent_id, str) and agent_id in self.agents

    def save(
        self, path: str | Path = ".", *, metadata: dict[str, Any] | None = None
    ) -> Path:
        return save_run(self, path, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        return load_run(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "graph_id": self.graph_id,
            "query": self.query,
            "inputs": dict(self.inputs),
            "model": self.model,
            "depth": self.depth,
            "parent_agent_id": self.parent_agent_id,
            "output_schema": self.output_schema,
            "nodes": [node_to_dict(node) for node in self.nodes],
            "children": {
                agent_id: child.to_dict() for agent_id, child in self.children.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        return cls(
            agent_id=data.get("agent_id", "root"),
            graph_id=data.get("graph_id") or new_graph_id(),
            query=data.get("query", ""),
            inputs=dict(data.get("inputs") or {}),
            model=data.get("model", "default"),
            depth=data.get("depth", 0),
            parent_agent_id=data.get("parent_agent_id"),
            output_schema=data.get("output_schema"),
            nodes=[node_from_dict(node) for node in data.get("nodes", [])],
            children={
                agent_id: cls.from_dict(child)
                for agent_id, child in (data.get("children") or {}).items()
            },
        )


def node_to_dict(node: Node) -> dict[str, Any]:
    return asdict(node)


def node_from_dict(data: dict[str, Any]) -> Node:
    cls = NODE_TYPES.get(data.get("type"), Node)
    allowed = getattr(cls, "__dataclass_fields__", {})
    fields = {key: value for key, value in data.items() if key in allowed}
    return cls(**fields)


def apply_graph_action(graph: Graph | None, action: Any) -> Graph:
    from rflow.graph.events import (
        AddChild,
        AppendNode,
        GraphCreated,
        RemoveChild,
        RemoveNode,
        ReplaceNode,
    )

    if isinstance(action, GraphCreated):
        return action.graph
    if graph is None:
        raise ValueError("first graph action must create a graph")

    if isinstance(action, AppendNode):
        graph[action.agent_id].commit(action.node)
    elif isinstance(action, ReplaceNode):
        agent = graph[action.agent_id]
        for index, node in enumerate(agent.nodes):
            if node.id == action.node_id:
                action.node.agent_id = agent.agent_id
                action.node.seq = node.seq
                agent.nodes[index] = action.node
                break
        else:
            raise KeyError(action.node_id)
    elif isinstance(action, RemoveNode):
        agent = graph[action.agent_id]
        index = next(
            (i for i, node in enumerate(agent.nodes) if node.id == action.node_id),
            None,
        )
        if index is None:
            raise KeyError(action.node_id)
        if action.subtree:
            del agent.nodes[index:]
        else:
            del agent.nodes[index]
        _resequence(agent)
    elif isinstance(action, AddChild):
        parent = graph[action.parent_agent_id]
        action.child.graph_id = parent.graph_id
        graph[action.parent_agent_id].children[action.child.agent_id] = action.child
    elif isinstance(action, RemoveChild):
        del graph[action.parent_agent_id].children[action.child_agent_id]
    else:
        raise TypeError(f"unknown graph action: {action!r}")
    return graph


def _resequence(graph: Graph) -> None:
    for index, node in enumerate(graph.nodes):
        node.seq = index


def save_run(
    graph: Graph,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    run_root = Path(path)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "graph_id": graph.graph_id,
        "root_agent_id": graph.agent_id,
        "agents": [agent.agent_id for agent in graph.walk()],
        "metadata": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        },
    }
    result = graph.result()
    if result:
        manifest["metadata"]["result"] = result
    (run_root / "graph.json").write_text(
        json.dumps(manifest, default=str, indent=2),
        encoding="utf-8",
    )

    agents_root = run_root / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    root_local = _safe_dirname(graph.agent_id)
    _write_agent_dir(graph, agents_root / root_local)
    for entry in agents_root.iterdir():
        if (
            entry.is_dir()
            and entry.name != root_local
            and (entry / "agent.json").is_file()
        ):
            shutil.rmtree(entry)
    return run_root


def load_run(path: str | Path) -> Graph:
    run_root = Path(path)
    manifest = json.loads((run_root / "graph.json").read_text(encoding="utf-8"))
    root_dir = run_root / "agents" / _safe_dirname(manifest["root_agent_id"])
    return _load_agent_dir(root_dir)


def _load_agent_dir(dir_path: Path) -> Graph:
    data = json.loads((dir_path / "agent.json").read_text(encoding="utf-8"))
    nodes = _load_session(dir_path / "session.jsonl")
    children: dict[str, Graph] = {}
    for child_dir in sorted(dir_path.iterdir()):
        if not child_dir.is_dir() or not (child_dir / "agent.json").is_file():
            continue
        child = _load_agent_dir(child_dir)
        children[child.agent_id] = child
    return Graph(
        agent_id=data.get("agent_id", "root"),
        graph_id=data.get("graph_id") or new_graph_id(),
        query=data.get("query", ""),
        inputs=dict(data.get("inputs") or {}),
        model=data.get("model", "default"),
        depth=data.get("depth", 0),
        parent_agent_id=data.get("parent_agent_id"),
        output_schema=data.get("output_schema"),
        nodes=nodes,
        children=children,
    )


def _load_session(path: Path) -> list[Node]:
    if not path.exists():
        return []
    nodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            nodes.append(node_from_dict(json.loads(line)))
    return nodes


def _write_agent_dir(graph: Graph, dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "agent.json").write_text(
        json.dumps(agent_meta_dict(graph), default=str, indent=2),
        encoding="utf-8",
    )
    session = "\n".join(
        json.dumps(node_to_dict(node), default=str, ensure_ascii=False)
        for node in graph.nodes
    )
    if session:
        session += "\n"
    (dir_path / "session.jsonl").write_text(session, encoding="utf-8")

    latest_path = dir_path / "latest.json"
    if graph.nodes:
        latest_path.write_text(
            json.dumps(latest_dict(graph.nodes[-1]), default=str, indent=2),
            encoding="utf-8",
        )
    elif latest_path.exists():
        latest_path.unlink()

    kept: set[str] = set()
    for child in graph.children.values():
        local = _local_dirname(child.agent_id, graph.agent_id)
        kept.add(local)
        _write_agent_dir(child, dir_path / local)
    for entry in dir_path.iterdir():
        if (
            entry.is_dir()
            and entry.name not in kept
            and (entry / "agent.json").is_file()
        ):
            shutil.rmtree(entry)


def agent_meta_dict(graph: Graph) -> dict[str, Any]:
    return {
        "agent_id": graph.agent_id,
        "graph_id": graph.graph_id,
        "depth": graph.depth,
        "query": graph.query,
        "inputs": dict(graph.inputs),
        "model": graph.model,
        "output_schema": graph.output_schema,
        "parent_agent_id": graph.parent_agent_id,
    }


def latest_dict(node: Node) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent_id": node.agent_id,
        "latest_node_id": node.id,
        "seq": node.seq,
        "type": node.type,
        "terminal": node.terminal,
    }
    for key in ("result", "error", "waiting_on", "resumed_from"):
        value = getattr(node, key, None)
        if value:
            payload[key] = value
    return payload


def _safe_dirname(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in name)
    return safe.strip("_") or "agent"


def _local_dirname(agent_id: str, parent_agent_id: str | None) -> str:
    local = agent_id
    if parent_agent_id and agent_id.startswith(parent_agent_id + "."):
        local = agent_id[len(parent_agent_id) + 1 :]
    return _safe_dirname(local)


__all__ = [
    "ActionNode",
    "apply_graph_action",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "Graph",
    "GraphCheckpoint",
    "LLMUsage",
    "LLMOutput",
    "load_run",
    "Node",
    "node_from_dict",
    "ObservationNode",
    "ResumeAction",
    "SupervisingOutput",
    "UserQuery",
]
