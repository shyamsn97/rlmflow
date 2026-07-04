"""Typed trajectory nodes and the recursive :class:`Graph`.

Every trajectory is a strictly alternating chain of *observations*
(input the system received) and *actions* (work the system did in
response). Every action is followed by exactly one observation.

Hierarchy::

    Node
    ├── ObservationNode               base — inputs the system received
    │   ├── UserQuery                   bootstrap input
    │   ├── LLMOutput                   what the LLM returned
    │   └── CodeObservation             base — anything from running code
    │       ├── ExecOutput                normal stdout
    │       ├── SupervisingOutput        code awaited children; scheduler waits
    │       ├── ErrorOutput               code errored
    │       └── DoneOutput                code called done(); terminal
    └── ActionNode                    base — work the system did
        ├── LLMAction                   called the LLM
        ├── ExecAction                  ran the LLM's fresh code
        └── ResumeAction                supervisor resumed paused code

A :class:`Graph` is **one agent**: its per-run invariants, its trajectory
of nodes, and a dict of child agents. Recursion lives in ``children``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Iterator, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

if TYPE_CHECKING:
    from rflow.graph.views import EdgesView, NodesView


def new_id() -> str:
    return f"n_{uuid4().hex[:12]}"


# ── nodes ─────────────────────────────────────────────────────────────


class Node(BaseModel):
    """One immutable state in an agent's trajectory."""

    model_config = ConfigDict(frozen=True)

    type: str
    id: str = Field(default_factory=new_id)
    agent_id: str = "root"
    seq: int = 0
    global_step: int | None = None

    @property
    def terminal(self) -> bool:
        return False

    def update(self, **changes: Any) -> Node:
        return self.model_copy(update=changes)

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class ObservationNode(Node):
    """Base for nodes that record an *input the system received*."""


class ActionNode(Node):
    """Base for nodes that record *work the system did*."""


class CodeObservation(ObservationNode):
    """Base for any observation produced by running code.

    ``output`` is the raw captured stdout; ``content`` is that output
    rendered for the model's next user turn. ``resumed_from`` is empty for
    an :class:`ExecAction` result and populated for a :class:`ResumeAction`
    result.
    """

    output: str = ""
    content: str = ""
    resumed_from: list[str] = Field(default_factory=list)


class UserQuery(ObservationNode):
    """The bootstrap input (root query or child spawn prompt). Always seq=0."""

    type: Literal["user_query"] = "user_query"
    content: str = ""


class LLMOutput(ObservationNode):
    """What the LLM returned for one turn. ``code`` is the extracted block."""

    type: Literal["llm_output"] = "llm_output"
    reply: str = ""
    code: str = ""
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ExecOutput(CodeObservation):
    """Code ran and produced normal stdout."""

    type: Literal["exec_output"] = "exec_output"


class SupervisingOutput(CodeObservation):
    """Code reached ``await launch_subagents(...)``.

    The durable graph records which children the parent is waiting on. In the
    local scheduler path, the parent coroutine awaits a live future while the
    scheduler runs those children; once they settle, a :class:`ResumeAction`
    records that the parent continued. ``output`` is anything printed before
    the await.
    """

    type: Literal["supervising_output"] = "supervising_output"
    waiting_on: list[str] = Field(default_factory=list)
    launch_id: str | None = None
    launch_specs: list[dict[str, Any]] = Field(default_factory=list)
    launch_names: list[str] = Field(default_factory=list)


class ErrorOutput(CodeObservation):
    """Code execution errored. ``content`` is the retry message shown next."""

    type: Literal["error_output"] = "error_output"
    error: str = ""


class DoneOutput(CodeObservation):
    """Code called ``done(...)``. Terminal. ``result`` is the answer."""

    type: Literal["done_output"] = "done_output"
    result: str = ""

    @property
    def terminal(self) -> bool:
        return True


class LLMAction(ActionNode):
    """The engine called the LLM. Reply/code live on the paired LLMOutput."""

    type: Literal["llm_action"] = "llm_action"
    model: str | None = None


class ExecAction(ActionNode):
    """The engine ran the LLM's fresh code."""

    type: Literal["exec_action"] = "exec_action"
    code: str = ""


class ResumeAction(ActionNode):
    """The supervisor resumed paused code after its children settled."""

    type: Literal["resume_action"] = "resume_action"
    resumed_from: list[str] = Field(default_factory=list)


# ── node predicates + parser ──────────────────────────────────────────


def is_observation(node: Node) -> bool:
    return isinstance(node, ObservationNode)


def is_action(node: Node) -> bool:
    return isinstance(node, ActionNode)


def is_code_observation(node: Node) -> bool:
    return isinstance(node, CodeObservation)


def is_user_query(node: Node) -> bool:
    return isinstance(node, UserQuery)


def is_llm_output(node: Node) -> bool:
    return isinstance(node, LLMOutput)


def is_exec_output(node: Node) -> bool:
    return isinstance(node, ExecOutput)


def is_supervising(node: Node) -> bool:
    return isinstance(node, SupervisingOutput)


def is_errored(node: Node) -> bool:
    return isinstance(node, ErrorOutput)


def is_done(node: Node) -> bool:
    return isinstance(node, DoneOutput)


def is_llm_action(node: Node) -> bool:
    return isinstance(node, LLMAction)


def is_exec_action(node: Node) -> bool:
    return isinstance(node, ExecAction)


def is_resume_action(node: Node) -> bool:
    return isinstance(node, ResumeAction)


def is_resumed(node: Node) -> bool:
    """A :class:`CodeObservation` produced by a resume (``resumed_from`` set)."""
    return isinstance(node, CodeObservation) and bool(node.resumed_from)


# Discriminated union over the concrete leaf nodes, keyed on ``type``. Used to
# rebuild typed nodes from plain dicts (``Graph.from_dict``). Unknown extra keys
# in persisted payloads are ignored, so old traces still load.
NodeUnion = Annotated[
    Union[
        UserQuery,
        LLMOutput,
        ExecOutput,
        SupervisingOutput,
        ErrorOutput,
        DoneOutput,
        LLMAction,
        ExecAction,
        ResumeAction,
    ],
    Field(discriminator="type"),
]

_NODE_ADAPTER: TypeAdapter[Node] = TypeAdapter(NodeUnion)


def parse_node_obj(data: dict) -> Node:
    """Rebuild the concrete :class:`Node` subtype from a ``to_dict()`` payload."""
    return _NODE_ADAPTER.validate_python(data)


# ── REPL protocol handles (transient; not stored on the graph) ────────


class ChildHandle:
    """Reference returned by delegation, passed to the wait primitive."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def __repr__(self) -> str:
        return f"ChildHandle({self.agent_id!r})"

    def to_dict(self) -> dict:
        """JSON-safe form for shipping across the remote-REPL proxy boundary."""
        return {"child_handle": self.agent_id}

    @classmethod
    def from_dict(cls, data: dict) -> "ChildHandle":
        return cls(data["child_handle"])


class WaitRequest:
    """Awaited to request suspension until the named children finish."""

    def __init__(
        self,
        agent_ids: list[str],
        *,
        launch_id: str | None = None,
        launch_specs: list[dict[str, Any]] | None = None,
        launch_names: list[str] | None = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.launch_id = launch_id or new_id()
        self.launch_specs = list(launch_specs or [])
        self.launch_names = list(launch_names or [])

    def __repr__(self) -> str:
        return f"WaitRequest({self.agent_ids!r}, launch_id={self.launch_id!r})"

    def __await__(self):
        results = yield self
        return results

    def to_dict(self) -> dict:
        """JSON-safe form for shipping across the remote-REPL proxy boundary."""
        return {
            "wait_request": list(self.agent_ids),
            "launch_id": self.launch_id,
            "launch_specs": list(self.launch_specs),
            "launch_names": list(self.launch_names),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WaitRequest":
        return cls(
            list(data["wait_request"]),
            launch_id=data.get("launch_id"),
            launch_specs=data.get("launch_specs") or [],
            launch_names=data.get("launch_names") or [],
        )


# ── Graph ─────────────────────────────────────────────────────────────


def append_message(messages: list[dict[str, str]], role: str, content: str) -> None:
    """Append a chat message, coalescing consecutive same-role blocks.

    Some providers reject two adjacent messages with the same role, so a
    new block is merged into the previous message when the roles match.
    """
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] += "\n\n" + content
    else:
        messages.append({"role": role, "content": content})


@dataclass
class Graph:
    """One agent's view of a run, recursive through ``children``."""

    agent_id: str
    depth: int = 0
    query: str = ""
    system_prompt: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    max_iters: int | None = None
    output_schema: dict[str, Any] | None = None
    parent_agent_id: str | None = None
    parent_node_id: str | None = None
    nodes: list[Node] = field(default_factory=list)
    children: dict[str, "Graph"] = field(default_factory=dict)

    def current(self) -> Node | None:
        return self.nodes[-1] if self.nodes else None

    @property
    def finished(self) -> bool:
        cur = self.current()
        if not (cur and cur.terminal):
            return False
        return all(child.finished for child in self.children.values())

    def result(self) -> str:
        cur = self.current()
        if cur is not None and cur.terminal:
            return getattr(cur, "result", "") or ""
        return ""

    def walk(self) -> Iterator["Graph"]:
        """Yield self plus every descendant agent, depth-first."""
        yield self
        for child in self.children.values():
            yield from child.walk()

    @property
    def agents(self) -> dict[str, "Graph"]:
        """Flat ``{agent_id: Graph}`` view over the whole subtree."""
        return {g.agent_id: g for g in self.walk()}

    @property
    def all_nodes(self) -> "NodesView":
        """Flat, queryable view over every node in the subtree."""
        from rflow.graph.views import NodesView

        return NodesView(self)

    @property
    def edges(self) -> "EdgesView":
        """Derived flow + spawn edges across the subtree."""
        from rflow.graph.views import EdgesView

        return EdgesView(self)

    def copy(self, *, deep: bool = True) -> "Graph":
        """Return a copy of this subtree (deep by default for safe editing)."""
        return deepcopy(self) if deep else _dc_replace(self)

    def snapshot(self) -> "Graph":
        """Return a stable deep copy for UI, recording, and external readers."""
        return self.copy(deep=True)

    def repl_inputs(self) -> dict[str, str]:
        """Public ``INPUTS`` dict for this agent's REPL.

        Holds only caller-provided inputs. The query is delivered as the first
        user message (see ``prompts.messages.first_prompt``), not mirrored into
        ``INPUTS`` — agents read their task from chat, and ``inputs`` is reserved
        for the (potentially large) supporting payloads.
        """
        return dict(self.inputs)

    def max_global_step(self) -> int | None:
        """Highest ``global_step`` stamped on any node in the subtree, or ``None``."""
        steps = [
            n.global_step
            for g in self.walk()
            for n in g.nodes
            if n.global_step is not None
        ]
        return max(steps) if steps else None

    def next_global_step(self) -> int:
        """The step number a freshly appended node should take."""
        current = self.max_global_step()
        return 0 if current is None else current + 1

    def get_runnable_nodes(self) -> list[str]:
        """Ids of agents that can advance by one action right now.

        A leaf is runnable unless it's finished or mid-step; a supervisor wait
        gates on its children. Until those children finish, recurse into them to
        surface runnable descendants.
        """
        agents = self.agents
        out: list[str] = []

        def visit(g: "Graph") -> None:
            if g.finished:
                return
            cur = g.current()
            if cur is None:
                return
            if cur.terminal:
                for child in g.children.values():
                    if not child.finished:
                        visit(child)
                return
            if isinstance(cur, SupervisingOutput):
                waiting = [agents.get(aid) for aid in cur.waiting_on]
                if any(w is None for w in waiting):
                    return
                if all(w.finished for w in waiting):
                    out.append(g.agent_id)
                    return
                for w in waiting:
                    if not w.finished:
                        visit(w)
                return
            out.append(g.agent_id)

        visit(self)
        return out

    def runnable_descendants(self) -> list[str]:
        """Runnable agents strictly below this one (excludes self).

        Used by work-conserving (``eager_children``) scheduling to keep the
        pool busy with a waiting supervisor's descendants.
        """
        return [aid for aid in self.get_runnable_nodes() if aid != self.agent_id]

    def __getitem__(self, agent_id: str) -> "Graph":
        for g in self.walk():
            if g.agent_id == agent_id:
                return g
        raise KeyError(agent_id)

    def __contains__(self, agent_id: object) -> bool:
        return any(g.agent_id == agent_id for g in self.walk())

    def __iter__(self) -> Iterator[str]:
        """Iterate agent ids in the subtree (depth-first)."""
        return (g.agent_id for g in self.walk())

    def __len__(self) -> int:
        """Number of agents in the subtree (self plus every descendant)."""
        return sum(1 for _ in self.walk())

    @property
    def parent_id(self) -> str | None:
        """Alias for :attr:`parent_agent_id`."""
        return self.parent_agent_id

    # ── node access helpers ───────────────────────────────────────────

    def node_owner(self, node_id: str) -> "Graph":
        """Return the sub-:class:`Graph` whose local ``nodes`` hold ``node_id``."""
        for g in self.walk():
            if any(n.id == node_id for n in g.nodes):
                return g
        raise KeyError(node_id)

    def _index_of(self, node_id: str) -> int:
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                return i
        raise KeyError(node_id)

    def find(self, node_id: str) -> Node | None:
        """Return the node with ``node_id`` anywhere in the subtree, or ``None``."""
        return self.all_nodes.find(node_id)

    def last_action(self, agent_id: str | None = None) -> Node | None:
        """The latest :class:`ActionNode` of ``agent_id`` (or self)."""
        g = self if agent_id is None else self[agent_id]
        for n in reversed(g.nodes):
            if isinstance(n, ActionNode):
                return n
        return None

    def last_observation(self, agent_id: str | None = None) -> Node | None:
        """The latest :class:`ObservationNode` of ``agent_id`` (or self)."""
        g = self if agent_id is None else self[agent_id]
        for n in reversed(g.nodes):
            if isinstance(n, ObservationNode):
                return n
        return None

    # ── in-place mutators (edit a loaded graph offline) ───────────────

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def update_node(self, node_id: str, **changes: Any) -> Node:
        """Copy-with-changes the node ``node_id`` in place, anywhere in subtree."""
        return self.all_nodes.update(node_id, **changes)

    def set_node(self, node_id: str, new_node: Node) -> Node:
        """Swap the node ``node_id`` for ``new_node`` in place."""
        return self.all_nodes.replace(node_id, new_node)

    def remove_node(self, node_id: str) -> Node:
        """Drop the node ``node_id`` from the subtree and return it."""
        return self.all_nodes.remove(node_id)

    def add_child(self, child: "Graph") -> "Graph":
        """Attach ``child`` under the agent named by its ``parent_agent_id``."""
        parent = self[child.parent_agent_id] if child.parent_agent_id else self
        parent.children[child.agent_id] = child
        return child

    def remove_child(self, agent_id: str) -> "Graph":
        """Detach the agent ``agent_id`` from wherever it hangs in the subtree."""
        for g in self.walk():
            if agent_id in g.children:
                return g.children.pop(agent_id)
        raise KeyError(agent_id)

    def update(self, **fields: Any) -> "Graph":
        """Bulk-set top-level graph fields (``query``, ``model``, ``inputs`` …)."""
        for key, value in fields.items():
            if not hasattr(self, key):
                raise AttributeError(f"Graph has no field {key!r}")
            setattr(self, key, value)
        return self

    # ── token rollups ─────────────────────────────────────────────────

    def tokens(self, *, recursive: bool = True) -> tuple[int, int]:
        """Return ``(input_tokens, output_tokens)`` summed over LLM outputs."""
        graphs = self.walk() if recursive else (self,)
        inp = out = 0
        for g in graphs:
            for n in g.nodes:
                if isinstance(n, LLMOutput):
                    inp += n.input_tokens
                    out += n.output_tokens
        return inp, out

    def total_tokens(self, *, recursive: bool = True) -> int:
        """Total input + output tokens over the subtree (or just self)."""
        inp, out = self.tokens(recursive=recursive)
        return inp + out

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path = ".") -> Path:
        """Persist this run to disk.

        A **directory** path writes the run layout (manifest ``graph.json``
        plus per-agent ``agent.json`` / ``session.jsonl`` / ``latest.json``).
        A **``.json`` file** path writes one nested monolithic snapshot instead.

        See ``docs/internal/run-layout.md``.
        """
        from rflow.graph.run_layout import save_run, save_snapshot

        p = Path(path)
        if p.suffix == ".json" and not p.is_dir():
            return save_snapshot(self, p)
        run_root = p if p.is_dir() or not p.suffix else p.parent
        return save_run(self, run_root)

    @classmethod
    def load(cls, path: str | Path) -> "Graph":
        """Rebuild a :class:`Graph` from a run directory or ``.json`` snapshot."""
        from rflow.graph.run_layout import (
            is_graph_snapshot,
            is_run_manifest,
            load_run,
            load_snapshot,
        )

        p = Path(path)
        if p.is_dir():
            graph_json = p / "graph.json"
            if not graph_json.is_file():
                raise ValueError(f"{p} has no graph.json")
            data = json.loads(graph_json.read_text(encoding="utf-8"))
            if is_run_manifest(data):
                return load_run(p)
            if is_graph_snapshot(data):
                return cls.from_dict(data)
            raise ValueError(f"{graph_json} is not a run manifest or graph snapshot")
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if is_run_manifest(data):
                return load_run(p.parent)
            return cls.from_dict(data)
        return load_snapshot(p)

    # ── out-of-band trajectory editing (pure: copy → edit → return) ───

    def replace_node(
        self, target: "str | Node", node: Node, *, truncate: str = "descendants"
    ) -> "Graph":
        from rflow.graph.replace import replace_node

        return replace_node(self, target, node, truncate=truncate)

    def replace_last_action(
        self, agent_id: str, node: Node, *, truncate: str = "descendants"
    ) -> "Graph":
        from rflow.graph.replace import replace_last_action

        return replace_last_action(self, agent_id, node, truncate=truncate)

    def replace_last_observation(
        self, agent_id: str, node: Node, *, truncate: str = "descendants"
    ) -> "Graph":
        from rflow.graph.replace import replace_last_observation

        return replace_last_observation(self, agent_id, node, truncate=truncate)

    def truncate_after(self, node_id: str, *, descendants: bool = True) -> "Graph":
        from rflow.graph.truncation import truncate_after

        return truncate_after(self, node_id, descendants=descendants)

    def truncate_agent(self, agent_id: str, *, after_seq: int) -> "Graph":
        from rflow.graph.truncation import truncate_agent

        return truncate_agent(self, agent_id, after_seq=after_seq)

    def prune_descendants_spawned_after(self, agent_id: str, seq: int) -> "Graph":
        from rflow.graph.truncation import prune_descendants_spawned_after

        return prune_descendants_spawned_after(self, agent_id, seq)

    def inject(self, *, target: Any, node: Node, mode: str = "append") -> "Graph":
        from rflow.graph.injection import inject

        return inject(self, target=target, node=node, mode=mode)

    def inject_output(
        self, *, target: Any, output: str, content: str | None = None
    ) -> "Graph":
        from rflow.graph.injection import inject_output

        return inject_output(self, target=target, output=output, content=content)

    def retrace_steps(self) -> "list[Graph]":
        """Reconstruct per-tick snapshots of this run (see :mod:`rflow.graph.timeline`)."""
        from rflow.graph.timeline import retrace_steps

        return retrace_steps(self)

    def trace(self) -> "Any":
        """Build a :class:`~rflow.utils.trace.Trace` of per-tick snapshots.

        The viz/viewer layer consumes a ``Trace``; this expands the single final
        graph into a stepped timeline (via :meth:`retrace_steps`) so a graph you
        only kept the final state of still renders a real stepper.
        """
        from rflow.utils.trace import Trace

        return Trace.from_graph(self)

    # ── rendering conveniences (delegate to rflow.utils.viewer) ───────

    @property
    def model_label(self) -> str:
        """Best-effort model name for this agent (for display)."""
        from rflow.utils.viewer import _model_label

        return _model_label(self)

    def tree(self) -> str:
        """Render the subtree as a nested text tree."""
        from rflow.utils.viewer import graph_tree

        return graph_tree(self)

    def transcript(self, *, include_system: bool = True) -> str:
        """Render this agent's trajectory as a chat-log transcript."""
        from rflow.utils.viewer import agent_transcript

        return agent_transcript(self, include_system=include_system)

    def session(self, *, include_system: bool = False) -> str:
        """Render every agent's trajectory in graph order (flat chat log)."""
        from rflow.utils.viewer import graph_session

        return graph_session(self, include_system=include_system)

    def save_html(self, path: str | Path, **kwargs: Any) -> Path:
        """Write an interactive HTML view of this graph to ``path``."""
        from rflow.utils.viewer import save_html

        return save_html(self, path, **kwargs)

    def save_image(self, path: str | Path, **kwargs: Any) -> Path:
        """Write a static image view of this graph to ``path``."""
        from rflow.utils.viewer import save_image

        return save_image(self, path, **kwargs)

    def messages(self, system_prompt: str | None = None) -> list[dict[str, str]]:
        """Render this agent's trajectory as a chat-message list.

        Observations the model should react to become ``user`` turns and the
        model's own replies become ``assistant`` turns; consecutive same-role
        messages are coalesced. This is exactly what the LLM saw, minus any
        engine-added nudge. Pass ``system_prompt`` to override the stored one
        (or pass ``""`` to omit the system message).
        """
        system = system_prompt if system_prompt is not None else self.system_prompt
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        for node in self.nodes:
            if isinstance(node, UserQuery):
                append_message(msgs, "user", node.content)
            elif isinstance(node, LLMOutput):
                append_message(msgs, "assistant", node.reply)
            elif (
                isinstance(node, CodeObservation) and not node.terminal and node.content
            ):
                append_message(msgs, "user", node.content)
        return msgs

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "depth": self.depth,
            "query": self.query,
            "system_prompt": self.system_prompt,
            "inputs": dict(self.inputs),
            "model": self.model,
            "max_iters": self.max_iters,
            "output_schema": self.output_schema,
            "parent_agent_id": self.parent_agent_id,
            "parent_node_id": self.parent_node_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "children": {aid: c.to_dict() for aid, c in self.children.items()},
        }

    _FIELDS = frozenset(
        {
            "agent_id",
            "depth",
            "query",
            "system_prompt",
            "inputs",
            "model",
            "max_iters",
            "output_schema",
            "parent_agent_id",
            "parent_node_id",
        }
    )

    @classmethod
    def from_meta_dict(
        cls,
        meta: dict[str, Any],
        *,
        nodes: "list[Node] | None" = None,
        children: "dict[str, Graph] | None" = None,
    ) -> "Graph":
        """Build a :class:`Graph` from a metadata dict plus nodes/children.

        Convenience for hand-constructing graphs (examples, tests): ``meta``
        carries the scalar fields (``agent_id``, ``depth``, ``query`` …) and
        unknown keys are ignored.
        """
        known = {k: v for k, v in meta.items() if k in cls._FIELDS}
        return cls(**known, nodes=list(nodes or []), children=dict(children or {}))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        """Rebuild a :class:`Graph` (and its subtree) from :meth:`to_dict`.

        Tolerant on read: a legacy ``"states"`` key is accepted in place of
        ``"nodes"``, and unknown extra fields are ignored, so older traces load.
        """
        raw_nodes = data.get("nodes", data.get("states", []))
        return cls(
            agent_id=data["agent_id"],
            depth=data.get("depth", 0),
            query=data.get("query", ""),
            system_prompt=data.get("system_prompt", ""),
            inputs=dict(data.get("inputs") or {}),
            model=data.get("model"),
            max_iters=data.get("max_iters"),
            output_schema=data.get("output_schema"),
            parent_agent_id=data.get("parent_agent_id"),
            parent_node_id=data.get("parent_node_id"),
            nodes=[parse_node_obj(n) for n in raw_nodes],
            children={
                aid: cls.from_dict(child)
                for aid, child in (data.get("children") or {}).items()
            },
        )

    def __repr__(self) -> str:
        return (
            f"Graph(agent_id={self.agent_id!r}, depth={self.depth}, "
            f"nodes={len(self.nodes)}, children={len(self.children)})"
        )


__all__ = [
    "ActionNode",
    "ChildHandle",
    "append_message",
    "CodeObservation",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "Graph",
    "LLMAction",
    "LLMOutput",
    "Node",
    "ObservationNode",
    "ResumeAction",
    "SupervisingOutput",
    "UserQuery",
    "WaitRequest",
    "new_id",
    "parse_node_obj",
    "is_observation",
    "is_action",
    "is_code_observation",
    "is_user_query",
    "is_llm_output",
    "is_exec_output",
    "is_supervising",
    "is_errored",
    "is_done",
    "is_llm_action",
    "is_exec_action",
    "is_resume_action",
    "is_resumed",
]
