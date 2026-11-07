"""What a graph holds: ``Agent`` records, ``Node``s in their histories, and ``Edge``s between them."""

from __future__ import annotations

import contextvars
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from rlmflow.graph.config import AgentConfig
    from rlmflow.graph.graph import Graph

#: The five node types.
LLM = "llm"
CODE = "code"
MESSAGE = "message"
IDLE = "idle"
DONE = "done"
NODE_TYPES = (LLM, CODE, MESSAGE, IDLE, DONE)

#: ``Step.node`` is the agent itself while it has no history yet.
AGENT = "agent"

#: The edge kinds the engine writes and reads; any other kind is the application's.
SPAWN = "spawn"
RETURN = "return"
EDGE_KINDS = (SPAWN, RETURN, MESSAGE)

#: Set on a ``code`` node until its cell has run: it is running, was cut off, or an edit
#: invalidated it. The agent runs it again before anything else.
PENDING = "pending"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def strict_json(value: Any, what: str) -> Any:
    """A deep copy of ``value`` as strict JSON, or ``TypeError``."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be strict JSON") from exc


@dataclass
class LLMUsage:
    """Tokens (and dollars, when a harness reports them) for model calls: one call, or many summed."""

    input_tokens: int = 0
    output_tokens: int = 0
    #: Reported for attribution only; providers already count these in ``output_tokens``.
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def of(cls, value: Any) -> LLMUsage:
        if isinstance(value, LLMUsage):
            return value
        value = value or {}
        return cls(
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            reasoning_tokens=int(value.get("reasoning_tokens", 0)),
        )


class Agent:
    """One agent: its task and arguments, its own history (``nodes``) and undelivered ``inbox``.

    A fork's history continues from ``base``, a node of another agent. ``metadata``
    is open JSON for the application; the engine never reads it.
    """

    __slots__ = (
        "id",
        "name",
        "task",
        "args",
        "config_name",
        "nodes",
        "inbox",
        "base",
        "metadata",
        "seq",
        "created_at",
        "_config",
    )
    #: Lets an agent with no history stand in ``Step.node``.
    type = AGENT

    def __init__(
        self,
        task: str,
        *,
        name: str,
        args: dict[str, Any] | None = None,
        config: str | None = None,
    ) -> None:
        self.id = new_id("agent")
        self.name = name
        self.task = task
        self.args: dict[str, Any] = dict(args or {})
        #: The Flow config it runs with; ``None`` for a root, which gets ``root_config``.
        self.config_name = config
        self.nodes: list[Node] = []
        self.inbox: list[Node] = []
        self.base: Node | None = None
        self.metadata: dict[str, Any] = {}
        self.seq = -1
        self.created_at = time.time()
        self._config: AgentConfig | None = None

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, id={self.id!r})"

    @property
    def agent_id(self) -> str:
        return self.id

    @property
    def config(self) -> AgentConfig:
        """The config a Flow bound to this agent, or the record of the last one that ran it."""
        if self._config is None:
            recorded = self.metadata.get("config")
            if recorded is None:
                raise AttributeError(f"agent {self.id!r} has no config until a Flow steps it")
            from rlmflow.graph.config import AgentConfig

            self._config = AgentConfig.from_dict(recorded)
        return self._config

    def bind_config(self, config: AgentConfig) -> None:
        """Run this agent with ``config`` and record its JSON in ``metadata["config"]``."""
        self._config = config
        self.metadata["config"] = strict_json(config.to_dict(), "config")

    def resolved_args(self) -> dict[str, Any]:
        """This agent's JSON arguments with description wrappers removed."""
        from rlmflow.graph.config import RECURSE, input_parts, transform_argument

        def unwrap(path: str, value: Any) -> Any:
            wrapped = input_parts(value)
            if wrapped is not None:
                return transform_argument(wrapped[0], unwrap, path=path)
            return RECURSE

        return {
            name: transform_argument(value, unwrap, path=name) for name, value in self.args.items()
        }

    def argument_descriptions(self) -> dict[str, str]:
        """Caller-supplied descriptions for direct arguments."""
        from rlmflow.graph.config import input_parts

        return {
            name: wrapped[1]
            for name, value in self.args.items()
            if (wrapped := input_parts(value)) is not None
        }


class Node:
    """One fact in an agent's history: a ``type`` string plus JSON ``data``.

    ``Node("code", code="print(1)", output="1\\n", error=False)``. The graph
    assigns ``agent_id`` and ``seq`` when the node lands. ``metadata`` is open
    JSON for the application; the engine never reads it.
    """

    __slots__ = (
        "type",
        "data",
        "id",
        "agent_id",
        "seq",
        "metadata",
        "created_at",
        "started_at",
        "finished_at",
    )

    def __init__(self, type: str, /, **data: Any) -> None:
        if type not in NODE_TYPES:
            raise ValueError(f"unknown node type {type!r}; expected one of {NODE_TYPES}")
        self.type = type
        self.data: dict[str, Any] = data
        self.id = new_id("node")
        self.agent_id = ""
        self.seq = -1
        self.metadata: dict[str, Any] = {}
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def __repr__(self) -> str:
        return f"Node({self.type!r}, id={self.id!r}, agent={self.agent_id!r})"

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def pending(self) -> bool:
        """A ``code`` node whose cell hasn't finished running."""
        return self.type == CODE and bool(self.data.get(PENDING))


@dataclass(slots=True)
class Edge:
    """A connection: ``spawn``, ``return``, ``message``, or an application kind.

    Endpoints are node or agent ids.
    """

    kind: str
    source: str
    target: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("edge"))
    seq: int = -1

    @property
    def structural(self) -> bool:
        """Whether the engine reads this edge (``spawn``, ``return``, ``message``)."""
        return self.kind in EDGE_KINDS


@dataclass(frozen=True, slots=True)
class Step:
    """One agent at one node. Every node that lands is streamed as a ``Step``.

    ``node`` is the agent itself before its first node, and an empty fork's base node.
    """

    graph: Graph
    agent: Agent
    node: Node | Agent

    @classmethod
    def latest(cls, graph: Graph, agent: Agent) -> Step:
        """The agent at its latest node, or at itself before it has one."""
        history = graph.history(agent)
        return cls(graph, agent, history[-1] if history else agent)

    @property
    def edges(self) -> list[Edge]:
        """The edges into and out of ``node``."""
        return self.graph.edges(touching=self.node.id)


_active_step: contextvars.ContextVar[Step | None] = contextvars.ContextVar(
    "rlmflow_active_step", default=None
)
_active_code: contextvars.ContextVar[Node | None] = contextvars.ContextVar(
    "rlmflow_active_code", default=None
)
_replaying: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rlmflow_replaying", default=False
)


def current_step() -> Step | None:
    """The step this code is running inside, if any."""
    return _active_step.get()


def current_code() -> Node | None:
    """The ``code`` node of the cell this code is running inside, if any."""
    return _active_code.get()


@contextmanager
def running_step(step: Step | None) -> Iterator[None]:
    """Run the enclosed code as part of ``step``."""
    token = _active_step.set(step)
    try:
        yield
    finally:
        _active_step.reset(token)


@contextmanager
def running_cell(node: Node | None) -> Iterator[None]:
    """Save tool calls from the enclosed code on the cell's ``code`` node."""
    token = _active_code.set(node)
    try:
        yield
    finally:
        _active_code.reset(token)


@contextmanager
def replaying(enabled: bool = True) -> Iterator[None]:
    """Code rerun to rebuild a REPL: its metadata writes are dropped."""
    token = _replaying.set(enabled)
    try:
        yield
    finally:
        _replaying.reset(token)


#: Key of the image paths a cell's tools attached, shown to the model with its output.
IMAGES_KEY = "images"


def attach_image(path: str | os.PathLike[str]) -> None:
    """Show the image file at ``path`` to the model along with the running cell's output.

    The graph records the path, not the pixels, so the file must outlive the run's
    use of it. Harnesses that don't send images to their model ignore it.
    """
    if _replaying.get():
        return
    code = current_code()
    if code is None:
        raise RuntimeError("attach_image can only be called while a cell runs")
    code.data.setdefault(IMAGES_KEY, []).append(os.path.abspath(os.fspath(path)))


def record_metadata(
    key: str, value: Any, *, append: bool = False, node: Node | Agent | None = None
) -> None:
    """Set ``metadata[key]`` on ``node``, the running cell, or the current step's node.

    ``append`` keeps a list at ``key``. Values must be JSON.
    """
    if _replaying.get():
        return
    step = current_step()
    target = node or current_code() or (step.node if step is not None else None)
    if target is None:
        raise RuntimeError("metadata can only be recorded during a graph step")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("metadata key must be a non-empty string")
    stored = strict_json(value, f"metadata {key!r}")
    if not append:
        target.metadata[key] = stored
        return
    existing = target.metadata.setdefault(key, [])
    if not isinstance(existing, list):
        raise TypeError(f"metadata {key!r} is not a list")
    existing.append(stored)


__all__ = [
    "AGENT",
    "CODE",
    "DONE",
    "EDGE_KINDS",
    "IDLE",
    "IMAGES_KEY",
    "LLM",
    "MESSAGE",
    "NODE_TYPES",
    "PENDING",
    "RETURN",
    "SPAWN",
    "Agent",
    "Edge",
    "LLMUsage",
    "Node",
    "Step",
    "attach_image",
    "current_code",
    "current_step",
    "new_id",
    "record_metadata",
    "replaying",
    "running_cell",
    "running_step",
    "strict_json",
]
