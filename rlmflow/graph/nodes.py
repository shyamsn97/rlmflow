"""What a graph holds: ``Node``s in agents' histories and ``Edge``s between them.

A node is text, inputs and code. Running it binds each input as a REPL
variable, then runs the code. An agent is its root node, which carries
``RootData``: the agent's name, task and history.
"""

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
    from rlmflow.graph.calls import CallLog
    from rlmflow.graph.config import AgentConfig
    from rlmflow.graph.graph import Graph

#: Code that does nothing: a node with only text and inputs.
PASS = "pass"

#: Node states: waiting to run, running, finished.
READY = "ready"
RUNNING = "running"
DONE = "done"

#: ``data["producer"]`` of nodes an agent wrote itself; anything else is news to it.
SELF = "self"

#: The edge kinds the engine writes and reads; any other kind is the application's.
SPAWN = "spawn"
FORK = "fork"
RETURN = "return"
WRITE = "write"
EDGE_KINDS = (SPAWN, FORK, RETURN, WRITE)


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


class RootData:
    """What only an agent's root node has: its name, task and history.

    ``config_name`` names the Flow config it runs with (``None`` for a root,
    which gets the run's ``root_config``). ``history`` is the agent's own nodes,
    the root first. ``metadata`` is open JSON for the application.
    """

    __slots__ = (
        "name",
        "task",
        "config_name",
        "arg_descriptions",
        "history",
        "metadata",
        "_config",
    )

    def __init__(
        self,
        name: str,
        task: str,
        *,
        config_name: str | None = None,
        arg_descriptions: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.task = task
        self.config_name = config_name
        self.arg_descriptions: dict[str, str] = dict(arg_descriptions or {})
        self.history: list[Node] = []
        self.metadata: dict[str, Any] = {}
        self._config: AgentConfig | None = None

    @property
    def config(self) -> AgentConfig:
        """The config a Flow bound to this agent, or the record of the last one that ran it."""
        if self._config is None:
            recorded = self.metadata.get("config")
            if recorded is None:
                raise AttributeError(f"agent {self.name!r} has no config until a Flow runs it")
            from rlmflow.graph.config import AgentConfig

            self._config = AgentConfig.from_dict(recorded)
        return self._config

    def bind_config(self, config: AgentConfig, *, record: bool = True) -> None:
        """Run this agent with ``config``; ``record`` keeps its JSON in ``metadata["config"]``."""
        self._config = config
        if record:
            self.metadata["config"] = strict_json(config.to_dict(), "config")


class Node:
    """One step of an agent's history: ``text``, ``input`` variables and ``code``.

    ``role`` is ``"user"`` or ``"assistant"``, as in a chat. Running the node
    binds ``input`` as REPL variables and runs ``code``; ``output``, ``result``
    and ``error`` are what that did. ``calls`` are its saved tool calls.
    ``data`` holds flags the engine reads (``producer``, ``seen``,
    ``terminal``, ``idle``, ``usage``, ...); ``metadata`` is open JSON for the
    application.
    """

    __slots__ = (
        "id",
        "role",
        "text",
        "input",
        "code",
        "calls",
        "output",
        "result",
        "error",
        "state",
        "data",
        "metadata",
        "root_data",
        "owner",
        "seq",
        "version",
        "created_at",
        "started_at",
        "finished_at",
    )

    def __init__(
        self,
        role: str = "user",
        text: str = "",
        *,
        input: dict[str, Any] | None = None,
        code: str = PASS,
        output: str = "",
        result: Any = None,
        error: bool = False,
        state: str = DONE,
        producer: str | None = None,
        **data: Any,
    ) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"node role must be 'user' or 'assistant', not {role!r}")
        if not isinstance(text, str):
            raise TypeError("node text must be a string")
        self.id = new_id("node")
        self.role = role
        self.text = text
        self.input: dict[str, Any] = dict(input or {})
        self.code = code or PASS
        self.calls: list[dict[str, Any]] = []
        self.output = output
        self.result = result
        self.error = error
        self.state = state
        self.data: dict[str, Any] = data
        self.data["producer"] = producer or (SELF if role == "assistant" else "user")
        self.metadata: dict[str, Any] = {}
        self.root_data: RootData | None = None
        #: The id of the root node whose history holds this node.
        self.owner = ""
        self.seq = -1
        #: Bumped by every edit; a run that started on an older version is discarded.
        self.version = 0
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def __repr__(self) -> str:
        if self.root_data is not None:
            return f"Node(root {self.root_data.name!r}, id={self.id!r})"
        return f"Node({self.role!r}, id={self.id!r}, state={self.state!r})"

    @property
    def producer(self) -> str:
        return self.data["producer"]

    @property
    def runs_code(self) -> bool:
        return self.code.strip() not in ("", PASS)


@dataclass(slots=True)
class Edge:
    """A connection between nodes: ``spawn``, ``fork``, ``return``, ``write``, or an application kind."""

    kind: str
    source: str
    target: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("edge"))
    seq: int = -1

    @property
    def structural(self) -> bool:
        """Whether the engine reads this edge."""
        return self.kind in EDGE_KINDS


@dataclass(frozen=True, slots=True)
class Step:
    """One agent (its root node) at one node of its history.

    Every node that finishes is streamed as a ``Step``; tools see the step of
    the node whose code is running. ``replaying`` marks code rerun to rebuild
    a REPL; ``calls`` is the running node's ``CallLog``, which tools save to.
    """

    graph: Graph
    agent: Node
    node: Node
    replaying: bool = False
    calls: CallLog | None = None

    @classmethod
    def latest(cls, graph: Graph, agent: Node) -> Step:
        """The agent at the last node of its history."""
        return cls(graph, agent, graph.history(agent)[-1])

    @property
    def config(self) -> AgentConfig:
        assert self.agent.root_data is not None
        return self.agent.root_data.config

    @property
    def edges(self) -> list[Edge]:
        """The edges into and out of ``node``."""
        return self.graph.edges(touching=self.node.id)


_active_step: contextvars.ContextVar[Step | None] = contextvars.ContextVar(
    "rlmflow_active_step", default=None
)


def current_step() -> Step | None:
    """The step whose code is running, if any."""
    return _active_step.get()


@contextmanager
def running_step(step: Step | None) -> Iterator[None]:
    """Run the enclosed code as part of ``step``."""
    token = _active_step.set(step)
    try:
        yield
    finally:
        _active_step.reset(token)


#: Key of the image paths a node's tools attached, shown to the model with its output.
IMAGES_KEY = "images"


def attach_image(path: str | os.PathLike[str]) -> None:
    """Show the image file at ``path`` to the model along with the running node's output.

    The graph records the path, not the pixels, so the file must outlive the run's
    use of it. Harnesses that don't send images to their model ignore it.
    """
    step = current_step()
    if step is None:
        raise RuntimeError("attach_image can only be called while a node runs")
    if not step.replaying:
        step.node.data.setdefault(IMAGES_KEY, []).append(os.path.abspath(os.fspath(path)))


def record_metadata(
    key: str, value: Any, *, append: bool = False, node: Node | None = None
) -> None:
    """Set ``metadata[key]`` on ``node``, or on the node whose code is running.

    ``append`` keeps a list at ``key``. Values must be JSON. Nothing is recorded
    while code is rerun to rebuild a REPL.
    """
    step = current_step()
    if step is not None and step.replaying:
        return
    target = node or (step.node if step is not None else None)
    if target is None:
        raise RuntimeError("metadata can only be recorded while a node runs")
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
    "DONE",
    "EDGE_KINDS",
    "FORK",
    "IMAGES_KEY",
    "PASS",
    "READY",
    "RETURN",
    "RUNNING",
    "SELF",
    "SPAWN",
    "WRITE",
    "Edge",
    "LLMUsage",
    "Node",
    "RootData",
    "Step",
    "attach_image",
    "current_step",
    "new_id",
    "record_metadata",
    "running_step",
    "strict_json",
]
