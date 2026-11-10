"""What a graph holds: ``Node``s, ``Agent``s and the ``Edge``s between them.

A node is text, inputs and code. Running it binds each input as a REPL
variable, then runs the code. An ``Agent`` is a node too, its agent's first:
it adds the agent's name, task and config. Every other node names its agent
in ``owner``.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
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
RETURN = "return"
WRITE = "write"
EDGE_KINDS = (SPAWN, RETURN, WRITE)


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


@dataclass(eq=False, repr=False, slots=True)
class Node:
    """One step of an agent's run: ``text``, ``input`` variables and ``code``.

    ``role`` is ``"user"`` or ``"assistant"``, as in a chat. Running the node
    binds ``input`` as REPL variables and runs ``code``; ``output``, ``result``
    and ``error`` are what that did. ``calls`` are its saved tool calls.
    ``data`` holds flags the engine reads (``seen``, ``terminal``, ``idle``,
    ``usage``, ...); ``metadata`` is open JSON for the application. Everything
    after ``text`` is passed by keyword. ``owner`` (its agent's id), ``seq`` and ``graph`` are set by
    the graph that adds it.
    """

    role: str = "user"
    text: str = ""
    _: KW_ONLY
    input: dict[str, Any] = field(default_factory=dict)
    code: str = PASS
    output: str = ""
    result: Any = None
    error: bool = False
    state: str = DONE
    calls: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Who wrote it: ``"self"`` for the agent, else a sender (``"user"``, an agent id, ...).
    producer: str = ""
    id: str = field(default_factory=lambda: new_id("node"))
    owner: str = ""
    seq: int = -1
    #: Bumped by every edit; a run that started on an older version is discarded.
    version: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    graph: Graph | None = None

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"node role must be 'user' or 'assistant', not {self.role!r}")
        if not isinstance(self.text, str):
            raise TypeError("node text must be a string")
        self.input = dict(self.input or {})
        self.code = self.code or PASS
        self.data = dict(self.data)
        self.producer = self.producer or (SELF if self.role == "assistant" else "user")

    def __repr__(self) -> str:
        return f"Node({self.role!r}, id={self.id!r}, state={self.state!r})"

    @property
    def runs_code(self) -> bool:
        return self.code.strip() not in ("", PASS)

    def _graph(self) -> Graph:
        if self.graph is None:
            raise LookupError(f"{self!r} is not in a graph")
        return self.graph

    @property
    def agent(self) -> Agent:
        """The agent this node belongs to."""
        return self._graph()._roots[self.owner]

    def transcript(self) -> list[Node]:
        """Its agent's nodes up to and including this one: what the agent had seen here."""
        own = self._graph()._own[self.owner]
        return own[: own.index(self) + 1]


@dataclass(eq=False, repr=False, slots=True, kw_only=True)
class Agent(Node):
    """An agent: its first node, plus what only an agent has.

    ``config_name`` names the Flow config it runs with (``None`` for a
    top-level agent, which gets the run's ``root_config``). ``repl_id`` names
    the REPL, workspace and session it runs in: its own id, unless a fork gave
    it a new one. ``config_record`` is the JSON of the config it last ran
    with, for viewers; a Flow never reads it back to run the agent.
    """

    name: str = ""
    task: str = ""
    config_name: str | None = None
    arg_descriptions: dict[str, str] = field(default_factory=dict)
    repl_id: str = ""
    config_record: dict[str, Any] | None = None
    _config: AgentConfig | None = field(default=None, init=False)

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, id={self.id!r})"

    @property
    def config(self) -> AgentConfig:
        """The config a Flow bound to this agent, or the record of the last one that ran it."""
        if self._config is None:
            recorded = self.config_record
            if recorded is None:
                raise AttributeError(f"agent {self.name!r} has no config until a Flow runs it")
            from rlmflow.graph.config import AgentConfig

            self._config = AgentConfig.from_dict(recorded)
        return self._config

    def bind_config(self, config: AgentConfig, *, record: bool = True) -> None:
        """Run this agent with ``config``; ``record`` keeps its JSON in ``config_record``."""
        self._config = config
        if record:
            self.config_record = strict_json(config.to_dict(), "config")

    def nodes(self) -> list[Node]:
        """Its nodes, this one first, in the order they were added."""
        return list(self._graph()._own[self.id])

    def last_turn(self) -> Node | None:
        """The node that ended its latest turn: the last one with ``seen``."""
        for node in reversed(self._graph()._own[self.id]):
            if "seen" in node.data:
                return node
        return None

    def status(self) -> str:
        """``"done"``, ``"idle"`` (waiting for news), or ``"running"``.

        Unfinished nodes, or nodes from others that arrived after its last turn
        started, mean ``running``. Otherwise the last turn says: ``terminal`` is
        done, ``idle`` is idle, anything else is still working.
        """
        own = self._graph()._own[self.id]
        if own[-1].state != DONE:
            return "running"
        turn = self.last_turn()
        if turn is None:
            return "running"
        if any(node.producer != SELF for node in own[turn.data["seen"] :]):
            return "running"
        if turn.data.get("terminal"):
            return "done"
        if turn.data.get("idle"):
            return "idle"
        return "running"

    def final_result(self) -> Any:
        """What it finished with, or ``None`` until it is done."""
        if self.status() != "done":
            return None
        turn = self.last_turn()
        return turn.result if turn is not None else None

    def parent(self) -> Agent | None:
        """The agent that launched this one; ``None`` when top-level."""
        g = self._graph()
        parent_id = g._parent.get(self.id)
        return g._roots.get(parent_id) if parent_id else None

    def children(self) -> list[Agent]:
        """Agents this one launched, in creation order."""
        g = self._graph()
        return [g._roots[i] for i in g._children.get(self.id, ()) if i in g._roots]

    def subtree(self) -> list[Agent]:
        """This agent and every agent it launched, transitively, depth first."""
        found: list[Agent] = []
        stack: list[Agent] = [self]
        while stack:
            current = stack.pop()
            found.append(current)
            stack.extend(reversed(current.children()))
        return found

    def lineage(self) -> list[Agent]:
        """Its top-level ancestor first, ending with this agent."""
        chain: list[Agent] = [self]
        while (parent := chain[-1].parent()) is not None and parent not in chain:
            chain.append(parent)
        return chain[::-1]

    def path(self) -> str:
        """Agent names from the top, joined with dots: ``root.worker-0``."""
        return ".".join(agent.name for agent in self.lineage())


@dataclass(slots=True)
class Edge:
    """A connection between nodes: ``spawn``, ``return``, ``write``, or an application kind."""

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
    """One agent at one of its nodes.

    Every node that finishes is streamed as a ``Step``; tools see the step of
    the node whose code is running. ``replaying`` marks code rerun to rebuild
    a REPL; ``calls`` is the running node's ``CallLog``, which tools save to.
    """

    graph: Graph
    agent: Agent
    node: Node
    replaying: bool = False
    calls: CallLog | None = None

    @classmethod
    def latest(cls, graph: Graph, agent: Agent) -> Step:
        """The agent at its last node."""
        return cls(graph, agent, agent.nodes()[-1])

    @property
    def config(self) -> AgentConfig:
        return self.agent.config

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
    "IMAGES_KEY",
    "PASS",
    "READY",
    "RETURN",
    "RUNNING",
    "SELF",
    "SPAWN",
    "WRITE",
    "Agent",
    "Edge",
    "LLMUsage",
    "Node",
    "Step",
    "attach_image",
    "current_step",
    "new_id",
    "record_metadata",
    "running_step",
    "strict_json",
]
