"""What a graph holds: ``Node``s and the ``Edge``s between them.

A node is text, inputs and code. Running it binds each input as a REPL
variable, then runs the code. An agent invocation's first node is an
``AgentRoot``: it adds the name, task and configured agent. Every node names
that root in ``root_id``.
"""

from __future__ import annotations

import json
import time
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph

#: Code that does nothing: a node with only text and inputs.
PASS = "pass"

#: Node states: waiting to run, running, finished.
READY = "ready"
RUNNING = "running"
DONE = "done"

#: ``data["producer"]`` of nodes an agent wrote itself; anything else is news to it.
SELF = "self"

#: The ``data`` keys the engine reads. ``seen`` marks the node that ended a turn
#: (how many of the agent's nodes that turn had seen). The turn ended the agent
#: (``terminal``) or left it waiting for news (``idle``). ``final`` marks rlmflow
#: asking for the last turn, ``dead`` a node whose REPL died, ``stopped`` why an
#: agent was stopped, and ``workspace`` the files' snapshot after the node.
SEEN_KEY = "seen"
TERMINAL_KEY = "terminal"
IDLE_KEY = "idle"
FINAL_KEY = "final"
DEAD_KEY = "dead"
STOPPED_KEY = "stopped"
WORKSPACE_KEY = "workspace"
#: The image paths a node's tools attached, shown to the model with its output.
IMAGES_KEY = "images"

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
    after ``text`` is passed by keyword. ``root_id`` (its ``AgentRoot``'s id),
    ``seq`` and ``graph`` are set by the graph that adds it.
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
    root_id: str = ""
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

    @property
    def seen(self) -> int | None:
        """How many of its agent's nodes the turn this node ended had seen; ``None`` if it ended none."""
        return self.data.get(SEEN_KEY)

    @property
    def ends_turn(self) -> bool:
        return SEEN_KEY in self.data

    @property
    def terminal(self) -> bool:
        """It ended its agent."""
        return bool(self.data.get(TERMINAL_KEY))

    @property
    def idle(self) -> bool:
        """It left its agent waiting for news."""
        return bool(self.data.get(IDLE_KEY))

    @property
    def final(self) -> bool:
        """rlmflow asking for the agent's last turn."""
        return bool(self.data.get(FINAL_KEY))

    @property
    def dead(self) -> bool:
        """Its REPL died while running it."""
        return bool(self.data.get(DEAD_KEY))

    @property
    def stopped(self) -> str | None:
        """Why its agent was stopped here, if it was."""
        return self.data.get(STOPPED_KEY)

    @property
    def workspace(self) -> Any:
        """The files' snapshot after this node, if one was taken."""
        return self.data.get(WORKSPACE_KEY)

    def _graph(self) -> Graph:
        if self.graph is None:
            raise LookupError(f"{self!r} is not in a graph")
        return self.graph

    @property
    def root(self) -> AgentRoot:
        """The ``AgentRoot`` whose chain contains this node."""
        return self._graph()._roots[self.root_id]

    def transcript(self, after: Node | None = None) -> list[Node]:
        """Its root's nodes up to and including this one, after ``after`` if given."""
        own = self._graph()._chains[self.root_id]
        start = own.index(after) + 1 if after is not None and after in own else 0
        return own[start : own.index(self) + 1]


def final_node(
    result: Any,
    seen: int,
    *,
    output: str = "",
    error: bool = False,
    data: dict[str, Any] | None = None,
) -> Node:
    """A step that ends its run with ``result``, after ``seen`` of its nodes."""
    return Node(
        "assistant",
        result=result,
        output=output,
        error=error,
        data={SEEN_KEY: seen, TERMINAL_KEY: True, **(data or {})},
    )


@dataclass(eq=False, repr=False, slots=True, kw_only=True)
class AgentRoot(Node):
    """The first node of one agent invocation: its task and chain identity.

    ``agent`` names the Flow agent it runs with. ``repl_id`` names the REPL,
    workspace and session it runs in: its own id, unless a fork gave it a new one.
    """

    name: str = ""
    task: str = ""
    agent: str = "default"
    arg_descriptions: dict[str, str] = field(default_factory=dict)
    repl_id: str = ""

    def __repr__(self) -> str:
        return f"AgentRoot({self.name!r}, id={self.id!r})"

    def nodes(self) -> list[Node]:
        """This root's nodes, in the order they were added."""
        return list(self._graph()._chains[self.id])

    def last_turn(self) -> Node | None:
        """The node that ended its latest step: the last one with ``seen``."""
        for node in reversed(self._graph()._chains[self.id]):
            if node.ends_turn:
                return node
        return None

    def status(self) -> str:
        """``"done"``, ``"idle"`` (waiting for news), or ``"running"``."""
        own = self._graph()._chains[self.id]
        if own[-1].state != DONE:
            return "running"
        turn = self.last_turn()
        if turn is None:
            return "running"
        if any(node.producer != SELF for node in own[turn.seen :]):
            return "running"
        if turn.terminal:
            return "done"
        if turn.idle:
            return "idle"
        return "running"

    def final_result(self) -> Any:
        """The terminal result, or ``None`` until this agent finishes."""
        if self.status() != "done":
            return None
        turn = self.last_turn()
        return turn.result if turn is not None else None

    def parent(self) -> AgentRoot | None:
        """The agent root that launched this one; ``None`` when top-level."""
        parent_id = self._graph()._parent.get(self.id)
        return self._graph()._roots.get(parent_id) if parent_id else None

    def children(self) -> list[AgentRoot]:
        """The agent roots this one launched, in creation order."""
        g = self._graph()
        return [g._roots[i] for i in g._children.get(self.id, ()) if i in g._roots]

    def subtree(self) -> list[AgentRoot]:
        """This root and every root below it, transitively, depth first."""
        found: list[AgentRoot] = []
        stack = [self]
        while stack:
            current = stack.pop()
            found.append(current)
            stack.extend(reversed(current.children()))
        return found

    def lineage(self) -> list[AgentRoot]:
        """Top-level ancestor first, ending with this root."""
        chain = [self]
        while (parent := chain[-1].parent()) is not None and parent not in chain:
            chain.append(parent)
        return chain[::-1]

    def path(self) -> str:
        """Root names from the top, joined with dots: ``root.worker-0``."""
        return ".".join(root.name for root in self.lineage())


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


__all__ = [
    "DEAD_KEY",
    "DONE",
    "EDGE_KINDS",
    "FINAL_KEY",
    "IDLE_KEY",
    "IMAGES_KEY",
    "PASS",
    "READY",
    "RETURN",
    "RUNNING",
    "SEEN_KEY",
    "SELF",
    "SPAWN",
    "STOPPED_KEY",
    "TERMINAL_KEY",
    "WORKSPACE_KEY",
    "WRITE",
    "Edge",
    "LLMUsage",
    "Node",
    "AgentRoot",
    "final_node",
    "new_id",
    "strict_json",
]
