"""Graph nodes: one generic ``Node`` with a type string and JSON data."""

from __future__ import annotations

import contextvars
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from rlmflow.graph.config import AgentConfig
    from rlmflow.graph.graph import Graph

#: The six node types.
AGENT_START = "agent_start"
MESSAGE = "message"
LLM_OUTPUT = "llm_output"
CODE = "code"
WAITING = "waiting"
DONE = "done"
NODE_TYPES = (AGENT_START, MESSAGE, LLM_OUTPUT, CODE, WAITING, DONE)

#: Key of the data-only agent reference in saved tool receipts and proxy calls.
AGENT_REF_KEY = "$rlmflow_agent"
#: Tool bindings whose receipts launch agents.
LAUNCH_TOOLS = ("launch_agent",)


def agent_ref_id(value: object) -> str | None:
    """Return the agent id from one exact ``{AGENT_REF_KEY: id}`` reference."""
    if (
        isinstance(value, dict)
        and set(value) == {AGENT_REF_KEY}
        and isinstance(value.get(AGENT_REF_KEY), str)
        and value[AGENT_REF_KEY]
    ):
        return value[AGENT_REF_KEY]
    return None


def new_agent_id() -> str:
    return f"agent_{uuid4().hex}"


def new_node_id() -> str:
    return f"node_{uuid4().hex}"


def strict_json(value: Any, what: str) -> Any:
    """A deep copy of ``value`` as strict JSON, or ``TypeError``."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be strict JSON") from exc


@dataclass
class LLMUsage:
    """Token counts for model calls: one call, or a whole subtree summed."""

    input_tokens: int = 0
    output_tokens: int = 0
    #: Reported for attribution only; providers already count these in ``output_tokens``.
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
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


class Node:
    """One fact in a run: a ``type`` string plus JSON ``data``.

    ``Node("code", code="print(1)", output="1\\n", error=False)``. The graph
    assigns ``agent_id`` and ``seq`` when the node lands.
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
        "_config",
    )

    def __init__(self, type: str, /, **data: Any) -> None:
        if type not in NODE_TYPES:
            raise ValueError(f"unknown node type {type!r}; expected one of {NODE_TYPES}")
        self.type = type
        self.data: dict[str, Any] = data
        self.id = new_agent_id() if type == AGENT_START else new_node_id()
        self.agent_id = ""
        self.seq = -1
        #: Open, application-defined JSON telemetry; not part of the node's meaning.
        self.metadata: dict[str, Any] = {}
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._config: AgentConfig | None = None

    def __repr__(self) -> str:
        return f"Node({self.type!r}, id={self.id!r}, agent={self.agent_id!r})"

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    # -- agent_start --------------------------------------------------------

    @property
    def config_name(self) -> str | None:
        """The Flow config an ``agent_start`` runs with; ``None`` for a root, which gets ``root_config``."""
        if self.type != AGENT_START:
            raise AttributeError(f"a {self.type} node has no config")
        return self.data.get("config")

    @property
    def config(self) -> AgentConfig:
        """The config a Flow bound to this agent, or the record of the last one that ran it."""
        if self._config is None:
            recorded = self.metadata.get("config") if self.type == AGENT_START else None
            if recorded is None:
                raise AttributeError(f"agent {self.id!r} has no config until a Flow steps it")
            from rlmflow.graph.config import AgentConfig

            self._config = AgentConfig.from_dict(recorded)
        return self._config

    def bind_config(self, config: AgentConfig) -> None:
        """Run this agent with ``config`` and record its JSON in ``metadata["config"]``."""
        if self.type != AGENT_START:
            raise AttributeError(f"a {self.type} node has no config")
        self._config = config
        self.annotate("config", config.to_dict())

    @property
    def name(self) -> str:
        """An ``agent_start``'s own name, e.g. ``worker-3``."""
        return self.data["name"]

    @property
    def task(self) -> str:
        return self.data.get("task", "")

    @property
    def args(self) -> dict[str, Any]:
        return self.data.setdefault("args", {})

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

    # -- code ---------------------------------------------------------------

    def record_call(self, tool: str, call_id: int, result: Any) -> None:
        """Save one successful tool result on a ``code`` node, for rebuilding the REPL."""
        if not isinstance(tool, str) or not tool:
            raise ValueError("saved tool name must be a non-empty string")
        if not isinstance(call_id, int) or call_id < 0:
            raise ValueError("saved tool call id must be a non-negative integer")
        result = strict_json(result, f"saved {tool} result")
        calls = self.data.setdefault("calls", {}).setdefault(tool, {})
        key = str(call_id)
        if key in calls and calls[key] != result:
            raise ValueError(f"{tool} call {call_id} already saved {calls[key]!r}, not {result!r}")
        if tool in LAUNCH_TOOLS and agent_ref_id(result) is None:
            raise ValueError(f"saved {tool} result must be an agent reference")
        calls[key] = result

    def calls(self, tool: str) -> dict[int, Any]:
        """Saved results for one tool, keyed by per-tool call index."""
        return {int(key): value for key, value in self.data.get("calls", {}).get(tool, {}).items()}

    def launched(self) -> list[str]:
        """Agent ids launched by this ``code`` node's cell, in call order."""
        return [
            agent_id
            for tool in LAUNCH_TOOLS
            for _call_id, result in sorted(self.calls(tool).items())
            if (agent_id := agent_ref_id(result)) is not None
        ]

    # -- annotations ----------------------------------------------------------

    def annotate(self, key: str, value: Any, *, append: bool = False) -> None:
        """Record one JSON-safe metadata value; ``append`` keeps a list at ``key``."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata key must be a non-empty string")
        stored = strict_json(value, f"metadata {key!r}")
        if append:
            existing = self.metadata.setdefault(key, [])
            if not isinstance(existing, list):
                raise TypeError(f"metadata {key!r} is not a list")
            existing.append(stored)
        else:
            self.metadata[key] = stored

    def timing(self) -> dict[str, Any]:
        """How long the step that produced this node ran for."""
        if self.started_at is None or self.finished_at is None:
            return {}
        return {
            "started_at": isoformat(self.started_at),
            "finished_at": isoformat(self.finished_at),
            "duration_ms": round((self.finished_at - self.started_at) * 1000, 3),
        }


def isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Step:
    """One agent at one node. Every node that lands is streamed as a ``Step``.

    The agent is explicit because an empty fork stands at its base agent's node.
    """

    graph: Graph
    agent: Node
    node: Node


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
    """The pending ``code`` node of the cell this code is running inside, if any."""
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
    """Record tool receipts from the enclosed code on the pending ``code`` node."""
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


def record_metadata(
    key: str, value: Any, *, append: bool = False, node: Node | None = None
) -> None:
    """Attach metadata to ``node``, the running cell, or the current step's node."""
    if _replaying.get():
        return
    step = current_step()
    target = node or current_code() or (step.node if step is not None else None)
    if target is None:
        raise RuntimeError("metadata can only be recorded during a graph step")
    target.annotate(key, value, append=append)


__all__ = [
    "AGENT_REF_KEY",
    "AGENT_START",
    "CODE",
    "DONE",
    "LAUNCH_TOOLS",
    "LLM_OUTPUT",
    "LLMUsage",
    "MESSAGE",
    "NODE_TYPES",
    "Node",
    "Step",
    "WAITING",
    "agent_ref_id",
    "current_code",
    "current_step",
    "isoformat",
    "new_agent_id",
    "new_node_id",
    "record_metadata",
    "replaying",
    "running_cell",
    "running_step",
    "strict_json",
]
