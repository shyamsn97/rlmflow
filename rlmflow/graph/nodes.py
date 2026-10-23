"""In-memory agent transcripts used by the running engine."""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID, uuid4

from rlmflow.graph.config import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    RECURSE,
    AgentConfig,
    input_parts,
    transform_argument,
    validate_agent_name,
    validate_argument_name,
)
from rlmflow.structured import DEFAULT_OUTPUT_CODEC

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph
    from rlmflow.prompts import Prompt

FINAL_ANSWER_ACTION = """This is your last turn. Give the best supported final answer now:
call `submit(value)` in exactly one fenced `repl` block, or reply in plain text with no
`repl` block. Do not start work that requires another observation."""

TRUNCATION_SUMMARY = """[earlier turns omitted to fit the context window; the most recent
turns follow. The REPL kept running, so variables, imports, and helpers defined in those
turns are still bound — reuse them instead of redefining them.]"""

COLD_REPL_NOTE = """[this agent's REPL was restarted, so variables and imports from
earlier turns are gone. Re-derive whatever you need before using it.]"""

#: Key of the data-only agent reference in saved tool receipts and proxy calls.
AGENT_REF_KEY = "$rlmflow_agent"
#: ``UserQuery.metadata`` key carrying an ask's per-turn terms.
ASK_KEY = "ask"
#: ``ExecAction.metadata`` key listing the agents whose results the action awaited.
WAITED_AGENTS_KEY = "waited_agents"


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


@dataclass
class LLMUsage:
    """Token counts for model calls: one call, or a whole subtree summed.

    Clients report it; graph nodes store it, so it lives at the graph layer.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    #: Reasoning tokens, reported for attribution only. Providers already count
    #: these inside ``output_tokens``, so ``total`` must not add them again — the
    #: token budget in ``budget_nearly_spent`` reads ``total``.
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


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One immutable message stored on a transcript node."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def message_snapshot(messages: object) -> tuple[ChatMessage, ...]:
    """Validate provider-shaped messages and freeze them for transcript history."""
    if not isinstance(messages, list):
        raise TypeError(
            f"build_messages() must return list[dict[str, str]], not {type(messages).__name__}"
        )
    snapshot: list[ChatMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {index} must be a dict, not {type(message).__name__}")
        if set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain exactly 'role' and 'content'")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not role:
            raise TypeError(f"message {index} role must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError(f"message {index} content must be a string")
        snapshot.append(ChatMessage(role, content))
    return tuple(snapshot)


@dataclass(frozen=True, slots=True)
class Step:
    """One engine step: the graph, the agent taking the step, and its frontier.

    The agent is explicit because an empty fork steps from its base agent's
    node.
    """

    graph: Graph
    agent: AgentStart
    node: Node


_active_step: contextvars.ContextVar[Step | None] = contextvars.ContextVar(
    "rlmflow_active_step", default=None
)
_metadata_writes: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "rlmflow_metadata_writes", default=True
)


def _isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


def system_prompt_id(text: str) -> str:
    """Stable, content-addressed id for a system prompt (the on-disk table key).

    Identical prompts collapse to one id, so it is a natural dedup key and stays
    stable across runs (handy for diffing whether the prompt changed).
    """
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return "sys_" + digest[:12]


def current_step() -> Step | None:
    """The step this code is running inside, if any.

    Each step is its own asyncio task, so this is per-step for free.
    """
    return _active_step.get()


def active_step() -> Node | None:
    """The frontier node of the step this code is running inside, if any."""
    step = _active_step.get()
    return step.node if step is not None else None


def record_metadata(
    key: str,
    value: Any,
    *,
    append: bool = False,
    node: Node | None = None,
) -> None:
    """Attach metadata to ``node`` or the currently executing graph step.

    Host-side custom tools can call this without receiving a Flow or Node.
    Direct worker-only tools have no host graph context and must be proxied if
    they need to annotate execution.
    """
    if not _metadata_writes.get():
        return
    target = node or active_step()
    if target is None:
        raise RuntimeError("metadata can only be recorded during a graph step")
    target.annotate(key, value, append=append)


@contextmanager
def metadata_writes(enabled: bool) -> Iterator[None]:
    """Enable or suppress execution-scoped metadata writes."""
    token = _metadata_writes.set(enabled)
    try:
        yield
    finally:
        _metadata_writes.reset(token)


@contextmanager
def running_step(step: Step | None) -> Iterator[None]:
    """Run the enclosed code as part of ``step``."""
    token = _active_step.set(step)
    try:
        yield
    finally:
        _active_step.reset(token)


class TurnMode(StrEnum):
    """What kind of REPL exit a model turn must produce."""

    NONE = "none"
    ACTION = "action"
    FINAL = "final"


@dataclass
class Node:
    id: str = field(default_factory=new_node_id)
    type: ClassVar[str] = "node"
    turn_mode: ClassVar[TurnMode] = TurnMode.NONE
    content: str = ""
    #: Durable, application-defined JSON metadata. Core graph structure remains
    #: typed; tools, evaluators, caches, and consumers can attach open telemetry
    #: here without adding a field or node class for every integration.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The owning agent; the recipient for an undelivered ``UserQuery``.
    agent_id: str = ""
    #: Position in the graph log, assigned by ``Graph.add``.
    seq: int = -1
    created_at: float = field(default_factory=time.time, compare=False)
    #: When the step that produced this node ran. Nodes written from inside a step
    #: (a nudge, a final-answer prod) were not executed, so they have no timing.
    started_at: float | None = None
    finished_at: float | None = None
    _messages: tuple[ChatMessage, ...] | None = field(
        default=(),
        repr=False,
        compare=False,
    )

    def annotate(self, key: str, value: Any, *, append: bool = False) -> None:
        """Record one JSON-safe metadata value and mark the graph dirty.

        ``append=True`` maintains a list at ``key`` and is intended for receipts
        such as tool or environment calls.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata key must be a non-empty string")
        try:
            stored = json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"metadata {key!r} must be strict JSON") from exc
        if append:
            existing = self.metadata.setdefault(key, [])
            if not isinstance(existing, list):
                raise TypeError(f"metadata {key!r} is not a list")
            existing.append(stored)
        else:
            self.metadata[key] = stored

    def instruction(self, flow: Flow | None = None) -> str:
        """Return this node's primary current-turn instruction text."""
        return self.content

    def build_system_prompt(
        self,
        system_prompt: Prompt,
        flow: Flow,
    ) -> Prompt:
        """Customize and return Flow's prepared system prompt."""
        return system_prompt

    def build_messages(
        self,
        user_prompt: Prompt,
        flow: Flow,
    ) -> list[dict[str, str]]:
        """Build only this node's current-turn messages."""
        content = str(user_prompt)
        return [{"role": "user", "content": content}] if content else []

    @property
    def messages(self) -> tuple[ChatMessage, ...] | None:
        """The durable message snapshot, or ``None`` before a query is built."""
        return self._messages

    def snapshot_messages(self, messages: list[dict[str, str]]) -> tuple[ChatMessage, ...]:
        """Store one validated message snapshot; an existing snapshot is immutable."""
        snapshot = message_snapshot(messages)
        if self._messages is None:
            self._messages = snapshot
        elif self._messages != snapshot:
            raise RuntimeError(f"node {self.id!r} already has a different message snapshot")
        return self._messages

    def message_dicts(self) -> list[dict[str, str]]:
        """Return mutable provider dictionaries copied from the durable snapshot."""
        if self._messages is None:
            raise RuntimeError(f"node {self.id!r} has no message snapshot")
        return [message.as_dict() for message in self._messages]

    def _history_message_dicts(self) -> list[dict[str, str]]:
        """Snapshot an unconsumed injected query before projecting it as history."""
        if self._messages is None:
            content = self.instruction()
            self.snapshot_messages([{"role": "user", "content": content}] if content else [])
        return self.message_dicts()

    def timing(self) -> dict[str, Any]:
        """How long the step that produced this node ran for."""
        if self.started_at is None or self.finished_at is None:
            return {}
        return {
            "started_at": _isoformat(self.started_at),
            "finished_at": _isoformat(self.finished_at),
            "duration_ms": round((self.finished_at - self.started_at) * 1000, 3),
        }


@dataclass
class AgentStart(Node):
    type: ClassVar[str] = "agent_start"
    id: str = field(default_factory=new_agent_id)
    content: str = DEFAULT_QUERY
    args: dict[str, Any] = field(default_factory=dict)
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    config: AgentConfig = field(default_factory=AgentConfig)
    #: Content-addressed table of the system prompts this agent's turns ran under:
    #: ``system_prompt_id(text) -> text``. Each ``LLMOutput`` keeps only the id, so
    #: the text is stored once and a saved run reconstructs every turn's prompt.
    system_prompts: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self._messages is None:
            self._messages = (ChatMessage("user", self.content),) if self.content else ()
        self.agent_id = self.agent_id or self.id

    def resolved_args(self) -> dict[str, Any]:
        """This agent's JSON arguments with description wrappers removed."""

        def unwrap(path: str, value: Any) -> Any:
            wrapped = input_parts(value)
            if wrapped is not None:
                return transform_argument(wrapped[0], unwrap, path=path)
            return RECURSE

        return {
            name: transform_argument(value, unwrap, path=name) for name, value in self.args.items()
        }

    def argument_descriptions(self) -> dict[str, str]:
        """Return caller-supplied descriptions for direct arguments."""
        return {
            name: wrapped[1]
            for name, value in self.args.items()
            if (wrapped := input_parts(value)) is not None
        }

    def record_prompt(self, text: str) -> str:
        """Keep a system prompt in this agent's table and return its id."""
        prompt_id = system_prompt_id(text)
        self.system_prompts.setdefault(prompt_id, text)
        return prompt_id

    def system_prompt_for(self, node: Node) -> str | None:
        """The prompt text an ``LLMOutput`` ran under, or ``None`` if unrecorded."""
        prompt_id = getattr(node, "prompt_id", "")
        return self.system_prompts.get(prompt_id) if prompt_id else None


@dataclass
class TransitionDecision(Node):
    """One durable choice among the semantic edges eligible at a source node."""

    type: ClassVar[str] = "transition_decision"
    candidates: dict[str, str] = field(default_factory=dict)
    selected_rule_id: str | None = None
    source_node_id: str = ""
    read_set_node_ids: list[str] = field(default_factory=list)
    read_set_hash: str = ""
    model: str = "default"
    prompt_hash: str = ""
    engine_metadata: dict[str, Any] = field(default_factory=dict)
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass
class UserQuery(Node):
    type: ClassVar[str] = "user_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass
class RepairQuery(UserQuery):
    """Ask the same agent to repair a rejected submitted candidate."""

    type: ClassVar[str] = "repair_query"


@dataclass
class FinalQuery(UserQuery):
    type: ClassVar[str] = "final_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.FINAL
    content: str = FINAL_ANSWER_ACTION


@dataclass
class TruncationSummary(UserQuery):
    type: ClassVar[str] = "truncation_summary"
    content: str = TRUNCATION_SUMMARY


@dataclass
class ActionNode(Node):
    """Engine work performed without opening an LLM turn."""

    type: ClassVar[str] = "action_node"


#: Tool bindings whose receipts launch agents.
LAUNCH_TOOLS = ("launch_agent", "launch_subagent")


@dataclass
class ExecAction(ActionNode):
    type: ClassVar[str] = "exec_action"
    code: str = ""
    #: Turn contract selected by the query that produced this action.
    turn_mode: TurnMode = TurnMode.NONE
    #: Successful restorable tool results, grouped by tool and per-tool call index.
    saved_tool_calls: dict[str, dict[int, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.turn_mode = TurnMode(self.turn_mode)

    def record_tool_call(self, tool_name: str, call_id: int, result: Any) -> None:
        """Save one successful tool result and update derived launch indexes."""
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("saved tool name must be a non-empty string")
        if not isinstance(call_id, int) or call_id < 0:
            raise ValueError("saved tool call id must be a non-negative integer")
        calls = self.saved_tool_calls.setdefault(tool_name, {})
        if call_id in calls and calls[call_id] != result:
            raise ValueError(
                f"{tool_name} call {call_id} already saved {calls[call_id]!r}, not {result!r}"
            )
        calls[call_id] = result
        if tool_name in LAUNCH_TOOLS and agent_ref_id(result) is None:
            raise ValueError(f"saved {tool_name} result must be an agent reference")

    def tool_calls(self, tool_name: str) -> dict[int, Any]:
        """Return saved calls for one tool."""
        return self.saved_tool_calls.get(tool_name, {})

    def launch_agent_ids(self) -> list[str]:
        """Agents launched by this action, in call order per launch binding."""
        return [
            agent_id
            for tool_name in LAUNCH_TOOLS
            for _call_id, result in sorted(self.tool_calls(tool_name).items())
            if (agent_id := agent_ref_id(result)) is not None
        ]


@dataclass
class BackendRun(ActionNode):
    """One durable invocation of a host-configured external agent."""

    type: ClassVar[str] = "backend_run"
    backend: str = ""
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend run backend must be a non-empty string")
        try:
            UUID(self.run_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("backend run run_id must be a UUID string") from exc
        if not isinstance(self.started, bool):
            raise TypeError("backend run started must be a boolean")


@dataclass
class AppendChild(ExecAction):
    """A controller-injected action that launches attached agent subtrees."""

    type: ClassVar[str] = "append_child"

    @classmethod
    def launching(cls, agents: Sequence[AgentStart]) -> AppendChild:
        """An action whose launch receipts resolve to ``agents`` when it runs or replays."""
        calls = "".join("    launch_agent(task=''),\n" for _agent in agents)
        action = cls(code=f"_handles = [\n{calls}]\nprint(await asyncio.gather(*_handles))")
        for call_id, agent in enumerate(agents):
            action.record_tool_call("launch_agent", call_id, {AGENT_REF_KEY: agent.id})
        return action


@dataclass
class OutputNode(Node):
    """Output produced by a model, runtime, or terminal engine step."""

    type: ClassVar[str] = "output_node"


@dataclass
class LLMOutput(OutputNode):
    type: ClassVar[str] = "llm_output"
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    code: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: Key into the owning agent's ``system_prompts`` table.
    prompt_id: str = ""

    def __post_init__(self) -> None:
        if self._messages is None:
            self._messages = (ChatMessage("assistant", self.content),)


@dataclass
class ExecOutput(OutputNode):
    type: ClassVar[str] = "exec_output"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._messages is None:
            self._messages = (ChatMessage("user", self.content),)


@dataclass
class ErrorOutput(OutputNode):
    type: ClassVar[str] = "error_output"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    error: str = "exec"
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._messages is None:
            self._messages = (ChatMessage("user", self.content),)


@dataclass
class ReplDead(ErrorOutput):
    type: ClassVar[str] = "repl_dead"
    error: str = "repl"
    content: str = COLD_REPL_NOTE


@dataclass
class VerificationFailure(OutputNode):
    """One schema or verifier rejection of a submitted candidate."""

    type: ClassVar[str] = "verification_failure"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    candidate: Any = None
    kind: str = "verifier"
    feedback: str = ""
    schema_hash: str | None = None
    verifier_hash: str | None = None
    observations: str = ""
    _messages: tuple[ChatMessage, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.content:
            self.content = self.feedback
        if self._messages is None:
            self._messages = (ChatMessage("user", self.content),)


@dataclass
class DoneOutput(OutputNode):
    type: ClassVar[str] = "done_output"
    result: Any = None
    schema_hash: str | None = None
    verifier_hash: str | None = None
    verified: bool = False


def build_agent(
    task: str,
    args: Mapping[str, Any] | None = None,
    *,
    config: AgentConfig | None = None,
) -> AgentStart:
    """Build one agent from a task, JSON arguments, and runtime configuration."""
    if not isinstance(task, str):
        raise TypeError("agent task must be a string")
    values = dict(args or {})
    for name in values:
        validate_argument_name(name)
    return AgentStart(
        content=task or DEFAULT_QUERY,
        args=values,
        config=normalize_schemas(deepcopy(config) if config is not None else AgentConfig()),
    )


def normalize_schemas(config: AgentConfig, codec: Any = None) -> AgentConfig:
    """Store output schemas as JSON Schema so the graph stays serializable."""
    codec = codec or DEFAULT_OUTPUT_CODEC
    changes: dict[str, Any] = {}
    if config.output_schema is not None:
        changes["output_schema"] = codec.normalize(config.output_schema)
    if config.child_config.get("output_schema") is not None:
        child = dict(config.child_config)
        child["output_schema"] = codec.normalize(child["output_schema"])
        changes["child_config"] = child
    return replace(config, **changes) if changes else config


__all__ = [
    "AGENT_REF_KEY",
    "ASK_KEY",
    "WAITED_AGENTS_KEY",
    "COLD_REPL_NOTE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "ActionNode",
    "AgentConfig",
    "AgentStart",
    "AppendChild",
    "BackendRun",
    "COLD_REPL_NOTE",
    "ChatMessage",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "FINAL_ANSWER_ACTION",
    "FinalQuery",
    "LAUNCH_TOOLS",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "OutputNode",
    "RepairQuery",
    "ReplDead",
    "Step",
    "TRUNCATION_SUMMARY",
    "TransitionDecision",
    "TruncationSummary",
    "TurnMode",
    "UserQuery",
    "VerificationFailure",
    "active_step",
    "agent_ref_id",
    "build_agent",
    "current_step",
    "message_snapshot",
    "metadata_writes",
    "new_agent_id",
    "new_node_id",
    "normalize_schemas",
    "record_metadata",
    "running_step",
    "validate_agent_name",
]
