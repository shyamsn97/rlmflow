"""In-memory agent transcripts used by the running engine."""

from __future__ import annotations

import contextvars
import hashlib
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from rlmflow.graph.arguments import RECURSE, transform_argument
from rlmflow.graph.config import (
    AGENT_REF_KEY,
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    agent_ref_id,
    argument_references,
    input_parts,
    validate_agent_name,
    validate_argument_name,
)
from rlmflow.llm import LLMUsage
from rlmflow.prompts.text import (
    COLD_REPL_NOTE,
    FINAL_ANSWER_ACTION,
    TRUNCATION_SUMMARY,
)

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.prompts.sections import Prompt


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


class AgentBusyError(RuntimeError):
    """Raised when something appends to an agent whose step is still in flight.

    Distinct from the frontier check: the frontier is right, but it is about to
    move. Appending here silently discards whatever the running step produces.
    """


_active_step: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "rlmflow_active_step", default=None
)


def new_agent_id() -> str:
    return f"agent_{uuid4().hex}"


def new_node_id() -> str:
    return f"node_{uuid4().hex}"


def _isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


def system_prompt_id(text: str) -> str:
    """Stable, content-addressed id for a system prompt (the on-disk table key).

    Identical prompts collapse to one id, so it is a natural dedup key and stays
    stable across runs (handy for diffing whether the prompt changed).
    """
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return "sys_" + digest[:12]


def active_step() -> Node | None:
    """The node whose step this code is running inside, if any.

    Each step is its own asyncio task, so this is per-step for free. It is what
    lets an agent's own step extend its transcript while everyone else is
    refused.
    """
    return _active_step.get()


@contextmanager
def running_step(node: Node) -> Iterator[None]:
    """Mark ``node``'s agent as busy for the duration of its step."""
    agent = node.parent_agent
    token = _active_step.set(node)
    previous = agent.in_flight if agent is not None else None
    if agent is not None:
        agent.in_flight = node
    try:
        yield
    finally:
        _active_step.reset(token)
        if agent is not None:
            agent.in_flight = previous


class TurnMode(StrEnum):
    """What kind of REPL exit a model turn must produce."""

    NONE = "none"
    ACTION = "action"
    FINAL = "final"


@dataclass(frozen=True)
class RunStats:
    """Immutable aggregate facts about one complete run."""

    revision: int
    node_count: int
    agent_count: int
    node_counts: Mapping[str, int]
    usage: LLMUsage
    error_count: int


@dataclass
class RunIndex:
    """Rebuildable live-run aggregates; never persisted or exposed directly."""

    revision: int = 0
    node_count: int = 0
    node_counts: Counter[str] = field(default_factory=Counter)
    usage: LLMUsage = field(default_factory=LLMUsage)
    agents_by_id: dict[str, AgentStart] = field(default_factory=dict)
    agents_by_path: dict[str, AgentStart] = field(default_factory=dict)
    launch_actions_by_agent_id: dict[str, ExecAction] = field(default_factory=dict)
    errors: list[ErrorOutput] = field(default_factory=list)

    def register(self, node: Node) -> None:
        self.node_count += 1
        self.node_counts[node.type] += 1
        if isinstance(node, AgentStart):
            if node.id in self.agents_by_id:
                raise ValueError(f"duplicate agent id {node.id!r}")
            if node.config.path in self.agents_by_path:
                raise ValueError(f"duplicate agent path {node.config.path!r}")
            self.agents_by_id[node.id] = node
            self.agents_by_path[node.config.path] = node
        if isinstance(node, (LLMOutput, TransitionDecision)):
            self.usage = self.usage + node.usage
        if isinstance(node, ErrorOutput):
            self.errors.append(node)
        if isinstance(node, ExecAction):
            for agent_id in node.launch_agent_ids().values():
                existing = self.launch_actions_by_agent_id.get(agent_id)
                if existing is not None and existing is not node:
                    raise ValueError(f"agent {agent_id!r} has multiple launch calls")
                self.launch_actions_by_agent_id[agent_id] = node

    def snapshot(self) -> RunStats:
        return RunStats(
            revision=self.revision,
            node_count=self.node_count,
            agent_count=len(self.agents_by_id),
            node_counts=MappingProxyType(dict(self.node_counts)),
            usage=replace(self.usage),
            error_count=len(self.errors),
        )


@dataclass
class Node:
    id: str = field(default_factory=new_node_id)
    type: ClassVar[str] = "node"
    turn_mode: ClassVar[TurnMode] = TurnMode.NONE
    content: str = ""
    root: AgentStart | None = None
    parent_agent: AgentStart | None = None
    children: list[Node] = field(default_factory=list, repr=False)
    #: The node this one was appended to; ``None`` on a run's root and until an
    #: append. Out of repr and eq, which would otherwise cycle through children.
    parent: Node | None = field(default=None, repr=False, compare=False)
    #: This node's position in its own agent's transcript; an agent starts at 0.
    #: The tree plus these is the whole run: concurrency changes when nodes are
    #: created, never where they land.
    seq: int = 0
    #: When this node was made. Every node has one, which is what makes a run
    #: orderable end to end: ``seq`` restarts inside each sub-agent, and a node
    #: that never executed has no ``started_at``, but everything is created.
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

    def append(self, node: Node) -> None:
        """Hang ``node`` off this frontier. Same-agent children become the new frontier."""
        agent = self.parent_agent
        if agent is None:
            raise RuntimeError("node is detached")

        if agent.in_flight is not None and active_step() is not agent.in_flight:
            # An agent's own step may extend its transcript (a nudge, a final-answer
            # prod). Anyone else appending while that step is unfinished would move
            # the frontier out from under it and corrupt the transcript.
            raise AgentBusyError(
                f"cannot append to {agent.config.path!r}: its step "
                f"{agent.in_flight.type} ({agent.in_flight.id}) is still in flight"
            )

        if self is not agent.frontier:
            raise ValueError(
                f"cannot append to {self.type} ({self.id}): it is not the frontier of "
                f"{agent.config.path!r}, which is at {agent.frontier.type} "
                f"({agent.frontier.id})"
            )

        if isinstance(node, AgentStart) and any(
            child.config.name == node.config.name for child in agent.sub_agents
        ):
            raise ValueError(f"duplicate child name {node.config.name!r}")

        root = agent.root
        if root is None:
            raise RuntimeError("agent is detached")
        appended = list(node.walk()) if isinstance(node, AgentStart) else [node]
        appended_agent_ids = [child.id for child in appended if isinstance(child, AgentStart)]
        if len(appended_agent_ids) != len(set(appended_agent_ids)):
            raise ValueError("appended subtree contains duplicate agent ids")
        duplicate = next(
            (
                child.id
                for child in appended
                if isinstance(child, AgentStart) and child.id in root._index.agents_by_id
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"duplicate agent id {duplicate!r}")

        self.children.append(node)
        node.parent = self

        if isinstance(node, AgentStart):
            node.root = root
            agent.sub_agents.append(node)
        else:
            node.parent_agent = agent
            node.root = root
            node.seq = self.seq + 1  # a sub-agent keeps its own 0-based sequence
            agent.frontier = node

        for child in appended:
            root._index.register(child)
        root._index.revision += 1

    def append_child(
        self,
        subtree: AgentStart,
        *,
        name: str | None = None,
    ) -> AgentStart:
        """Create or reuse one launch action and attach a subtree beneath it."""
        agent = self.parent_agent
        if agent is None:
            raise RuntimeError("node is detached")

        if isinstance(self, AppendChild):
            action = self
        elif isinstance(self.next, AppendChild) and self.next is agent.frontier:
            action = self.next
        elif self is agent.frontier:
            action = AppendChild()
            self.append(action)
        else:
            raise ValueError(f"{self.id} cannot append children from a stale frontier")

        return action.attach(subtree, name=name)

    @property
    def next(self) -> Node | None:
        """The following node in this node's own agent, or ``None`` at the frontier.

        Most nodes have exactly one child. An ``ExecAction`` that delegated also has
        a child per sub-agent it launched; those are branches, and the agent's own
        sequel is the output of the step that launched them.
        """
        for child in self.children:
            if child.parent_agent is self.parent_agent:
                return child
        return None

    @property
    def prev(self) -> Node | None:
        """The previous node in this node's own agent, or ``None`` at its start.

        An ``AgentStart`` has a parent in the agent that launched it, which is where
        one transcript ends and another begins.
        """
        return None if isinstance(self, AgentStart) else self.parent

    def walk(self) -> Iterator[Node]:
        """Yield this subtree in structural pre-order without Python recursion."""
        yield self
        stack: list[Iterator[Node]] = [iter(self.children)]
        while stack:
            try:
                node = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            yield node
            stack.append(iter(node.children))

    def iter_backwards(self) -> Iterator[Node]:
        """Yield this node through its same-agent predecessors."""
        node: Node | None = self
        while node is not None:
            yield node
            node = node.prev

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

    def project(
        self,
        keep: int | None = None,
    ) -> list[dict[str, str]]:
        """This agent's canonical chat history, walking backwards from here.

        ``keep`` counts flattened messages. Invisible nodes consume no capacity,
        and a multi-message node is trimmed from its oldest messages when only
        part of it fits.
        """
        if keep is not None and keep <= 0:
            return []

        groups: list[list[dict[str, str]]] = []
        remaining = keep
        for node in self.iter_backwards():
            messages = node._history_message_dicts()
            if remaining is not None:
                messages = messages[-remaining:]
            if not messages:
                continue
            groups.append(messages)
            if remaining is not None:
                remaining -= len(messages)
                if remaining == 0:
                    break

        return [message for messages in reversed(groups) for message in messages]

    def subtree_usage(self) -> LLMUsage:
        """Model usage below this node, computed in O(subtree size)."""
        spent = (
            node.usage for node in self.walk() if isinstance(node, (LLMOutput, TransitionDecision))
        )
        return sum(spent, LLMUsage())

    def timing(self) -> dict[str, Any]:
        """How long the step that produced this node ran for."""
        if self.started_at is None or self.finished_at is None:
            return {}
        return {
            "started_at": _isoformat(self.started_at),
            "finished_at": _isoformat(self.finished_at),
            "duration_ms": round((self.finished_at - self.started_at) * 1000, 3),
        }

    def to_record(self) -> dict[str, Any]:
        """This node as one shallow persistence record."""
        from rlmflow.graph import persistence

        return persistence.to_record(self)

    def fork(self) -> AgentStart:
        """Create an independent, fresh-identity branch ending at this node."""
        if self.root is None or self.parent_agent is None:
            raise RuntimeError("node is detached")
        from rlmflow.graph.persistence import clone_until

        return clone_until(self)


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
    sub_agents: list[AgentStart] = field(default_factory=list, repr=False)
    #: Content-addressed table of the system prompts this agent's turns ran under:
    #: ``system_prompt_id(text) -> text``. Each ``LLMOutput`` keeps only the id, so
    #: the text is stored once and a saved run reconstructs every turn's prompt.
    system_prompts: dict[str, str] = field(default_factory=dict, repr=False)
    frontier: Node = field(init=False, repr=False)
    #: The node whose step is currently running for this agent, if any. Engine
    #: bookkeeping, not transcript state: never saved, never compared.
    in_flight: Node | None = field(default=None, init=False, repr=False, compare=False)
    #: Live child-result mailbox. Engine bookkeeping: never saved, never compared.
    pending_handoffs: list = field(default_factory=list, init=False, repr=False, compare=False)
    #: Whether a dependency-blocked agent has been submitted in this process.
    dependency_scheduled: bool = field(default=False, init=False, repr=False, compare=False)
    _index: RunIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._messages is None:
            self._messages = (ChatMessage("user", self.content),) if self.content else ()
        self.root = self
        self.parent_agent = self
        self.frontier = self
        self.pending_handoffs = []
        self.dependency_scheduled = False
        self._index = RunIndex()
        self._index.register(self)

    @property
    def terminal(self) -> bool:
        return isinstance(self.frontier, DoneOutput)

    @property
    def supervisor(self) -> AgentStart | None:
        """The agent whose action launched this agent, or ``None`` for a root."""
        return self.parent.parent_agent if self.parent is not None else None

    def result(self) -> Any:
        node = self.frontier
        return node.result if isinstance(node, DoneOutput) else None

    @property
    def blocked(self) -> bool:
        """Whether at least one handle-valued input is not terminal."""
        root = self.root
        references = self.argument_references
        if not references or root is None:
            return False
        return any(
            (source := root.find_agent(source_id)) is None or not source.terminal
            for _input_name, source_id in references
        )

    @property
    def argument_references(self) -> tuple[tuple[str, str], ...]:
        """Return persistent ``(argument_name, source_id)`` dependency bindings."""
        return argument_references(self.args)

    def resolved_args(self) -> dict[str, Any]:
        """Materialize handle-valued invocation arguments for this agent's REPL."""
        root = self.root
        if root is None:
            raise RuntimeError("cannot resolve inputs for a detached agent")

        def resolve(path: str, value: Any) -> Any:
            wrapped = input_parts(value)
            if wrapped is not None:
                return transform_argument(wrapped[0], resolve, path=path)
            source_id = agent_ref_id(value)
            if source_id is not None:
                source = root.find_agent(source_id)
                if source is None:
                    raise ValueError(f"input {path!r} references missing agent {source_id!r}")
                if not source.terminal:
                    raise RuntimeError(
                        f"input {path!r} source {source.config.path!r} is not complete"
                    )
                return source.result()
            return RECURSE

        return {
            name: transform_argument(value, resolve, path=name)
            for name, value in self.args.items()
        }

    def argument_descriptions(self) -> dict[str, str]:
        """Return caller-supplied descriptions for direct arguments."""
        return {
            name: wrapped[1]
            for name, value in self.args.items()
            if (wrapped := input_parts(value)) is not None
        }

    def invocation_arguments(self) -> dict[str, Any]:
        """Return this invocation's task and resolved JSON arguments."""
        return {"task": self.content, **self.resolved_args()}

    def leaves(self) -> list[Node]:
        out: list[Node] = []
        stack: list[tuple[AgentStart, bool]] = [(self, False)]
        while stack:
            agent, expanded = stack.pop()
            if expanded:
                out.append(agent.frontier)
                continue
            stack.append((agent, True))
            stack.extend((child, False) for child in reversed(agent.sub_agents))
        return out

    def transcript(self) -> list[Node]:
        """This agent's own nodes, start to frontier; sub-agent branches are skipped."""
        return list(self.frontier.iter_backwards())[::-1]

    def retrieved_agent_ids(self) -> set[str]:
        """Agent results explicitly retrieved by this agent across all actions."""
        return {
            agent_id
            for node in self.transcript()
            if isinstance(node, ExecAction)
            for agent_id in node.retrieved_agent_ids
        }

    def _require_run_root(self) -> None:
        if self.root is not self:
            raise RuntimeError("run aggregates belong to agent.root")

    @property
    def usage(self) -> LLMUsage:
        """Total model usage for this complete run in O(1)."""
        self._require_run_root()
        return replace(self._index.usage)

    @property
    def stats(self) -> RunStats:
        """Immutable aggregate facts for this complete run."""
        self._require_run_root()
        return self._index.snapshot()

    def iter_agents(self) -> Iterator[AgentStart]:
        """Yield agents in structural depth-first order."""
        self._require_run_root()
        stack = [self]
        while stack:
            agent = stack.pop()
            yield agent
            stack.extend(reversed(agent.sub_agents))

    def find_agent(self, agent_id: str) -> AgentStart | None:
        """Find an agent by exact runtime ID without scanning transcript nodes."""
        self._require_run_root()
        return self._index.agents_by_id.get(agent_id)

    def errors(self) -> tuple[ErrorOutput, ...]:
        """Errors in append order without scanning the graph."""
        self._require_run_root()
        return tuple(self._index.errors)

    def rebuild_index(self) -> RunIndex:
        """Recompute private run aggregates after isolated graph construction."""
        index = RunIndex()
        for node in self.walk():
            index.register(node)
        index.revision = max(index.node_count - 1, 0)
        self._index = index
        return index

    def llm_turns(self) -> int:
        return sum(isinstance(node, LLMOutput) for node in self.transcript())

    def record_prompt(self, text: str) -> str:
        """Keep a system prompt in this agent's table and return its id."""
        prompt_id = system_prompt_id(text)
        self.system_prompts.setdefault(prompt_id, text)
        return prompt_id

    def system_prompt_for(self, node: Node) -> str | None:
        """The prompt text an ``LLMOutput`` ran under, or ``None`` if unrecorded."""
        prompt_id = getattr(node, "prompt_id", "")
        return self.system_prompts.get(prompt_id) if prompt_id else None

    def latest_system_prompt(self) -> str | None:
        for node in self.frontier.iter_backwards():
            text = self.system_prompt_for(node)
            if text:
                return text
        return None

    def save(self, path: str | Path) -> Path:
        """Write this tree to ``path`` as a run directory, and return it."""
        from rlmflow.graph import persistence  # file work is not part of the model

        return persistence.save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> AgentStart:
        """Read back a run directory written by :meth:`save`."""
        from rlmflow.graph import persistence

        return persistence.load(path)


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
    """Engine work: a REPL block. Does not open an LLM turn."""

    type: ClassVar[str] = "action_node"
    code: str = ""


@dataclass
class ExecAction(ActionNode):
    type: ClassVar[str] = "exec_action"
    #: Submission order within the action's worker session, used for shared replay.
    repl_execution_order: int | None = None
    #: Agent results explicitly read during this action, in first-read order.
    retrieved_agent_ids: list[str] = field(default_factory=list)
    #: Successful restorable tool results, grouped by tool and per-tool call index.
    saved_tool_calls: dict[str, dict[int, Any]] = field(default_factory=dict)

    def mark_agent_retrieved(self, agent_id: str) -> None:
        """Record one result access idempotently."""
        if agent_id not in self.retrieved_agent_ids:
            self.retrieved_agent_ids.append(agent_id)

    def record_tool_call(self, tool_name: str, call_id: int, result: Any) -> None:
        """Save one successful tool result and update derived launch indexes."""
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("saved tool name must be a non-empty string")
        if not isinstance(call_id, int) or call_id < 0:
            raise ValueError("saved tool call id must be a non-negative integer")
        calls = self.saved_tool_calls.setdefault(tool_name, {})
        if call_id in calls and calls[call_id] != result:
            raise ValueError(
                f"{tool_name} call {call_id} already saved {calls[call_id]!r}, "
                f"not {result!r}"
            )
        calls[call_id] = result
        if tool_name != "launch_subagent":
            return
        agent_id = agent_ref_id(result)
        if agent_id is None:
            raise ValueError("saved launch_subagent result must be an agent reference")
        root = self.root
        if root is None:
            raise RuntimeError("cannot record a tool call on a detached action")
        existing_action = root._index.launch_actions_by_agent_id.get(agent_id)
        if existing_action is not None and existing_action is not self:
            raise ValueError(f"agent {agent_id!r} already has a launch call")
        root._index.launch_actions_by_agent_id[agent_id] = self

    def tool_calls(self, tool_name: str) -> dict[int, Any]:
        """Return saved calls for one tool."""
        return self.saved_tool_calls.get(tool_name, {})

    def launch_agent_ids(self) -> dict[int, str]:
        """Return launch call indexes mapped to child IDs."""
        launched: dict[int, str] = {}
        for call_id, result in self.tool_calls("launch_subagent").items():
            agent_id = agent_ref_id(result)
            if agent_id is not None:
                launched[call_id] = agent_id
        return launched

    def launch_call_id_for(self, agent_id: str) -> int | None:
        """Return the first call on this action that produced ``agent_id``."""
        return min(
            (
                call_id
                for call_id, launched_id in self.launch_agent_ids().items()
                if launched_id == agent_id
            ),
            default=None,
        )


@dataclass
class AppendChild(ExecAction):
    """A controller-injected action that launches attached agent subtrees."""

    type: ClassVar[str] = "append_child"

    @property
    def child_agents(self) -> list[AgentStart]:
        return [child for child in self.children if isinstance(child, AgentStart)]

    def append(self, node: Node) -> None:
        super().append(node)
        if isinstance(node, AgentStart):
            self._refresh_code()

    def attach(
        self,
        subtree: AgentStart,
        *,
        name: str | None = None,
    ) -> AgentStart:
        """Rehome a standalone subtree beneath this launch action."""
        parent_agent = self.parent_agent
        if parent_agent is None:
            raise RuntimeError("node is detached")
        if subtree.parent is not None or subtree.root is not subtree:
            raise ValueError("child must be a standalone agent root")

        child_name = name or (
            subtree.config.name
            if subtree.config.name != "root"
            else f"child{len(self.child_agents)}"
        )
        validate_agent_name(child_name)
        if any(child.config.name == child_name for child in parent_agent.sub_agents):
            raise ValueError(f"duplicate child name {child_name!r}")

        old_path = subtree.config.path
        new_path = f"{parent_agent.config.path}.{child_name}"
        depth_delta = parent_agent.config.depth + 1 - subtree.config.depth
        for node in subtree.walk():
            node.root = parent_agent.root
            if isinstance(node, AgentStart):
                suffix = node.config.path.removeprefix(old_path)
                node.config = replace(
                    node.config,
                    name=child_name if node is subtree else node.config.name,
                    path=new_path + suffix,
                    depth=node.config.depth + depth_delta,
                )

        self.append(subtree)
        return subtree

    def _refresh_code(self) -> None:
        calls = ["launch_subagent(task='')" for _child in self.child_agents]
        for call_id, child in enumerate(self.child_agents):
            self.record_tool_call(
                "launch_subagent",
                call_id,
                {AGENT_REF_KEY: child.id},
            )
        if len(calls) == 1:
            self.code = f"_handle = {calls[0]}\nprint(await _handle)"
            return
        self.code = (
            "_handles = [\n"
            + "".join(f"    {call},\n" for call in calls)
            + "]\nprint(await asyncio.gather(*_handles))"
        )


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
class Submission(OutputNode):
    """A locally valid answer candidate awaiting graph acceptance policy."""

    type: ClassVar[str] = "submission"
    result: Any = None
    schema_hash: str | None = None
    verifier_hash: str | None = None
    verified: bool = False


@dataclass
class DoneOutput(OutputNode):
    type: ClassVar[str] = "done_output"
    result: Any = None
    schema_hash: str | None = None
    verifier_hash: str | None = None
    verified: bool = False

    @classmethod
    def from_submission(cls, submission: Submission) -> DoneOutput:
        """Accept a submission without duplicating its completion payload."""
        return cls(
            content=submission.content,
            result=submission.result,
            schema_hash=submission.schema_hash,
            verifier_hash=submission.verifier_hash,
            verified=submission.verified,
        )


def turn_mode_for(node: Node) -> TurnMode:
    """Return the model-turn mode governing a query, reply, action, or result."""
    if isinstance(node, (ExecOutput, ErrorOutput, Submission, DoneOutput)):
        action = node.prev
        return turn_mode_for(action) if isinstance(action, ExecAction) else TurnMode.NONE
    if isinstance(node, ExecAction):
        reply = node.prev
        return turn_mode_for(reply) if isinstance(reply, LLMOutput) else TurnMode.NONE
    if isinstance(node, LLMOutput):
        source = node.prev
        return source.turn_mode if source is not None else TurnMode.NONE
    return node.turn_mode


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
        config=deepcopy(config) if config is not None else AgentConfig(),
    )


def start(
    task: str = "",
    *,
    config: AgentConfig | None = None,
    args: Mapping[str, Any] | None = None,
    **config_overrides: Any,
) -> AgentStart:
    """Build a graph root with explicit arguments and configuration overrides."""
    configured = deepcopy(config) if config is not None else AgentConfig()
    if config_overrides:
        configured = replace(configured, **config_overrides)
    return build_agent(task, args, config=configured)


__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "AgentBusyError",
    "ChatMessage",
    "active_step",
    "running_step",
    "DEFAULT_QUERY",
    "COLD_REPL_NOTE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "AgentConfig",
    "ActionNode",
    "AgentStart",
    "AppendChild",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "FinalQuery",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "OutputNode",
    "ReplDead",
    "RepairQuery",
    "Submission",
    "RunStats",
    "TransitionDecision",
    "TruncationSummary",
    "TurnMode",
    "UserQuery",
    "VerificationFailure",
    "build_agent",
    "new_agent_id",
    "new_node_id",
    "start",
    "turn_mode_for",
    "validate_agent_name",
    "message_snapshot",
]
