"""In-memory agent transcripts used by the running engine."""

from __future__ import annotations

import contextvars
import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from rlmflow.graph.config import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    agent_ref_id,
    can_spawn,
    input_references,
    validate_agent_name,
)
from rlmflow.graph.prompts import (
    COLD_REPL_NOTE,
    EXECUTION_POLICY,
    FINAL_ANSWER_ACTION,
    FIRST_TURN_SAFEGUARD,
    LAST_TURN_ADDENDUM,
    ORCHESTRATOR_ADDENDUM,
    PLANNING_ACTION,
    TRUNCATION_SUMMARY,
    TURN_ONE_FINISH_SENTENCE,
    USER_PROMPT,
    WORKER_ADDENDUM,
    orchestrator_addendum,
)
from rlmflow.llm import LLMUsage


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


def agent_payload(agent: AgentStart | None) -> dict[str, Any]:
    """The agent fields the run format repeats on every query node."""
    config = agent.config if agent is not None else AgentConfig()
    return {
        "inputs": dict(config.inputs),
        "model": config.model,
        "prompt_profile": config.prompt_profile,
        "output_schema": config.output_schema,
        "child_output_schema": config.child_output_schema,
        "reuse_repl": config.reuse_repl,
        "max_depth": config.max_depth,
        "max_iters": config.max_iters,
        "max_budget": config.max_budget,
        "keep_n_messages": config.keep_n_messages,
        "max_output_length": config.max_output_length,
        "max_query_chars": config.max_query_chars,
        "resume_on_child_done": config.resume_on_child_done,
    }


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
        if isinstance(node, LLMOutput):
            self.usage = self.usage + node.usage
        if isinstance(node, ErrorOutput):
            self.errors.append(node)
        if isinstance(node, ExecAction):
            for agent_id in node.launched_agent_ids.values():
                existing = self.launch_actions_by_agent_id.get(agent_id)
                if existing is not None and existing is not node:
                    raise ValueError(f"agent {agent_id!r} has multiple launch receipts")
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

    def render(self) -> list[dict[str, str]]:
        """This node as zero or more canonical chat messages."""
        return []

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
            messages = node.render()
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
        spent = (node.usage for node in self.walk() if isinstance(node, LLMOutput))
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
        agent = self.parent_agent
        timing = self.timing()
        metadata: dict[str, Any] = {"created_at": _isoformat(self.created_at)}
        if timing:
            metadata["timing"] = timing
        return {
            "id": self.id,
            "parent_id": self.parent.id if self.parent is not None else None,
            "type": self.type,
            "agent_id": agent.config.path if agent is not None else None,
            "order": self.seq,
            "metadata": metadata,
            "payload": {"content": self.content},
        }

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
        self.root = self
        self.parent_agent = self
        self.frontier = self
        self.pending_handoffs = []
        self.dependency_scheduled = False
        self._index = RunIndex()
        self._index.register(self)

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.content}]

    @property
    def terminal(self) -> bool:
        return isinstance(self.frontier, DoneOutput)

    def result(self) -> Any:
        node = self.frontier
        return node.result if isinstance(node, DoneOutput) else None

    @property
    def blocked(self) -> bool:
        """Whether at least one handle-valued input is not terminal."""
        root = self.root
        references = self.input_references
        if not references or root is None:
            return False
        return any(
            (source := root.find_agent(source_id)) is None or not source.terminal
            for _input_name, source_id in references
        )

    @property
    def input_references(self) -> tuple[tuple[str, str], ...]:
        """Return persistent ``(input_name, source_id)`` dependency bindings."""
        return input_references(self.config.inputs)

    def resolved_inputs(self) -> dict[str, str]:
        """Materialize handle-valued launch inputs for this agent's REPL."""
        root = self.root
        if root is None:
            raise RuntimeError("cannot resolve inputs for a detached agent")
        resolved: dict[str, str] = {}
        for name, value in self.config.inputs.items():
            if isinstance(value, str):
                resolved[name] = value
                continue
            source_id = agent_ref_id(value)
            if source_id is None:
                raise TypeError(f"input {name!r} must be a string or agent handle")
            source = root.find_agent(source_id)
            if source is None:
                raise ValueError(f"input {name!r} references missing agent {source_id!r}")
            if not source.terminal:
                raise RuntimeError(f"input {name!r} source {source.config.path!r} is not complete")
            result = source.result()
            resolved[name] = (
                result
                if isinstance(result, str)
                else json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        return resolved

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

    def find_agent_by_path(self, path: str) -> AgentStart | None:
        """Find an agent by its stable dotted path without scanning the tree."""
        self._require_run_root()
        return self._index.agents_by_path.get(path)

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

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update(agent_payload(self))
        data["payload"]["system_prompts"] = dict(self.system_prompts)
        return data

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
class UserQuery(Node):
    type: ClassVar[str] = "user_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    finish_description: ClassVar[str] = "Submit your final answer."

    def instruction(self) -> str:
        return self.content

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.instruction()}]

    def build_system_prompt(self, flow: Any) -> str:
        """System prompt for this turn. Default is the flow's inherited protocol."""
        return flow.build_system_prompt(self)

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update(agent_payload(self.parent_agent))
        return data


@dataclass
class PlanQuery(UserQuery):
    type: ClassVar[str] = "plan_query"

    def instruction(self) -> str:
        agent = self.parent_agent
        if agent is None:
            turns = 0
            max_iters = None
        else:
            turns = sum(
                isinstance(node, LLMOutput) and node.seq < self.seq for node in agent.transcript()
            )
            max_iters = agent.config.max_iters
        if self.content:
            body = self.content
        else:
            if max_iters is None:
                body = f"Turn {turns + 1}:"
            else:
                body = USER_PROMPT.format(iter_1=turns + 1, max_iter=max_iters)
            if turns == 0:
                body = FIRST_TURN_SAFEGUARD + body
        return f"{body}\n\n{PLANNING_ACTION}"


@dataclass
class FinalQuery(UserQuery):
    type: ClassVar[str] = "final_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.FINAL
    finish_description: ClassVar[str] = "Submit your final answer now."
    content: str = FINAL_ANSWER_ACTION

    def build_system_prompt(self, flow: Any) -> str:
        text = flow.build_system_prompt(self).replace(TURN_ONE_FINISH_SENTENCE, "").rstrip()
        return f"{text}\n\n{LAST_TURN_ADDENDUM}"


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
    #: Stable proxy-call receipt: launch call ordinal -> created agent id.
    launched_agent_ids: dict[int, str] = field(default_factory=dict)

    def mark_agent_retrieved(self, agent_id: str) -> None:
        """Record one result access idempotently."""
        if agent_id not in self.retrieved_agent_ids:
            self.retrieved_agent_ids.append(agent_id)

    def record_agent_launch(self, call_id: int, agent: AgentStart) -> None:
        """Bind one launch call to its child without changing ownership."""
        existing_id = self.launched_agent_ids.get(call_id)
        if existing_id is not None and existing_id != agent.id:
            raise ValueError(f"launch call {call_id} already points to agent {existing_id!r}")
        root = self.root
        if root is None:
            raise RuntimeError("cannot record a launch on a detached action")
        existing_action = root._index.launch_actions_by_agent_id.get(agent.id)
        if existing_action is not None and existing_action is not self:
            raise ValueError(f"agent {agent.id!r} already has a launch receipt")
        self.launched_agent_ids[call_id] = agent.id
        root._index.launch_actions_by_agent_id[agent.id] = self

    def launch_call_id_for(self, agent_id: str) -> int | None:
        """Return the first call on this action that produced ``agent_id``."""
        return min(
            (
                call_id
                for call_id, launched_id in self.launched_agent_ids.items()
                if launched_id == agent_id
            ),
            default=None,
        )

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"] = {
            "code": self.code,
            "repl_execution_order": self.repl_execution_order,
            "retrieved_agent_ids": list(self.retrieved_agent_ids),
            "launched_agent_ids": {
                str(call_id): agent_id for call_id, agent_id in self.launched_agent_ids.items()
            },
        }
        return data


@dataclass
class AppendChild(ExecAction):
    """A controller-injected action that launches attached agent subtrees."""

    type: ClassVar[str] = "append_child"

    @property
    def child_agents(self) -> list[AgentStart]:
        return [child for child in self.children if isinstance(child, AgentStart)]

    def child(self, name: str) -> AgentStart | None:
        return next(
            (child for child in self.child_agents if child.config.name == name),
            None,
        )

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
        calls = [
            (f"launch_subagent('', model={child.config.model!r}, name={child.config.name!r})")
            for child in self.child_agents
        ]
        for call_id, child in enumerate(self.child_agents):
            self.record_agent_launch(call_id, child)
        if len(calls) == 1:
            self.code = f"_handle = await {calls[0]}\nprint(await _handle.wait_for_result())"
            return
        self.code = (
            "_handles = [\n"
            + "".join(f"    await {call},\n" for call in calls)
            + "]\nprint([await h.wait_for_result() for h in _handles])"
        )


@dataclass
class OutputNode(Node):
    """Output produced by a model, runtime, or terminal engine step."""

    type: ClassVar[str] = "output_node"


@dataclass
class LLMOutput(OutputNode):
    type: ClassVar[str] = "llm_output"
    code: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: Key into the owning agent's ``system_prompts`` table.
    prompt_id: str = ""

    def render(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.content}]

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"]["code"] = self.code
        agent = self.parent_agent
        config = agent.config if agent is not None else AgentConfig()
        metadata: dict[str, Any] = {
            "model": config.model,
            "usage": asdict(self.usage),
            "system_prompt": self.prompt_id,
        }
        if config.keep_n_messages is not None:
            metadata["keep_n_messages"] = config.keep_n_messages
        data["metadata"] = {**metadata, **data["metadata"]}
        return data


@dataclass
class ExecOutput(OutputNode):
    type: ClassVar[str] = "exec_output"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.content}]

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"]["output"] = self.content
        return data


@dataclass
class ErrorOutput(OutputNode):
    type: ClassVar[str] = "error_output"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    error: str = "exec"

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.content}]

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update({"error": self.error, "output": self.content})
        return data


@dataclass
class ReplDead(ErrorOutput):
    type: ClassVar[str] = "repl_dead"
    error: str = "repl"
    content: str = COLD_REPL_NOTE


@dataclass
class DoneOutput(OutputNode):
    type: ClassVar[str] = "done_output"
    result: Any = None

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update({"result": self.result, "output": self.content})
        return data


def start(query: str = "", *, config: AgentConfig | None = None, **overrides: Any) -> AgentStart:
    """A root agent to hand to ``Flow.run``.

    ``config`` is the starting point, a fresh :class:`AgentConfig` when omitted, and
    keyword overrides replace fields on a copy of it the way ``AgentConfig.child``
    does — so ``AgentConfig`` stays the one place a default is written down::

        start("find the bug", max_iters=8)

    ``Flow.start`` is the same thing carrying that flow's defaults.
    """
    base = deepcopy(config) if config is not None else AgentConfig()
    if "inputs" in overrides:
        overrides["inputs"] = dict(overrides["inputs"] or {})
    return AgentStart(
        content=query or DEFAULT_QUERY,
        config=replace(base, **overrides) if overrides else base,
    )


__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "AgentBusyError",
    "active_step",
    "running_step",
    "DEFAULT_QUERY",
    "COLD_REPL_NOTE",
    "EXECUTION_POLICY",
    "FINAL_ANSWER_ACTION",
    "FIRST_TURN_SAFEGUARD",
    "LAST_TURN_ADDENDUM",
    "ORCHESTRATOR_ADDENDUM",
    "PLANNING_ACTION",
    "TRUNCATION_SUMMARY",
    "TURN_ONE_FINISH_SENTENCE",
    "USER_PROMPT",
    "WORKER_ADDENDUM",
    "orchestrator_addendum",
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
    "PlanQuery",
    "ReplDead",
    "RunStats",
    "TruncationSummary",
    "TurnMode",
    "UserQuery",
    "can_spawn",
    "new_agent_id",
    "new_node_id",
    "start",
    "validate_agent_name",
]
