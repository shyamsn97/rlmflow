"""In-memory agent transcripts used by the running engine."""

from __future__ import annotations

import contextvars
import hashlib
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

from rlmflow.llm import LLMUsage

DEFAULT_QUERY = """Please read through the provided INPUTS if present and answer any
queries or respond to any instructions contained within it."""
DEFAULT_MAX_QUERY_CHARS = 4_000


class TurnMode(StrEnum):
    """What kind of REPL exit a model turn must produce."""

    NONE = "none"
    ACTION = "action"
    FINAL = "final"


def new_agent_id() -> str:
    return f"agent_{uuid4().hex}"


def new_node_id() -> str:
    return f"node_{uuid4().hex}"


def _isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


class AgentBusyError(RuntimeError):
    """Raised when something appends to an agent whose step is still in flight.

    Distinct from the frontier check: the frontier is right, but it is about to
    move. Appending here silently discards whatever the running step produces.
    """


_active_step: contextvars.ContextVar[Node | None] = contextvars.ContextVar(
    "rlmflow_active_step", default=None
)


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


def system_prompt_id(text: str) -> str:
    """Stable, content-addressed id for a system prompt (the on-disk table key).

    Identical prompts collapse to one id, so it is a natural dedup key and stays
    stable across runs (handy for diffing whether the prompt changed)."""
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    return "sys_" + digest[:12]


@dataclass
class AgentConfig:
    name: str = "root"
    path: str = "root"
    depth: int = 0
    model: str = "default"
    prompt_profile: str = "default"
    inputs: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    #: Place a child in its caller's live Python worker.
    reuse_repl: bool = False
    #: Stable ordinal of the launch call that created this child.
    launch_call_id: int | None = None
    max_depth: int = 1
    #: How many model turns this agent may take. ``None`` explicitly opts out.
    max_iters: int | None = 30
    child_max_iters: int | None = None
    max_budget: int | None = 100_000
    #: How many of the agent's own transcript turns a prompt carries. ``None``
    #: keeps the full history. The system message and the truncation notice sit
    #: on top of this count when it is set.
    keep_n_messages: int | None = None
    max_output_length: int = 4_000
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS

    def child(self, name: str, **overrides: Any) -> AgentConfig:
        validate_agent_name(name)
        values = {
            "name": name,
            "path": f"{self.path}.{name}",
            "depth": self.depth + 1,
            "inputs": {},
            "output_schema": None,
            "reuse_repl": False,
            "launch_call_id": None,
            "max_iters": self.child_max_iters or self.max_iters,
        }
        return replace(self, **{**values, **overrides})


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
    errors: list[ErrorOutput] = field(default_factory=list)

    def register(self, node: Node) -> None:
        self.node_count += 1
        self.node_counts[node.type] += 1
        if isinstance(node, AgentStart):
            if node.id in self.agents_by_id:
                raise ValueError(f"duplicate agent id {node.id!r}")
            self.agents_by_id[node.id] = node
        if isinstance(node, LLMOutput):
            self.usage = self.usage + node.usage
        if isinstance(node, ErrorOutput):
            self.errors.append(node)

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

    def append(self, node: Node) -> Node:
        """Hang a node off this one; whatever appends becomes its agent's frontier."""
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
        return node

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
            action = self.append(AppendChild())
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
    _index: RunIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.root = self
        self.parent_agent = self
        self.frontier = self
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


FINAL_ANSWER_ACTION = """This is your last turn: the run is out of budget to keep working.
Based on the work above, call finish(answer) now with only the final answer, in the
exact form the query requested. Do not investigate further, and do not hold the answer
back for verification — no turn follows this one to read a print, so submit your best
inference rather than ending the run with nothing."""

CONTINUE_NUDGE = """Continue using the REPL environment and determine your answer.
Execute the next step of your plan in exactly one ```repl``` block, or call
finish(...) if you have already printed and verified the final answer."""

TRUNCATION_SUMMARY = """[earlier turns omitted to fit the context window; the most recent
turns follow. The REPL kept running, so variables, imports, and helpers defined in those
turns are still bound — reuse them instead of redefining them.]"""

COLD_REPL_NOTE = """[this agent's REPL was restarted, so variables and imports from
earlier turns are gone. Re-derive whatever you need before using it.]"""

# Adapted from alexzhang13/rlm's current RLM_SYSTEM_PROMPT.
WORKING_ACTION = """As a general strategy, start by probing the available context to
understand it better (e.g. print a few lines, count them, etc.). Then plan briefly
in prose and execute one ```repl``` block. Use its output as feedback for the next
turn, reuse work already in the history, and do not repeat resolved checks."""

ORCHESTRATOR_ADDENDUM = """As an RLM, act as an orchestrator, not a solver.

After the initial probe, state briefly how the task decomposes into subagent and
REPL steps, and sketch the concrete sequence of turns: what each turn computes and
which subagent call, if any, it issues. Then execute one ```repl``` block immediately.

Your own context window is small. Push long-context work that would not fit
comfortably in it into subagents or available query tools instead of pulling that
text into your own history. Conversely, if a Python keyword or regex search, or one
visible passage, already pins the answer, read it directly. Long REPL output pollutes
history too, so print only small results.

Give subagents clean, focused inputs and ask for terse outputs that you can combine
programmatically. Launch independent subagents together before waiting for them.
When work is a deterministic Python loop, complete the loop in one block; do not
spend one model turn per item or batch.

Reserve your own turns for high-level decisions: what to ask next, how to combine
results, and when to finalize."""


def can_spawn(agent: AgentStart | None) -> bool:
    if agent is None:
        return False
    config = agent.config
    return config.max_depth > 0 and config.depth < config.max_depth


@dataclass
class UserQuery(Node):
    type: ClassVar[str] = "user_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION
    finish_description: ClassVar[str] = "Submit your final answer."

    def instruction(self) -> str:
        return self.content

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.instruction()}]

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update(agent_payload(self.parent_agent))
        return data


def working_instruction(query: UserQuery) -> str:
    if query.content:
        return query.content

    from rlmflow.prompts.messages import profile_inputs

    sections = [WORKING_ACTION]
    profile = profile_inputs(query.parent_agent.config.inputs)
    if profile:
        sections.append(profile)
    if can_spawn(query.parent_agent):
        sections.append(ORCHESTRATOR_ADDENDUM)
    return "\n\n".join(sections)


@dataclass
class InspectQuery(UserQuery):
    """Compatibility node for persisted runs; new flows never create it."""

    type: ClassVar[str] = "inspect_query"
    name: ClassVar[str] = "inspect"
    transition_description: ClassVar[str] = "Continue working."
    action_description: ClassVar[str] = "Continue working from the latest result."

    def instruction(self) -> str:
        return working_instruction(self)


@dataclass
class PlanQuery(UserQuery):
    type: ClassVar[str] = "plan_query"
    name: ClassVar[str] = "plan"
    transition_description: ClassVar[str] = "Continue working."
    action_description: ClassVar[str] = "Continue working from the latest result."

    def instruction(self) -> str:
        return working_instruction(self)


@dataclass
class FinalQuery(UserQuery):
    type: ClassVar[str] = "final_query"
    turn_mode: ClassVar[TurnMode] = TurnMode.FINAL
    finish_description: ClassVar[str] = "Submit your final answer now."
    content: str = FINAL_ANSWER_ACTION


@dataclass
class ContinueQuery(UserQuery):
    """Compatibility node for persisted runs; new flows never create it."""

    type: ClassVar[str] = "continue_query"
    content: str = CONTINUE_NUDGE


@dataclass
class TruncationSummary(UserQuery):
    type: ClassVar[str] = "truncation_summary"
    content: str = TRUNCATION_SUMMARY


@dataclass
class LLMOutput(Node):
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
class ExecAction(Node):
    type: ClassVar[str] = "exec_action"
    code: str = ""
    requested_transition: str | None = None
    #: Submission order within the action's worker session, used for shared replay.
    repl_execution_order: int | None = None
    #: Agent results explicitly read during this action, in first-read order.
    retrieved_agent_ids: list[str] = field(default_factory=list)

    def mark_agent_retrieved(self, agent_id: str) -> None:
        """Record one result access idempotently."""
        if agent_id not in self.retrieved_agent_ids:
            self.retrieved_agent_ids.append(agent_id)

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"] = {
            "code": self.code,
            "requested_transition": self.requested_transition,
            "repl_execution_order": self.repl_execution_order,
            "retrieved_agent_ids": list(self.retrieved_agent_ids),
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

    def append(self, node: Node) -> Node:
        child = super().append(node)
        if isinstance(node, AgentStart):
            self._refresh_code()
        return child

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

        return self.append(subtree)

    def _refresh_code(self) -> None:
        calls = [
            (f"launch_subagent('', model={child.config.model!r}, name={child.config.name!r})")
            for child in self.child_agents
        ]
        if len(calls) == 1:
            self.code = f"_handle = await {calls[0]}\nprint(await _handle.wait_for_result())"
            return
        self.code = (
            "_handles = [\n"
            + "".join(f"    await {call},\n" for call in calls)
            + "]\nprint([await h.wait_for_result() for h in _handles])"
        )


@dataclass
class ExecOutput(Node):
    type: ClassVar[str] = "exec_output"
    turn_mode: ClassVar[TurnMode] = TurnMode.ACTION

    def render(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.content}]

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"]["output"] = self.content
        return data


@dataclass
class ErrorOutput(Node):
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
class DoneOutput(Node):
    type: ClassVar[str] = "done_output"
    result: Any = None

    def to_record(self) -> dict[str, Any]:
        data = super().to_record()
        data["payload"].update({"result": self.result, "output": self.content})
        return data


def agent_payload(agent: AgentStart | None) -> dict[str, Any]:
    """The agent fields the run format repeats on every query node."""
    config = agent.config if agent is not None else AgentConfig()
    return {
        "inputs": dict(config.inputs),
        "model": config.model,
        "prompt_profile": config.prompt_profile,
        "output_schema": config.output_schema,
        "reuse_repl": config.reuse_repl,
        "launch_call_id": config.launch_call_id,
        "max_depth": config.max_depth,
        "max_iters": config.max_iters,
        "child_max_iters": config.child_max_iters,
        "max_budget": config.max_budget,
        "keep_n_messages": config.keep_n_messages,
        "max_output_length": config.max_output_length,
        "max_query_chars": config.max_query_chars,
    }


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid child name {name!r}")


def requested_transition(node: Node) -> str | None:
    """The transition selected by the action that produced this node."""
    previous = node.prev
    return previous.requested_transition if isinstance(previous, ExecAction) else None


def _rebuild_index(root: AgentStart) -> RunIndex:
    """Recompute private run aggregates after isolated direct graph construction."""
    index = RunIndex()
    for node in root.walk():
        index.register(node)
    index.revision = max(index.node_count - 1, 0)
    root._index = index
    return index


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
    "CONTINUE_NUDGE",
    "FINAL_ANSWER_ACTION",
    "ORCHESTRATOR_ADDENDUM",
    "TRUNCATION_SUMMARY",
    "WORKING_ACTION",
    "AgentConfig",
    "AgentStart",
    "AppendChild",
    "ContinueQuery",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "FinalQuery",
    "InspectQuery",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "PlanQuery",
    "ReplDead",
    "RunStats",
    "TruncationSummary",
    "TurnMode",
    "UserQuery",
    "can_spawn",
    "new_agent_id",
    "working_instruction",
    "new_node_id",
    "requested_transition",
    "start",
    "validate_agent_name",
]
