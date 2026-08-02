"""In-memory agent transcripts used by the running engine."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from rlmflow.llm import LLMUsage

DEFAULT_QUERY = (
    "Please read through the context and answer any queries or respond to any "
    "instructions contained within it."
)
DEFAULT_MAX_QUERY_CHARS = 2_000


def new_agent_id() -> str:
    return f"agent_{uuid4().hex}"


def new_node_id() -> str:
    return f"node_{uuid4().hex}"


def _isoformat(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, UTC).isoformat()


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
    max_depth: int = 2
    max_iters: int = 20
    child_max_iters: int | None = None
    max_budget: int | None = None
    #: How many of the agent's own transcript turns a prompt carries. The system
    #: message and the truncation notice sit on top of this count.
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
            "max_iters": self.child_max_iters or self.max_iters,
        }
        return replace(self, **{**values, **overrides})


@dataclass
class Node:
    id: str = field(default_factory=new_node_id)
    type: ClassVar[str] = "node"
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
    #: When the step that produced this node ran. Nodes written from inside a step
    #: (a nudge, a final-answer prod) were not executed, so they have no timing.
    started_at: float | None = None
    finished_at: float | None = None

    def append(self, node: Node) -> Node:
        """Hang a node off this one; whatever appends becomes its agent's frontier."""
        agent = self.parent_agent
        if agent is None:
            raise RuntimeError("node is detached")

        if self is not agent.frontier:
            raise ValueError(f"{self.id} is not the frontier of {agent.id}")

        if isinstance(node, AgentStart) and any(
            child.config.name == node.config.name for child in agent.sub_agents
        ):
            raise ValueError(f"duplicate child name {node.config.name!r}")

        self.children.append(node)
        node.parent = self

        if isinstance(node, AgentStart):
            node.root = agent.root
            agent.sub_agents.append(node)
            return node

        node.parent_agent = agent
        node.root = agent.root
        node.seq = self.seq + 1  # a sub-agent keeps its own 0-based sequence
        agent.frontier = node
        return node

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

    def walk(self, *, reverse: bool = False) -> Iterator[Node]:
        """Every node from here down, or with ``reverse`` back to this agent's start.

        The two directions cover different ground, since only the forward one can
        branch: down is the whole subtree including sub-agents, back is a single
        chain through this node's own agent.
        """
        if reverse:
            node: Node | None = self
            while node is not None:
                yield node
                node = node.prev
            return
        yield self
        for child in self.children:
            yield from child.walk()

    def tokens(self) -> LLMUsage:
        """What every model call from here down cost, added up."""
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

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        """This node as run-format data; each type extends payload and metadata."""
        agent = self.parent_agent
        timing = self.timing()
        return {
            "id": self.id,
            "type": self.type,
            "agent_id": agent.config.path if agent is not None else None,
            "order": self.seq,
            "metadata": {"timing": timing} if timing else {},
            "payload": {"content": self.content},
            "children": [child.to_dict() for child in self.children] if nested else [],
        }

    def fork(self, new_ids: bool = True) -> AgentStart:
        """Copy the whole graph, then cut everything after this node."""
        if self.root is None or self.parent_agent is None:
            raise RuntimeError("node is detached")

        root = deepcopy(self.root)
        node = next(n for n in root.walk() if n.id == self.id)
        node.children = []

        agent = node.parent_agent
        agent.frontier = node
        kept = {id(n) for n in root.walk()}
        agent.sub_agents = [sub for sub in agent.sub_agents if id(sub) in kept]

        if new_ids:
            for n in root.walk():
                n.id = new_agent_id() if isinstance(n, AgentStart) else new_node_id()
        return root


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

    def __post_init__(self) -> None:
        self.root = self
        self.parent_agent = self
        self.frontier = self

    @property
    def terminal(self) -> bool:
        return isinstance(self.frontier, DoneOutput)

    def result(self) -> Any:
        node = self.frontier
        return node.result if isinstance(node, DoneOutput) else None

    def leaves(self) -> list[Node]:
        out: list[Node] = []
        for child in self.sub_agents:
            out.extend(child.leaves())
        out.append(self.frontier)
        return out

    def transcript(self) -> list[Node]:
        """This agent's own nodes, start to frontier; sub-agent branches are skipped."""
        return list(self.frontier.walk(reverse=True))[::-1]

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
        for node in self.frontier.walk(reverse=True):
            text = self.system_prompt_for(node)
            if text:
                return text
        return None

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
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

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
        data["payload"].update(agent_payload(self.parent_agent))
        return data


@dataclass
class LLMOutput(Node):
    type: ClassVar[str] = "llm_output"
    code: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)
    #: Key into the owning agent's ``system_prompts`` table.
    prompt_id: str = ""

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
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

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
        data["payload"] = {"code": self.code}
        return data


@dataclass
class ExecOutput(Node):
    type: ClassVar[str] = "exec_output"

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
        data["payload"]["output"] = self.content
        return data


@dataclass
class ErrorOutput(Node):
    type: ClassVar[str] = "error_output"
    error: str = "exec"

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
        data["payload"].update({"error": self.error, "output": self.content})
        return data


@dataclass
class DoneOutput(Node):
    type: ClassVar[str] = "done_output"
    result: Any = None

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
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
    }


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid child name {name!r}")


def start(
    query: str = "",
    *,
    inputs: dict[str, str] | None = None,
    model: str = "default",
    prompt_profile: str = "default",
    output_schema: dict[str, Any] | None = None,
    max_depth: int = 2,
    max_iters: int = 20,
    child_max_iters: int | None = None,
    max_budget: int | None = None,
    keep_n_messages: int | None = None,
    max_output_length: int = 4_000,
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
) -> AgentStart:
    return AgentStart(
        content=query or DEFAULT_QUERY,
        config=AgentConfig(
            inputs=dict(inputs or {}),
            model=model,
            prompt_profile=prompt_profile,
            output_schema=output_schema,
            max_depth=max_depth,
            max_iters=max_iters,
            child_max_iters=child_max_iters,
            max_budget=max_budget,
            keep_n_messages=keep_n_messages,
            max_output_length=max_output_length,
            max_query_chars=max_query_chars,
        ),
    )


__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "AgentConfig",
    "AgentStart",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "UserQuery",
    "new_agent_id",
    "new_node_id",
    "start",
    "validate_agent_name",
]
