"""Agents as the REPL sees them: ``AgentHandle`` objects and the ``AGENTS`` snapshot."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rlmflow.runtime.repl import AGENT_REF_KEY, AgentReference

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph
    from rlmflow.graph.nodes import Node

AgentStatus = Literal["done", "idle", "running"]


def _json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _snapshot() -> AgentDirectory | None:
    """The ``AGENTS`` snapshot of the running node, if any."""
    from rlmflow.runtime.repl import current_binding

    try:
        agents = current_binding().get("agents")
    except RuntimeError:
        return None
    return agents if isinstance(agents, AgentDirectory) else None


def _binding(name: str, what: str):
    from rlmflow.runtime.repl import current_binding

    fn = current_binding().get(name)
    if fn is None:
        raise RuntimeError(f"{what} is unavailable in this execution")
    return fn


@dataclass(frozen=True, slots=True)
class AgentHandle(AgentReference):
    """One agent in the REPL: what ``launch_agent`` returns and ``AGENTS[...]`` looks up.

    ``await agent`` waits for the result of an agent you launched, and
    ``agent.send(message)`` messages any agent. Everything else (``status``,
    ``result()``, ``children``, ...) reads the ``AGENTS`` snapshot taken when
    the code started, so it doesn't change while it runs; only ``await`` waits.
    """

    agent_id: str
    name: str
    path: str
    _directory: AgentDirectory | None = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return f"Agent({self.path!r}, status={self.status!r})"

    @property
    def id(self) -> str:
        return self.agent_id

    def _record(self) -> AgentRecord | None:
        directory = _snapshot() or self._directory
        return directory._by_id.get(self.agent_id) if directory else None

    def _agents(self, ids: tuple[str, ...]) -> list[AgentHandle]:
        directory = _snapshot() or self._directory
        if directory is None:
            return []
        return [directory._handle(directory._by_id[i]) for i in ids if i in directory._by_id]

    @property
    def status(self) -> AgentStatus:
        record = self._record()
        return record.status if record is not None else "running"

    @property
    def config(self) -> str | None:
        record = self._record()
        return record.config if record is not None else None

    @property
    def task(self) -> str | None:
        record = self._record()
        return record.task if record is not None else None

    @property
    def parent(self) -> AgentHandle | None:
        record = self._record()
        found = self._agents((record.parent_id,) if record and record.parent_id else ())
        return found[0] if found else None

    @property
    def children(self) -> list[AgentHandle]:
        record = self._record()
        return self._agents(record.child_ids if record else ())

    def done(self) -> bool:
        return self.status == "done"

    def result(self) -> object | None:
        """The result, or raise if the agent hadn't finished when the code started."""
        record = self._record()
        if record is None or record.status != "done":
            raise asyncio.InvalidStateError(f"agent {self.path!r} is not done")
        return record.result

    def __await__(self):
        """Wait for the result of an agent you launched: ``await agent``."""
        return _binding("wait_agent", "awaiting agents")(self.agent_id).__await__()

    def send(self, message: str, **inputs: object) -> AgentHandle:
        """Message this agent, reopening it if it finished, and return it.

        Keyword arguments become variables in its REPL. ``await agent.send(...)``
        also waits for its next result, for agents you launched.
        """
        return _binding("send_agent", "messaging agents")(self.agent_id, message, **inputs)


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """What the ``AGENTS`` snapshot knows about one agent."""

    agent_id: str
    name: str
    config: str
    path: str
    task: str
    parent_id: str | None
    child_ids: tuple[str, ...]
    status: AgentStatus
    result: object | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class AgentEdge:
    """One application edge between two agents, such as ``peer``."""

    id: str
    kind: str
    source: str
    target: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentDirectory:
    """``AGENTS``: every agent in the run, as of when the code started."""

    viewer_id: str
    records: tuple[AgentRecord, ...]
    links: tuple[AgentEdge, ...] = ()
    _by_id: dict[str, AgentRecord] = field(init=False, repr=False, compare=False)
    _by_path: dict[str, AgentRecord] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_id", {r.agent_id: r for r in self.records})
        object.__setattr__(self, "_by_path", {r.path: r for r in self.records})
        if self.viewer_id not in self._by_id:
            raise ValueError(f"viewer {self.viewer_id!r} is not in the run")

    def _handle(self, record: AgentRecord) -> AgentHandle:
        return AgentHandle(record.agent_id, record.name, record.path, self)

    def __getitem__(self, selector: str | AgentReference) -> AgentHandle:
        record = self._find(selector)
        if record is None:
            raise KeyError(f"unknown agent {selector!r}")
        return self._handle(record)

    def __contains__(self, selector: object) -> bool:
        try:
            return self._find(selector) is not None  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[AgentHandle]:
        return (self._handle(record) for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def self(self) -> AgentHandle:
        """The agent running this code."""
        return self._handle(self._by_id[self.viewer_id])

    @property
    def parent(self) -> AgentHandle | None:
        return self.self.parent

    @property
    def children(self) -> list[AgentHandle]:
        """The agents you launched, in creation order."""
        return self.self.children

    def _find(self, selector: str | AgentReference) -> AgentRecord | None:
        """An id or path, a relative's name, or a name that is unique in the run."""
        if isinstance(selector, AgentReference):
            return self._by_id.get(selector.agent_id)
        if not isinstance(selector, str):
            raise TypeError("agent selector must be a string or an agent")
        exact = self._by_id.get(selector) or self._by_path.get(selector)
        if exact is not None:
            return exact
        me = self._by_id[self.viewer_id]
        near = {me.parent_id, *me.child_ids}
        if me.parent_id in self._by_id:
            near.update(self._by_id[me.parent_id].child_ids)
        relatives = [r for r in self.records if r.agent_id in near and r.name == selector]
        matches = relatives or [r for r in self.records if r.name == selector]
        if len(matches) > 1:
            raise ValueError(f"ambiguous agent name {selector!r}; use a path or id")
        return matches[0] if matches else None

    def edges(self, *, kind: str | None = None) -> list[AgentEdge]:
        """Application edges between agents, such as ``peer``."""
        return [edge for edge in self.links if kind is None or edge.kind == kind]

    def render_graph(self, *, show_results: bool = False, max_result_chars: int = 160) -> str:
        """The agent tree with each agent's config and status."""
        lines: list[str] = []

        def visit(record: AgentRecord, prefix: str, connector: str, child_prefix: str) -> None:
            text = f"{prefix}{connector}{record.name} ({record.config}) [{record.status}]"
            if record.agent_id == self.viewer_id:
                text += " (you)"
            if show_results and record.status == "done" and record.result is not None:
                preview = " ".join(json.dumps(record.result, ensure_ascii=False).split())
                if len(preview) > max_result_chars:
                    preview = preview[: max(max_result_chars - 3, 0)] + "..."
                text += f" -> {preview}"
            lines.append(text)
            children = [self._by_id[i] for i in record.child_ids if i in self._by_id]
            for index, child in enumerate(children):
                last = index == len(children) - 1
                visit(
                    child,
                    child_prefix,
                    "└── " if last else "├── ",
                    child_prefix + ("    " if last else "│   "),
                )

        for record in self.records:
            if record.parent_id is None:
                visit(record, "", "", "")
        return "\n".join(lines)

    def print_graph(self, *, show_results: bool = False, max_result_chars: int = 160) -> None:
        print(self.render_graph(show_results=show_results, max_result_chars=max_result_chars))


def build_agent_directory(graph: Graph, viewer: Node) -> AgentDirectory:
    """Snapshot every agent in ``graph`` as seen by ``viewer``."""
    records = []
    for agent in graph.agents():
        parent = graph.parent(agent)
        status = graph.status(agent)
        records.append(
            AgentRecord(
                agent_id=agent.id,
                name=agent.root_data.name,
                config=agent.root_data.config_name or "root",
                path=graph.path(agent),
                task=agent.root_data.task,
                parent_id=parent.id if parent is not None else None,
                child_ids=tuple(child.id for child in graph.children(agent)),
                status=status,  # type: ignore[arg-type]
                result=_json_safe(graph.result(agent)) if status == "done" else None,
            )
        )
    links = tuple(
        AgentEdge(edge.id, edge.kind, edge.source, edge.target, _json_safe(dict(edge.data)))  # type: ignore[arg-type]
        for edge in graph.edges()
        if not edge.structural and edge.source in graph._roots and edge.target in graph._roots
    )
    return AgentDirectory(viewer_id=viewer.id, records=tuple(records), links=links)


def resolve_agent(g: Graph, selector: object) -> Node:
    """An agent by root node, handle, id, path, or unique name."""
    from rlmflow.graph.nodes import Node

    if isinstance(selector, Node):
        return g.agent(selector.id)
    if isinstance(selector, AgentReference):
        selector = selector.agent_id
    if isinstance(selector, dict) and isinstance(selector.get(AGENT_REF_KEY), str):
        selector = selector[AGENT_REF_KEY]
    if not isinstance(selector, str):
        raise TypeError("agent selector must be an agent or a string")
    if selector in g._roots:
        return g._roots[selector]
    agents = g.agents()
    matches = [a for a in agents if g.path(a) == selector] or [
        a for a in agents if a.root_data.name == selector
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"ambiguous agent {selector!r}; use a path or id")
    raise KeyError(f"unknown agent {selector!r}")


def agent_handle(g: Graph, agent: Node) -> AgentHandle:
    return AgentHandle(agent_id=agent.id, name=agent.root_data.name, path=g.path(agent))


__all__ = [
    "AgentDirectory",
    "AgentEdge",
    "AgentHandle",
    "AgentRecord",
    "AgentStatus",
    "agent_handle",
    "build_agent_directory",
    "resolve_agent",
]
