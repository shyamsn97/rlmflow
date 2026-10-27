"""Read-only snapshots of the agents in one graph."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rlmflow.graph.nodes import Node, agent_ref_id
from rlmflow.harnesses.repl.runtime.repl import AgentReference

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph

AgentStatus = Literal["blocked", "running", "waiting", "completed"]


def _json_safe(value: object) -> object:
    """Return a detached, JSON-compatible result value."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True, slots=True)
class AgentFrontier:
    """Small, immutable description of an agent's current frontier."""

    node_id: str
    type: str
    seq: int


@dataclass(frozen=True, slots=True)
class AgentHandle(AgentReference):
    """Future-like child selector whose observations refresh on every REPL action."""

    agent_id: str
    name: str
    path: str

    @property
    def id(self) -> str:
        return self.agent_id

    def _current(self) -> AgentInfo | None:
        from rlmflow.harnesses.repl.runtime.repl import current_binding

        agents = current_binding().get("agents")
        if not isinstance(agents, AgentDirectory):
            return None
        return agents.get(self.agent_id)

    @property
    def status(self) -> AgentStatus:
        current = self._current()
        return current.status if current is not None else "running"

    def done(self) -> bool:
        return self.status == "completed"

    def result(self) -> object | None:
        """Return the completed result or raise when it is not ready."""
        current = self._current()
        if current is None or not current.done():
            raise asyncio.InvalidStateError(f"agent {self.path!r} is not completed")
        return current.result()

    async def wait_for_result(self) -> object | None:
        """Wait for completion and return the result."""
        from rlmflow.harnesses.repl.runtime.repl import current_binding

        waiter = current_binding().get("wait_agent")
        if waiter is None:
            raise RuntimeError("agent waiting is unavailable in this execution")
        return await waiter(self.agent_id)

    def __await__(self):
        """Wait for this agent using Python's ordinary future syntax."""
        return self.wait_for_result().__await__()


@dataclass(frozen=True, slots=True)
class AgentInfo(AgentReference):
    """One agent as exposed through the REPL ``AGENTS`` snapshot."""

    agent_id: str
    name: str
    config: str
    path: str
    depth: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    status: AgentStatus
    frontier: AgentFrontier
    dependency_paths: tuple[str, ...]
    dependent_paths: tuple[str, ...]
    blocked_on_paths: tuple[str, ...]
    _result: object | None = field(default=None, repr=False)

    def done(self) -> bool:
        return self.status == "completed"

    def result(self) -> object | None:
        """Return the completed result or raise when it is not ready."""
        if not self.done():
            raise asyncio.InvalidStateError(f"agent {self.path!r} is not completed")
        return self._result

    async def wait_for_result(self) -> object | None:
        """Wait for completion and return the result."""
        from rlmflow.harnesses.repl.runtime.repl import current_binding

        waiter = current_binding().get("wait_agent")
        if waiter is None:
            raise RuntimeError("agent waiting is unavailable in this execution")
        return await waiter(self.agent_id)

    def __await__(self):
        return self.wait_for_result().__await__()


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
    """Viewer-scoped, read-only queries over one snapshotted agent graph."""

    viewer_id: str
    root_id: str
    agents: tuple[AgentInfo, ...]
    links: tuple[AgentEdge, ...] = ()
    _by_id: dict[str, AgentInfo] = field(init=False, repr=False, compare=False)
    _by_path: dict[str, AgentInfo] = field(init=False, repr=False, compare=False)
    _by_name: dict[str, tuple[AgentInfo, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_id = {agent.agent_id: agent for agent in self.agents}
        by_path = {agent.path: agent for agent in self.agents}
        names: dict[str, list[AgentInfo]] = {}
        for agent in self.agents:
            names.setdefault(agent.name, []).append(agent)
        if self.viewer_id not in by_id:
            raise ValueError(f"viewer {self.viewer_id!r} is not in the agent tree")
        if self.root_id not in by_id:
            raise ValueError(f"root {self.root_id!r} is not in the agent tree")
        object.__setattr__(self, "_by_id", by_id)
        object.__setattr__(self, "_by_path", by_path)
        object.__setattr__(
            self,
            "_by_name",
            {name: tuple(matches) for name, matches in names.items()},
        )

    @property
    def self(self) -> AgentInfo:
        """The agent whose REPL received this directory."""
        return self._by_id[self.viewer_id]

    def get(self, selector: str | AgentHandle | AgentInfo | None = None) -> AgentInfo | None:
        """Resolve self, an id/path, a direct relative, or a unique tree-wide name."""
        if selector is None:
            return self.self
        if isinstance(selector, AgentHandle):
            return self._by_id.get(selector.agent_id)
        if isinstance(selector, AgentInfo):
            return self._by_id.get(selector.agent_id)
        if not isinstance(selector, str):
            raise TypeError("agent selector must be a string, AgentHandle, AgentInfo, or None")

        exact = self._by_id.get(selector) or self._by_path.get(selector)
        if exact is not None:
            return exact

        relatives = [
            agent
            for agent in [
                self.get_parent(),
                *self.get_siblings(),
                *self.get_children(),
            ]
            if agent is not None and agent.name == selector
        ]
        if len(relatives) == 1:
            return relatives[0]
        if len(relatives) > 1:
            raise ValueError(f"ambiguous relative agent name {selector!r}; use a path or id")

        matches = self._by_name.get(selector, ())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"ambiguous agent name {selector!r}; use a path or id")
        return None

    def get_parent(self, agent: str | AgentHandle | AgentInfo | None = None) -> AgentInfo | None:
        """Return an agent's direct parent, defaulting to the viewer."""
        selected = self._selected(agent)
        return self._by_id.get(selected.parent_id) if selected.parent_id is not None else None

    def get_children(self, agent: str | AgentHandle | AgentInfo | None = None) -> list[AgentInfo]:
        """Return an agent's direct children in creation order."""
        selected = self._selected(agent)
        return [self._by_id[child_id] for child_id in selected.child_ids]

    def get_siblings(self, agent: str | AgentHandle | AgentInfo | None = None) -> list[AgentInfo]:
        """Return an agent's siblings in creation order, excluding itself."""
        selected = self._selected(agent)
        if selected.parent_id is None:
            return []
        parent = self._by_id[selected.parent_id]
        return [
            self._by_id[child_id] for child_id in parent.child_ids if child_id != selected.agent_id
        ]

    def edges(
        self,
        *,
        source: str | AgentHandle | AgentInfo | None = None,
        target: str | AgentHandle | AgentInfo | None = None,
        kind: str | None = None,
    ) -> list[AgentEdge]:
        """Application edges between agents, filtered by endpoint and kind."""
        source_id = self._selected(source).agent_id if source is not None else None
        target_id = self._selected(target).agent_id if target is not None else None
        return [
            edge
            for edge in self.links
            if (source_id is None or edge.source == source_id)
            and (target_id is None or edge.target == target_id)
            and (kind is None or edge.kind == kind)
        ]

    def render_graph(
        self,
        *,
        view: Literal["ownership", "dependencies"] = "ownership",
        show_results: bool = False,
        max_result_chars: int = 160,
    ) -> str:
        """Render the ownership tree or consumer-rooted dependency projection."""
        if max_result_chars < 1:
            raise ValueError("max_result_chars must be >= 1")
        if view not in ("ownership", "dependencies"):
            raise ValueError("view must be 'ownership' or 'dependencies'")

        lines: list[str] = []

        def label(agent: AgentInfo, connector: str) -> str:
            label = f"{connector}{agent.name} ({agent.config}) [{agent.status}]"
            if agent.agent_id == self.viewer_id:
                label += " (you)"
            if show_results and agent.status == "completed":
                preview = self._result_preview(agent.result(), max_result_chars)
                if preview:
                    label += f" -> {preview}"
            return label

        def visit(agent: AgentInfo, prefix: str, connector: str) -> None:
            lines.append(prefix + label(agent, connector))

            children = [self._by_id[child_id] for child_id in agent.child_ids]
            child_prefix = prefix + ("    " if connector == "└── " else "│   ")
            if not connector:
                child_prefix = ""
            for index, child in enumerate(children):
                last = index == len(children) - 1
                visit(child, child_prefix, "└── " if last else "├── ")

        def visit_dependencies(
            agent: AgentInfo,
            prefix: str,
            connector: str,
            branch: frozenset[str],
        ) -> None:
            lines.append(prefix + label(agent, connector))
            if agent.path in branch:
                lines[-1] += " (cycle)"
                return
            sources = [self._by_path[path] for path in agent.dependency_paths]
            child_prefix = prefix + ("    " if connector == "└── " else "│   ")
            if not connector:
                child_prefix = ""
            nested = branch | {agent.path}
            for index, source in enumerate(sources):
                last = index == len(sources) - 1
                visit_dependencies(
                    source,
                    child_prefix,
                    "└── " if last else "├── ",
                    nested,
                )

        if view == "ownership":
            visit(self._by_id[self.root_id], "", "")
        else:
            sinks = [
                agent
                for agent in self.agents
                if agent.dependency_paths and not agent.dependent_paths
            ]
            for index, sink in enumerate(sinks):
                if index:
                    lines.append("")
                visit_dependencies(sink, "", "", frozenset())
            if not sinks:
                lines.append("(no dependency edges)")
        return "\n".join(lines)

    def print_graph(
        self,
        *,
        view: Literal["ownership", "dependencies"] = "ownership",
        show_results: bool = False,
        max_result_chars: int = 160,
    ) -> None:
        """Print :meth:`render_graph` for the REPL's stdout-only observation."""
        print(
            self.render_graph(
                view=view,
                show_results=show_results,
                max_result_chars=max_result_chars,
            )
        )

    def _selected(self, selector: str | AgentHandle | AgentInfo | None) -> AgentInfo:
        selected = self.get(selector)
        if selected is None:
            raise KeyError(f"unknown agent {selector!r}")
        return selected

    @staticmethod
    def _result_preview(result: object | None, limit: int) -> str:
        if result is None:
            return ""
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        text = " ".join(encoded.split())
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3] + "..."


def build_agent_directory(graph: Graph, viewer: Node) -> AgentDirectory:
    """Snapshot every agent in ``graph`` as seen by ``viewer``."""
    agents = graph.agents()
    paths = {agent.id: graph.path(agent) for agent in agents}
    dependents: dict[str, list[str]] = {}
    for agent in agents:
        for producer in graph.producers(agent):
            dependents.setdefault(producer.id, []).append(paths[agent.id])
    info: list[AgentInfo] = []

    for agent in agents:
        launcher = graph.launcher(agent)
        producers = graph.producers(agent)
        terminal = graph.terminal(agent)
        if terminal:
            status: AgentStatus = "completed"
        elif any(not graph.terminal(producer) for producer in producers):
            status = "blocked"
        elif graph.status(agent) == "waiting":
            status = "waiting"
        else:
            status = "running"
        frontier = graph.frontier(agent)
        info.append(
            AgentInfo(
                agent_id=agent.id,
                name=agent.name,
                config=agent.config.name,
                path=paths[agent.id],
                depth=graph.depth(agent),
                parent_id=launcher.id if launcher is not None else None,
                child_ids=tuple(child.id for child in graph.children(agent)),
                status=status,
                frontier=AgentFrontier(node_id=frontier.id, type=frontier.type, seq=frontier.seq),
                dependency_paths=tuple(paths[producer.id] for producer in producers),
                dependent_paths=tuple(dependents.get(agent.id, ())),
                blocked_on_paths=tuple(
                    paths[producer.id]
                    for producer in producers
                    if not graph.terminal(producer)
                ),
                _result=_json_safe(graph.result(agent)) if terminal else None,
            )
        )

    ids = {agent.id for agent in agents}
    links = tuple(
        AgentEdge(
            id=edge.id,
            kind=edge.kind,
            source=edge.source,
            target=edge.target,
            data=_json_safe(dict(edge.data)),
        )
        for edge in graph.edges()
        if not edge.structural and edge.source in ids and edge.target in ids
    )
    return AgentDirectory(
        viewer_id=viewer.id,
        root_id=graph.lineage(viewer)[0].id,
        agents=tuple(info),
        links=links,
    )


def resolve_agent(g: Graph, selector: object) -> Node:
    """An agent by handle, directory entry, id, path, or unique name."""
    if isinstance(selector, (AgentHandle, AgentInfo)):
        selector = selector.agent_id
    selector = agent_ref_id(selector) or selector
    if not isinstance(selector, str):
        raise TypeError("agent selector must be a handle, an AGENTS entry, or a string")
    if selector in g and (found := g.node(selector)).type == "agent_start":
        return found
    agents = g.agents()
    by_path = [agent for agent in agents if g.path(agent) == selector]
    matches = by_path or [agent for agent in agents if agent.name == selector]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"ambiguous agent {selector!r}; use a path or id")
    raise KeyError(f"unknown agent {selector!r}")


def agent_handle(g: Graph, agent: Node) -> AgentHandle:
    return AgentHandle(agent_id=agent.id, name=agent.name, path=g.path(agent))


__all__ = [
    "agent_handle",
    "resolve_agent",
    "AgentDirectory",
    "AgentEdge",
    "AgentFrontier",
    "AgentHandle",
    "AgentInfo",
    "AgentStatus",
    "build_agent_directory",
]
