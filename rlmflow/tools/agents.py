"""Read-only snapshots of one recursive agent tree."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal

from rlmflow.graph.nodes import AgentStart, ExecAction, Node

AgentStatus = Literal["running", "waiting", "idle", "completed"]
AGENTS_BINDING = "__rlmflow_agents__"
AGENT_WAIT_TOOL = "__rlmflow_wait_agent__"


def _json_safe(value: object) -> object:
    """Return a detached, JSON-compatible result value."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _supervisor(agent: AgentStart) -> AgentStart | None:
    """The agent whose action launched ``agent``, or ``None`` for a root."""
    return agent.parent.parent_agent if agent.parent is not None else None


@dataclass(frozen=True, slots=True)
class AgentFrontier:
    """Small, immutable description of an agent's current frontier."""

    node_id: str
    type: str
    seq: int


@dataclass(frozen=True, slots=True)
class AgentHandle:
    """Future-like child selector whose observations refresh on every REPL action."""

    agent_id: str
    name: str
    path: str

    @property
    def id(self) -> str:
        return self.agent_id

    def _current(self) -> AgentInfo | None:
        from rlmflow.runtime.repl import current_binding

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
        from rlmflow.runtime.repl import current_binding

        waiter = current_binding().get("wait_agent")
        if waiter is None:
            raise RuntimeError("agent waiting is unavailable in this execution")
        return await waiter(self.agent_id)


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """One agent as exposed through the REPL ``AGENTS`` snapshot."""

    agent_id: str
    name: str
    path: str
    depth: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    status: AgentStatus
    frontier: AgentFrontier
    _result: object | None = field(default=None, repr=False)

    def done(self) -> bool:
        return self.status == "completed"

    def result(self) -> object | None:
        """Return the completed result or raise when it is not ready."""
        if not self.done():
            raise asyncio.InvalidStateError(f"agent {self.path!r} is not completed")
        return self._result

    def get_result(self) -> object | None:
        """Compatibility alias for :meth:`result`."""
        return self.result()

    async def wait_for_result(self) -> object | None:
        """Wait for completion and return the result."""
        from rlmflow.runtime.repl import current_binding

        waiter = current_binding().get("wait_agent")
        if waiter is None:
            raise RuntimeError("agent waiting is unavailable in this execution")
        return await waiter(self.agent_id)


@dataclass(frozen=True, slots=True)
class AgentDirectory:
    """Viewer-scoped, read-only queries over one snapshotted agent tree."""

    viewer_id: str
    root_id: str
    agents: tuple[AgentInfo, ...]
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

    def all(self) -> list[AgentInfo]:
        """Every agent in root-first tree order."""
        return list(self.agents)

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

    def render_graph(
        self,
        *,
        show_results: bool = False,
        max_result_chars: int = 160,
    ) -> str:
        """Render the snapshotted tree as compact ASCII text."""
        if max_result_chars < 1:
            raise ValueError("max_result_chars must be >= 1")

        lines: list[str] = []

        def visit(agent: AgentInfo, prefix: str, connector: str) -> None:
            label = f"{connector}{agent.name} [{agent.status}]"
            if agent.agent_id == self.viewer_id:
                label += " (you)"
            if show_results and agent.status == "completed":
                preview = self._result_preview(agent.result(), max_result_chars)
                if preview:
                    label += f" -> {preview}"
            lines.append(prefix + label)

            children = [self._by_id[child_id] for child_id in agent.child_ids]
            child_prefix = prefix + ("    " if connector == "└── " else "│   ")
            if not connector:
                child_prefix = ""
            for index, child in enumerate(children):
                last = index == len(children) - 1
                visit(child, child_prefix, "└── " if last else "├── ")

        visit(self._by_id[self.root_id], "", "")
        return "\n".join(lines)

    def print_graph(
        self,
        *,
        show_results: bool = False,
        max_result_chars: int = 160,
    ) -> None:
        """Print :meth:`render_graph` for the REPL's stdout-only observation."""
        print(
            self.render_graph(
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


def build_agent_directory(
    viewer: AgentStart,
    *,
    running_nodes: tuple[Node, ...] = (),
) -> AgentDirectory:
    """Snapshot ``viewer``'s recursive root and current in-flight nodes."""
    root = viewer.root
    if not isinstance(root, AgentStart):
        raise RuntimeError("viewer is detached")

    running_by_agent = {
        node.parent_agent.id: node for node in running_nodes if node.parent_agent is not None
    }
    agents = [node for node in root.walk() if isinstance(node, AgentStart)]
    info: list[AgentInfo] = []

    for agent in agents:
        parent = _supervisor(agent)
        running = running_by_agent.get(agent.id)
        if agent.terminal:
            status: AgentStatus = "completed"
        elif running is None:
            status = "idle"
        elif isinstance(running, ExecAction) and any(
            isinstance(child, AgentStart) and not child.terminal for child in running.children
        ):
            status = "waiting"
        else:
            status = "running"

        info.append(
            AgentInfo(
                agent_id=agent.id,
                name=agent.config.name,
                path=agent.config.path,
                depth=agent.config.depth,
                parent_id=parent.id if parent is not None else None,
                child_ids=tuple(child.id for child in agent.sub_agents),
                status=status,
                frontier=AgentFrontier(
                    node_id=agent.frontier.id,
                    type=agent.frontier.type,
                    seq=agent.frontier.seq,
                ),
                _result=_json_safe(agent.result()) if agent.terminal else None,
            )
        )

    return AgentDirectory(
        viewer_id=viewer.id,
        root_id=root.id,
        agents=tuple(info),
    )


__all__ = [
    "AGENTS_BINDING",
    "AGENT_WAIT_TOOL",
    "AgentDirectory",
    "AgentFrontier",
    "AgentHandle",
    "AgentInfo",
    "AgentStatus",
    "build_agent_directory",
]
