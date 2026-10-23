"""The graph store: edges, the ordered log, lookups, application-edge writes, and the invariant check."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from rlmflow.graph.nodes import AgentStart, Node, UserQuery

if TYPE_CHECKING:
    from rlmflow.graph.config import AgentConfig

#: History order. A branch in history is a fork.
NEXT = "next"
#: Message to sender. On an ``AgentStart`` that is who launched the agent.
FROM = "from"
#: Consumer ``AgentStart`` to producer ``AgentStart``; ``data["as"]`` binds the result.
DEPENDS_ON = "depends_on"
STRUCTURAL = frozenset({NEXT, FROM, DEPENDS_ON})
#: Application edge ``agent -> recipient``: the agent's results go to the recipient.
#: Launching implies it; ``connect_agent(target, "reports_to")`` adds it for peers.
REPORTS_TO = "reports_to"


def new_edge_id() -> str:
    return f"edge_{uuid4().hex}"


def new_graph_id() -> str:
    return f"graph_{uuid4().hex}"


@dataclass(frozen=True, kw_only=True)
class Edge:
    """One directed relation between two nodes.

    ``next``, ``from``, and ``depends_on`` are structural. Any other ``kind`` is
    application data that the graph stores but does not interpret.
    """

    kind: str
    source: str
    target: str
    data: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_edge_id)
    seq: int = -1
    removed_seq: int | None = None

    @property
    def active(self) -> bool:
        return self.removed_seq is None

    @property
    def structural(self) -> bool:
        return self.kind in STRUCTURAL


N = TypeVar("N", bound=Node)
Item = Node | Edge


class InvariantError(ValueError):
    """A write would break one of the graph's structural invariants."""


class StaleFrontier(InvariantError):
    """A write was based on a frontier that has since moved."""


class Graph:
    """Every durable fact of a run, as nodes and edges in one ordered log.

    History (``next``, ``from``, ``depends_on``) is written only by the functions
    in :mod:`rlmflow.graph.ops`, which check their own preconditions.
    Application edges are written with ``add_edge`` and ``remove_edge``;
    ``merge`` copies in another graph. Nothing is ever deleted: removing an edge
    logs its updated record.
    """

    def __init__(self, id: str | None = None) -> None:
        self.id = id or new_graph_id()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}
        self._log: list[Item] = []
        self._lock = threading.RLock()
        # Private indexes: each agent's last own node, and undelivered messages.
        self._tail: dict[str, str] = {}
        self._inbox: dict[str, list[str]] = {}
        self._by_agent: dict[str, list[str]] = {}
        self._by_type: dict[type[Node], list[str]] = {}

    def __repr__(self) -> str:
        return f"Graph({self.id!r}, nodes={len(self._nodes)}, edges={len(self._edges)})"

    # -- writes -------------------------------------------------------------

    def add_edge(self, edge: Edge) -> Edge:
        """Record one application edge between two nodes already in the graph."""
        with self._lock:
            if edge.structural:
                raise InvariantError(
                    f"{edge.kind!r} edges are written by rlmflow.graph.ops, not add_edge"
                )
            if not edge.active:
                raise InvariantError(f"edge {edge.id!r} is removed; add its active record")
            for endpoint in (edge.source, edge.target):
                if endpoint not in self._nodes:
                    raise InvariantError(f"edge endpoint {endpoint!r} is not in the graph")
            (added,) = self._append_unchecked(edge)
            return added

    def remove_edge(self, edge: Edge | str) -> Edge:
        """Deactivate one application edge and return its updated record."""
        edge_id = edge if isinstance(edge, str) else edge.id
        with self._lock:
            current = self._edges.get(edge_id)
            if current is None:
                raise KeyError(f"unknown edge {edge_id!r}")
            if current.structural:
                raise InvariantError(f"structural {current.kind!r} edges cannot be removed")
            if not current.active:
                raise InvariantError(f"edge {edge_id!r} is already removed")
            (removed,) = self._append_unchecked(current, removing=True)
            return removed

    def merge(self, other: Graph) -> None:
        """Append ``other``'s log to this graph, in order.

        The node objects move: they belong to this graph afterwards and take its
        ``seq`` numbers. Ids must not collide; ``other`` is already valid, so
        nothing else is checked.
        """
        with self._lock:
            shared = [i for i in (*other._nodes, *other._edges) if i in self]
            if shared:
                raise InvariantError(f"graphs share ids: {', '.join(map(repr, shared[:3]))}")
            for item in other.items():
                removing = isinstance(item, Edge) and not item.active
                self._append_unchecked(
                    self._edges[item.id] if removing else item, removing=removing
                )

    def as_of(self, seq: int) -> Graph:
        """A copy of the graph as it stood once log item ``seq`` was written."""
        past = Graph(self.id)
        with self._lock:
            for item in self._log[: seq + 1]:
                if isinstance(item, Node):
                    past._append_unchecked(copy(item))
                elif item.active:
                    past._append_unchecked(item)
                else:
                    past._append_unchecked(past._edges[item.id], removing=True)
        return past

    # -- lookups ----------------------------------------------------------

    def node(self, id: str) -> Node:
        try:
            return self._nodes[id]
        except KeyError:
            raise KeyError(f"unknown node {id!r}") from None

    def nodes(self, type: type[N] = Node, *, agent: str | AgentStart | None = None) -> list[N]:
        """Nodes of ``type`` in ``seq`` order, optionally owned by one agent."""
        agent_id = agent.id if isinstance(agent, AgentStart) else agent
        if agent_id is not None:
            owned = (self._nodes[node_id] for node_id in self._by_agent.get(agent_id, ()))
            return [node for node in owned if isinstance(node, type)]
        groups = [ids for cls, ids in self._by_type.items() if issubclass(cls, type)]
        found = [self._nodes[node_id] for ids in groups for node_id in ids]
        if len(groups) > 1:
            found.sort(key=lambda node: node.seq)
        return found

    def edges(
        self,
        *,
        kind: str | None = None,
        source: str | Node | None = None,
        target: str | Node | None = None,
        active: bool | None = True,
    ) -> list[Edge]:
        """Edges matching every given filter, in ``seq`` order."""
        source_id = source.id if isinstance(source, Node) else source
        target_id = target.id if isinstance(target, Node) else target
        if source_id is not None:
            ids: Iterable[str] = self._out.get(source_id, ())
        elif target_id is not None:
            ids = self._in.get(target_id, ())
        else:
            ids = self._edges
        found = []
        for edge_id in ids:
            edge = self._edges[edge_id]
            if kind is not None and edge.kind != kind:
                continue
            if target_id is not None and edge.target != target_id:
                continue
            if active is not None and edge.active is not active:
                continue
            found.append(edge)
        return found

    def items(self, after: int = -1) -> list[Item]:
        """The log after ``seq``: the graph's one change stream."""
        with self._lock:
            return self._log[after + 1 :]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, Node):
            return self._nodes.get(item.id) is item
        if isinstance(item, Edge):
            return item.id in self._edges
        if isinstance(item, str):
            return item in self._nodes or item in self._edges
        return False

    def __len__(self) -> int:
        return len(self._log)

    def create_agent(
        self,
        task: str = "",
        *,
        config: AgentConfig | None = None,
        name: str | None = None,
        depends_on: Iterable[AgentStart] = (),
        inputs: Mapping[str, Any] | None = None,
        **args: Any,
    ) -> AgentStart:
        """Add a top-level agent; keyword arguments become its REPL variables.

        Nothing runs here. A ``Flow`` driving this graph creates the agent's
        REPL the first time the agent steps.
        """
        from rlmflow.graph import ops

        return ops.spawn(
            self,
            task,
            config=config,
            name=name,
            depends_on=depends_on,
            inputs=inputs,
            **args,
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the log and its argument blobs as a run directory."""
        from rlmflow.graph import persistence

        return persistence.save_graph(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """Read a run directory and check the whole graph's invariants."""
        from rlmflow.graph import persistence

        return persistence.load_graph(path)

    # -- private ------------------------------------------------------------

    def _last(self, agent_id: str) -> Node:
        """The agent's last own node: its ``AgentStart`` until it appends."""
        return self._nodes[self._tail[agent_id]]

    def _append_unchecked(self, *items: Item, removing: bool = False) -> list[Item]:
        """Index and log ``items`` in order, checking only that ids are new.

        Structural rules are the caller's job: ``ops`` checks them per write,
        and ``Graph.load`` runs ``check_invariants`` over the whole result.
        With ``removing``, each item is an edge already in the graph.
        """
        logged: list[Item] = []
        for item in items:
            seq = len(self._log)
            if removing:
                item = replace(self._edges[item.id], removed_seq=seq)
                self._edges[item.id] = item
            elif isinstance(item, Node):
                if item.id in self._nodes:
                    raise InvariantError(f"duplicate node id {item.id!r}")
                item.seq = seq
                if isinstance(item, AgentStart):
                    item.agent_id = item.agent_id or item.id
                    self._tail[item.id] = item.id
                elif isinstance(item, UserQuery):
                    self._inbox.setdefault(item.agent_id, []).append(item.id)
                self._nodes[item.id] = item
                self._out.setdefault(item.id, [])
                self._in.setdefault(item.id, [])
                self._by_agent.setdefault(item.agent_id, []).append(item.id)
                self._by_type.setdefault(type(item), []).append(item.id)
            else:
                if item.id in self._edges:
                    raise InvariantError(f"duplicate edge id {item.id!r}")
                item = replace(item, seq=seq)
                self._edges[item.id] = item
                self._out.setdefault(item.source, []).append(item.id)
                self._in.setdefault(item.target, []).append(item.id)
                target = self._nodes.get(item.target)
                if item.kind == NEXT and target is not None and not isinstance(target, AgentStart):
                    self._tail[target.agent_id] = target.id
                    queued = self._inbox.get(target.agent_id)
                    if queued and target.id in queued:
                        queued.remove(target.id)
            self._log.append(item)
            logged.append(item)
        return logged


def check_invariants(g: Graph) -> None:
    """Walk the whole graph; raise ``InvariantError`` at the first broken rule.

    The writes in ``ops`` keep these true one at a time. This walk is for a
    log read from disk: every node owned by an agent, one predecessor per node,
    each agent's history one chain from its ``AgentStart``, forks only at landed
    nodes, an acyclic launcher tree, and acyclic ``depends_on`` declared before
    the consumer starts.
    """
    nodes = g._nodes
    for node in nodes.values():
        if isinstance(node, AgentStart):
            if node.agent_id != node.id:
                raise InvariantError(f"agent {node.id!r} must own itself")
        elif not isinstance(nodes.get(node.agent_id), AgentStart):
            raise InvariantError(f"{node.type} {node.id!r} names unknown agent {node.agent_id!r}")
    predecessor: dict[str, Edge] = {}
    successor: dict[str, str] = {}
    forks: list[Edge] = []
    launcher: dict[str, str] = {}
    depends: dict[str, list[Edge]] = {}
    for edge in g._edges.values():
        for endpoint in (edge.source, edge.target):
            if endpoint not in nodes:
                raise InvariantError(f"edge endpoint {endpoint!r} is not in the graph")
        source, target = nodes[edge.source], nodes[edge.target]
        if edge.kind == NEXT:
            if target.id in predecessor:
                raise InvariantError(f"{target.type} {target.id!r} already has a predecessor")
            predecessor[target.id] = edge
            if isinstance(target, AgentStart):
                forks.append(edge)
                continue
            if source.agent_id != target.agent_id:
                raise InvariantError(
                    f"{target.type} {target.id!r} follows {source.id!r} of another agent"
                )
            if source.id in successor:
                raise InvariantError(f"{source.type} {source.id!r} has two successors")
            successor[source.id] = target.id
        elif edge.kind == FROM:
            if isinstance(source, AgentStart):
                if source.id in launcher:
                    raise InvariantError(f"agent {source.id!r} already has a launcher")
                launcher[source.id] = target.agent_id
            elif not isinstance(source, UserQuery):
                raise InvariantError(
                    f"a from edge leaves an AgentStart or UserQuery, not {source.type}"
                )
        elif edge.kind == DEPENDS_ON:
            if not isinstance(source, AgentStart) or not isinstance(target, AgentStart):
                raise InvariantError("depends_on edges connect AgentStart nodes")
            if source.id == target.id:
                raise InvariantError(f"agent {source.id!r} cannot depend on itself")
            depends.setdefault(source.id, []).append(edge)
    for edge in forks:
        source = nodes[edge.source]
        if not isinstance(source, AgentStart) and source.id not in predecessor:
            raise InvariantError(f"cannot fork at undelivered {source.type} {source.id!r}")
    for agent in g.nodes(AgentStart):
        reached, current = {agent.id}, agent.id
        while current in successor:
            current = successor[current]
            reached.add(current)
        for node_id in g._by_agent.get(agent.id, ()):
            node = nodes[node_id]
            if node_id in reached or isinstance(node, UserQuery) and node_id not in predecessor:
                continue
            if node_id not in predecessor:
                raise InvariantError(f"{node.type} {node_id!r} has no incoming next edge")
            raise InvariantError(f"{node.type} {node_id!r} is not in its agent's history")
        owner, seen = agent.id, set()
        while owner in launcher:
            seen.add(owner)
            owner = launcher[owner]
            if owner in seen:
                raise InvariantError(f"agent {owner!r} cannot launch itself")
        first = successor.get(agent.id)
        for edge in depends.get(agent.id, ()):
            if first is not None and edge.seq > predecessor[first].seq:
                raise InvariantError(f"agent {agent.id!r} has already started")
    _check_depends_acyclic(depends)


def _check_depends_acyclic(depends: Mapping[str, list[Edge]]) -> None:
    done: set[str] = set()
    for root in depends:
        path: set[str] = set()
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            agent_id, leaving = stack.pop()
            if leaving:
                path.discard(agent_id)
                done.add(agent_id)
                continue
            if agent_id in done:
                continue
            if agent_id in path:
                raise InvariantError(f"depends_on cycle through {agent_id!r}")
            path.add(agent_id)
            stack.append((agent_id, True))
            stack.extend((edge.target, False) for edge in depends.get(agent_id, ()))


__all__ = [
    "DEPENDS_ON",
    "Edge",
    "FROM",
    "Graph",
    "InvariantError",
    "NEXT",
    "REPORTS_TO",
    "STRUCTURAL",
    "StaleFrontier",
    "check_invariants",
    "new_edge_id",
    "new_graph_id",
]
