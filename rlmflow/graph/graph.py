"""The graph: nodes and edges in one ordered log, the writes that keep its history valid, and the queries over it."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from rlmflow.graph.config import (
    DEFAULT_QUERY,
    input_parts,
    validate_agent_name,
    validate_argument_name,
)
from rlmflow.graph.nodes import AGENT_START, CODE, DONE, MESSAGE, WAITING, LLMUsage, Node

T = TypeVar("T")

#: History order. A branch in history is a fork.
NEXT = "next"
#: Message to sender. On an ``agent_start`` that is who launched the agent.
FROM = "from"
#: Consumer ``agent_start`` to producer ``agent_start``; ``data["as"]`` binds the result.
DEPENDS_ON = "depends_on"
STRUCTURAL = frozenset({NEXT, FROM, DEPENDS_ON})
#: Application edge ``agent -> recipient``: the agent's results go to the recipient.
#: Launching implies it; ``connect_agent`` adds it for peers.
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


Item = Node | Edge


class InvariantError(ValueError):
    """A write would break one of the graph's structural invariants."""


class StaleFrontier(InvariantError):
    """A write was based on a frontier that has since moved."""


#: Frontiers where a queued message may enter history without cutting a turn.
DELIVERABLE = frozenset({AGENT_START, MESSAGE, CODE, WAITING, DONE})


def locked(write: Callable[..., T]) -> Callable[..., T]:
    """Check and write under the graph's lock, so no other write lands between."""

    @wraps(write)
    def write_under_lock(g: Graph, *args: Any, **kwargs: Any) -> T:
        with g._lock:
            return write(g, *args, **kwargs)

    return write_under_lock


def agent_node(
    task: str,
    args: Mapping[str, Any] | None,
    *,
    config: str | None,
    name: str,
) -> Node:
    """An unlanded ``agent_start`` naming the Flow config it runs with (``None``: a root)."""
    if config is not None:
        validate_agent_name(config)
    if not isinstance(task, str):
        raise TypeError("agent task must be a string")
    validate_agent_name(name)
    values = dict(args or {})
    for arg in values:
        validate_argument_name(arg)
    data: dict[str, Any] = {"task": task or DEFAULT_QUERY, "name": name, "args": values}
    if config is not None:
        data["config"] = config
    return Node(AGENT_START, **data)


class Graph:
    """Every durable fact of a run, as nodes and edges in one ordered log.

    History (``next``, ``from``, ``depends_on``) is written only by ``spawn``,
    ``append``, ``send``, ``deliver``, ``fork`` and ``attach``, which check their
    own preconditions. The queries below them read agents' history, status and
    results. Application edges are written with ``add_edge`` and ``remove_edge``;
    ``merge`` copies in another graph. Nothing is ever deleted: removing an edge
    logs its updated record. Agents name their config; a ``Flow`` decides what
    each name means.
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
        self._by_type: dict[str, list[str]] = {}

    def __repr__(self) -> str:
        return f"Graph({self.id!r}, nodes={len(self._nodes)}, edges={len(self._edges)})"

    # -- history writes ---------------------------------------------------

    @locked
    def append(self, agent: Node, node: Node, *, after: Node | None = None) -> Node:
        """Land ``node`` as the agent's next node.

        ``after`` is a compare-and-set guard: the write raises ``StaleFrontier``
        unless ``after`` is still the agent's frontier.
        """
        self._require_agent(agent)
        if node.type in (AGENT_START, MESSAGE):
            raise InvariantError(f"a {node.type} node is added by spawn, fork or send, not append")
        if node.id in self._nodes:
            raise InvariantError(f"duplicate node id {node.id!r}")
        source = self._last(agent.id)
        if after is not None and after is not source:
            if not (source is agent and self.prev(agent) is after):
                raise StaleFrontier(f"{after.type} {after.id!r} is not the frontier of {agent.id!r}")
            source = agent  # an empty fork extends from its own agent_start
        node.agent_id = agent.id
        self._append_unchecked(node, Edge(kind=NEXT, source=source.id, target=node.id))
        return node

    @locked
    def spawn(
        self,
        task: str = "",
        *,
        by: Node | None = None,
        name: str | None = None,
        config: str | None = None,
        depends_on: Iterable[Node] = (),
        inputs: Mapping[str, Any] | None = None,
        **args: Any,
    ) -> Node:
        """Add an agent; ``by`` is the node that created it, ``None`` for top-level.

        ``config`` names one of the Flow's configs (``"default"`` when ``by`` is
        given); ``None`` without ``by`` makes a root, which runs with the run's
        ``root_config``. ``inputs`` binds producer results to names; a value may
        be an agent or ``Input(agent, description=...)``.
        """
        self._require(by)
        if by is not None and config is None:
            config = "default"
        start = agent_node(task, args, config=config, name=name or self._default_name(config, by))
        items: list[Node | Edge] = [start]
        if by is not None:
            items.append(Edge(kind=FROM, source=start.id, target=by.id))
        for producer in depends_on:
            self._require_agent(producer)
            items.append(Edge(kind=DEPENDS_ON, source=start.id, target=producer.id))
        for input_name, value in (inputs or {}).items():
            wrapped = input_parts(value)
            producer, description = wrapped if wrapped is not None else (value, None)
            self._require_agent(producer)
            data = {"as": input_name, **({"description": description} if description else {})}
            items.append(Edge(kind=DEPENDS_ON, source=start.id, target=producer.id, data=data))
        self._append_unchecked(*items)
        return start

    @locked
    def send(
        self,
        agent: Node,
        text: str,
        *,
        sender: Node | Sequence[Node] | str | None = None,
        **data: Any,
    ) -> Node:
        """Queue a durable ``message`` in the agent's inbox without touching its history.

        ``sender`` is the node(s) it comes from (each gets a ``from`` edge), or a
        label such as ``"user"`` or ``"rlmflow"``.
        """
        self._require_agent(agent)
        if not isinstance(text, str):
            raise TypeError("message text must be a string")
        if isinstance(sender, str) or sender is None:
            senders: list[Node] = []
            label = sender or "user"
        else:
            senders = [sender] if isinstance(sender, Node) else list(sender)
            self._require(*senders)
            owners = sorted({node.agent_id for node in senders})
            label = owners[0] if len(owners) == 1 else ",".join(owners)
        message = Node(MESSAGE, text=text, sender=label, **data)
        message.agent_id = agent.id
        self._append_unchecked(
            message, *(Edge(kind=FROM, source=message.id, target=item.id) for item in senders)
        )
        return message

    @locked
    def deliver(self, agent: Node) -> list[Node]:
        """Move queued messages into history, in ``seq`` order, at a turn boundary.

        A fork that owns nothing yet takes messages at any base node.
        """
        queued = self.inbox(agent)
        source = self._last(agent.id)
        if not queued or not (source is agent or self.frontier(agent).type in DELIVERABLE):
            return []
        edges = []
        for message in queued:
            edges.append(Edge(kind=NEXT, source=source.id, target=message.id))
            source = message
        self._append_unchecked(*edges)
        return queued

    @locked
    def fork(
        self,
        at: Node,
        *,
        name: str | None = None,
        by: Node | None = None,
        config: str | None = None,
    ) -> Node:
        """Branch a new agent whose history is everything up to ``at``.

        Nothing is copied: the fork shares the prefix through one ``next`` edge.
        It runs with the owner's config name unless ``config`` names another; a
        fork of a root is a root.
        """
        self._require(at, by)
        if at.type != AGENT_START and not self.edges(kind=NEXT, target=at.id):
            raise InvariantError(f"cannot fork at undelivered {at.type} {at.id!r}")
        owner = self.node(at.agent_id)
        if name is None:
            forks = sum(1 for agent in self.agents() if self.prev(agent) is not None)
            name = f"{owner.name}-fork{forks}"
        start = agent_node(
            owner.task,
            deepcopy(owner.args),
            config=config if config is not None else owner.config_name,
            name=name,
        )
        items: list[Node | Edge] = [start, Edge(kind=NEXT, source=at.id, target=start.id)]
        if by is not None:
            items.append(Edge(kind=FROM, source=start.id, target=by.id))
        self._append_unchecked(*items)
        return start

    @locked
    def attach(self, parent: Node, *agents: Node) -> Node:
        """Make ``agents`` children of ``parent`` and tell it with a ``message``.

        Attached agents may be finished or still running. Each must not already
        have a launcher.
        """
        if not agents:
            raise ValueError("attach needs at least one agent")
        self._require_agent(parent)
        ancestors = {agent.id for agent in self.lineage(parent)}
        for agent in agents:
            self._require_agent(agent)
            if agent.id in ancestors:
                raise InvariantError(f"agent {agent.id!r} cannot launch itself")
            if self.edges(kind=FROM, source=agent.id):
                raise ValueError(f"agent {agent.id!r} already has a launcher")
        listing = "\n".join(f"- {agent.name} ({agent.id}): {agent.task[:200]}" for agent in agents)
        message = self.send(parent, f"These agents now report to you:\n{listing}", sender="rlmflow")
        self._append_unchecked(*(Edge(kind=FROM, source=agent.id, target=message.id) for agent in agents))
        return message

    # -- other writes -----------------------------------------------------

    def add_edge(self, edge: Edge) -> Edge:
        """Record one application edge between two nodes already in the graph."""
        with self._lock:
            if edge.structural:
                raise InvariantError(
                    f"{edge.kind!r} edges are written by the graph's history writes, not add_edge"
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

    def nodes(self, type: str | None = None, *, agent: str | Node | None = None) -> list[Node]:
        """Nodes of ``type`` (all types when ``None``) in ``seq`` order, optionally one agent's."""
        agent_id = agent.id if isinstance(agent, Node) else agent
        if agent_id is not None:
            owned = (self._nodes[node_id] for node_id in self._by_agent.get(agent_id, ()))
            return [node for node in owned if type is None or node.type == type]
        if type is not None:
            return [self._nodes[node_id] for node_id in self._by_type.get(type, ())]
        return [item for item in self._log if isinstance(item, Node)]

    def agents(self) -> list[Node]:
        """Every ``agent_start`` node, in ``seq`` order."""
        return self.nodes(AGENT_START)

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

    # -- agent queries ----------------------------------------------------

    def prev(self, node: Node) -> Node | None:
        """The node before ``node`` in history, or ``None`` at a run's first node."""
        edges = self.edges(kind=NEXT, target=node.id)
        return self.node(edges[0].source) if edges else None

    def history(self, at: Node) -> list[Node]:
        """Nodes leading to ``at``; for an agent, its whole history to the frontier.

        A fork's walk continues through its base agent's prefix.
        """
        node: Node | None = self._last(at.id) if at.type == AGENT_START else at
        nodes: list[Node] = []
        while node is not None:
            nodes.append(node)
            node = self.prev(node)
        nodes.reverse()
        return nodes

    def own_history(self, agent: Node) -> list[Node]:
        """The agent's history from its own ``agent_start``, without a fork's base prefix."""
        nodes = self.history(agent)
        return nodes[next(i for i, node in enumerate(nodes) if node is agent) :]

    def frontier(self, agent: Node) -> Node:
        """The agent's latest node. An empty fork stands at its base node."""
        last = self._last(agent.id)
        if last is agent:
            return self.prev(agent) or agent
        return last

    def terminal(self, agent: Node) -> bool:
        """Finished, with no queued message that would reopen it."""
        return self.frontier(agent).type == DONE and not self._inbox.get(agent.id)

    def status(self, agent: Node) -> str:
        """``"done"``, ``"waiting"`` or ``"running"``."""
        if self.terminal(agent):
            return "done"
        if self.frontier(agent).type == WAITING and not self._inbox.get(agent.id):
            return "waiting"
        return "running"

    def result(self, agent: Node) -> Any:
        """The agent's result, or ``None`` until it is done."""
        node = self.frontier(agent)
        return node.data.get("result") if node.type == DONE else None

    def inbox(self, agent: Node) -> list[Node]:
        """Messages sent to ``agent`` that are not yet part of its history."""
        return [self.node(node_id) for node_id in self._inbox.get(agent.id, ())]

    def producers(self, agent: Node) -> list[Node]:
        """Agents this agent must wait for, through active ``depends_on`` edges."""
        return [self.node(edge.target) for edge in self.edges(kind=DEPENDS_ON, source=agent.id)]

    def inputs(self, agent: Node) -> dict[str, Any]:
        """Producer results bound to names by ``depends_on`` edges with ``as``."""
        return {
            edge.data["as"]: self.result(self.node(edge.target))
            for edge in self.edges(kind=DEPENDS_ON, source=agent.id)
            if edge.data.get("as")
        }

    def ready(self) -> list[Node]:
        """Agents with work to do: not done, not waiting with an empty inbox, producers done."""
        return [
            agent
            for agent in self.agents()
            if self.status(agent) == "running"
            and all(self.terminal(producer) for producer in self.producers(agent))
        ]

    def launcher(self, agent: Node) -> Node | None:
        """The agent that owns this agent's ``from`` target; ``None`` when top-level."""
        edges = self.edges(kind=FROM, source=agent.id)
        if not edges:
            return None
        return self.node(self.node(edges[0].target).agent_id)

    def children(self, agent: Node) -> list[Node]:
        """Agents created by ``agent``, in ``seq`` order."""
        launched = [
            self.node(edge.source)
            for node in self.nodes(agent=agent.id)
            for edge in self.edges(kind=FROM, target=node.id)
            if self.node(edge.source).type == AGENT_START
        ]
        return sorted(launched, key=lambda child: child.seq)

    def launched(self, agent: Node) -> list[Node]:
        """Agents created anywhere in ``agent``'s history, a fork's shared prefix included.

        A launch hangs off the node before the cell that made it. Where a fork's
        history leaves a base agent right after that node, the cell isn't in it, and
        neither are that cell's launches; a cell the fork reran records them again.
        """
        nodes = self.history(agent)
        found: dict[str, Node] = {}
        for i, node in enumerate(nodes):
            following = nodes[i + 1] if i + 1 < len(nodes) else None
            skipped = self._skipped_launches(node) if following is not None and following.agent_id != node.agent_id else ()
            for edge in self.edges(kind=FROM, target=node.id):
                child = self.node(edge.source)
                if child.type == AGENT_START and child.id not in skipped:
                    found[child.id] = child
            if node.type == CODE:
                for child_id in node.launched():
                    if child_id in self:
                        found[child_id] = self.node(child_id)
        return sorted(found.values(), key=lambda child: child.seq)

    def _skipped_launches(self, node: Node) -> list[str]:
        """What the cell right after ``node`` in its own agent's history launched."""
        for edge in self.edges(kind=NEXT, source=node.id):
            after = self.node(edge.target)
            if after.agent_id == node.agent_id and after.type == CODE:
                return after.launched()
        return []

    def lineage(self, agent: Node) -> list[Node]:
        """The launcher chain from the top-level agent down to ``agent``."""
        chain: list[Node] = []
        current: Node | None = agent
        while current is not None:
            chain.append(current)
            current = self.launcher(current)
        chain.reverse()
        return chain

    def path(self, agent: Node) -> str:
        return ".".join(item.name for item in self.lineage(agent))

    def depth(self, agent: Node) -> int:
        return len(self.lineage(agent)) - 1

    def tree(self, agent: Node) -> list[Node]:
        """``agent`` and every agent it created, transitively, depth first."""
        found: list[Node] = []
        stack = [agent]
        while stack:
            current = stack.pop()
            found.append(current)
            stack.extend(reversed(self.children(current)))
        return found

    def usage(self, agents: list[Node] | None = None) -> LLMUsage:
        """Model usage spent by ``agents``, or anywhere in the graph.

        Our harness records it on ``llm_output`` nodes; external harnesses on the
        ``waiting`` or ``done`` node that ends each session turn.
        """
        nodes = self.nodes() if agents is None else [n for a in agents for n in self.nodes(agent=a.id)]
        return sum((LLMUsage.of(node.data["usage"]) for node in nodes if node.data.get("usage")), LLMUsage())

    def cost_usd(self, agents: list[Node] | None = None) -> float:
        """Dollar cost reported by external harnesses on their turn-ending nodes."""
        nodes = self.nodes() if agents is None else [n for a in agents for n in self.nodes(agent=a.id)]
        return sum(float(node.data.get("cost_usd") or 0) for node in nodes)

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
        name: str | None = None,
        depends_on: Iterable[Node] = (),
        inputs: Mapping[str, Any] | None = None,
        **args: Any,
    ) -> Node:
        """Add a root agent; keyword arguments become its REPL variables.

        Nothing runs here: a ``Flow`` steps the agent with the run's ``root_config``.
        """
        return self.spawn(
            task,
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

    def _require(self, *nodes: Node | None) -> None:
        for node in nodes:
            if node is not None and node.id not in self._nodes:
                raise InvariantError(f"{node.type} {node.id!r} is not in the graph")

    def _require_agent(self, agent: Node) -> None:
        self._require(agent)
        if agent.type != AGENT_START:
            raise InvariantError(f"expected an agent_start, not {agent.type} {agent.id!r}")

    def _default_name(self, config: str | None, by: Node | None) -> str:
        """The config name (``root`` for a root) for the first top-level agent of a kind, else ``<name>-<n>``."""
        base = config or "root"
        launcher = None if by is None else self.node(by.agent_id)
        peers = (
            self.children(launcher)
            if launcher is not None
            else [a for a in self.agents() if self.launcher(a) is None]
        )
        used = {peer.name for peer in peers}
        if launcher is None and base not in used:
            return base
        index = 0
        while f"{base}-{index}" in used:
            index += 1
        return f"{base}-{index}"

    def _last(self, agent_id: str) -> Node:
        """The agent's last own node: its ``agent_start`` until it appends."""
        return self._nodes[self._tail[agent_id]]

    def _append_unchecked(self, *items: Item, removing: bool = False) -> list[Item]:
        """Index and log ``items`` in order, checking only that ids are new.

        Structural rules are the caller's job: the history writes check them,
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
                if item.type == AGENT_START:
                    item.agent_id = item.agent_id or item.id
                    self._tail[item.id] = item.id
                elif item.type == MESSAGE:
                    self._inbox.setdefault(item.agent_id, []).append(item.id)
                self._nodes[item.id] = item
                self._out.setdefault(item.id, [])
                self._in.setdefault(item.id, [])
                self._by_agent.setdefault(item.agent_id, []).append(item.id)
                self._by_type.setdefault(item.type, []).append(item.id)
            else:
                if item.id in self._edges:
                    raise InvariantError(f"duplicate edge id {item.id!r}")
                item = replace(item, seq=seq)
                self._edges[item.id] = item
                self._out.setdefault(item.source, []).append(item.id)
                self._in.setdefault(item.target, []).append(item.id)
                target = self._nodes.get(item.target)
                if item.kind == NEXT and target is not None and target.type != AGENT_START:
                    self._tail[target.agent_id] = target.id
                    queued = self._inbox.get(target.agent_id)
                    if queued and target.id in queued:
                        queued.remove(target.id)
            self._log.append(item)
            logged.append(item)
        return logged


def check_invariants(g: Graph) -> None:
    """Walk the whole graph; raise ``InvariantError`` at the first broken rule.

    The graph's history writes keep these true one at a time. This walk is for a
    log read from disk: every node owned by an agent, one predecessor per node,
    each agent's history one chain from its ``agent_start``, forks only at landed
    nodes, an acyclic launcher tree, and acyclic ``depends_on`` declared before
    the consumer starts.
    """
    nodes = g._nodes
    for node in nodes.values():
        if node.type == AGENT_START:
            if node.agent_id != node.id:
                raise InvariantError(f"agent {node.id!r} must own itself")
        elif getattr(nodes.get(node.agent_id), "type", None) != AGENT_START:
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
            if target.type == AGENT_START:
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
            if source.type == AGENT_START:
                if source.id in launcher:
                    raise InvariantError(f"agent {source.id!r} already has a launcher")
                launcher[source.id] = target.agent_id
            elif source.type != MESSAGE:
                raise InvariantError(
                    f"a from edge leaves an agent_start or message, not {source.type}"
                )
        elif edge.kind == DEPENDS_ON:
            if source.type != AGENT_START or target.type != AGENT_START:
                raise InvariantError("depends_on edges connect agent_start nodes")
            if source.id == target.id:
                raise InvariantError(f"agent {source.id!r} cannot depend on itself")
            depends.setdefault(source.id, []).append(edge)
    for edge in forks:
        source = nodes[edge.source]
        if source.type != AGENT_START and source.id not in predecessor:
            raise InvariantError(f"cannot fork at undelivered {source.type} {source.id!r}")
    for agent in g.agents():
        reached, current = {agent.id}, agent.id
        while current in successor:
            current = successor[current]
            reached.add(current)
        for node_id in g._by_agent.get(agent.id, ()):
            node = nodes[node_id]
            if node_id in reached or node.type == MESSAGE and node_id not in predecessor:
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
