"""The run as a graph: agents, the nodes in their histories, and the edges between them.

Each ``Agent`` owns its history (``agent.nodes``) and an inbox of undelivered
messages. A fork's history continues from ``agent.base``. Edges say how agents
affect each other:

- ``spawn``: the cell (or agent) that launched a child -> the child agent
- ``return``: a child's ``done`` -> the cell that awaited it, or the message that delivered it
- ``message``: the cell that sent a message -> the ``message`` node

The graph is mutable. ``edit``, ``delete``, ``remove`` and ``replace`` change it
and invalidate everything downstream along those edges, so the next run redoes
only what the change affected.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from rlmflow.graph import calls, queries
from rlmflow.graph.config import DEFAULT_QUERY, validate_agent_name, validate_argument_name
from rlmflow.graph.nodes import (
    CODE,
    DONE,
    IDLE,
    IMAGES_KEY,
    LLM,
    MESSAGE,
    PENDING,
    RETURN,
    SPAWN,
    Agent,
    Edge,
    LLMUsage,
    Node,
    new_id,
    strict_json,
)

T = TypeVar("T")
_MISSING = object()

#: What a cell's run wrote; an invalidated cell loses them until it runs again.
CELL_RESULTS = ("output", "error", "dead", IMAGES_KEY, "workspace")


class GraphError(Exception):
    """A write that would leave the graph inconsistent."""


@dataclass
class Invalidation:
    """What an edit changed: agents rewound or removed, nodes dropped, cells marked to re-run."""

    rewound: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    rerun: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.rewound or self.removed or self.dropped or self.rerun)

    def _add(self, kind: str, item: str) -> None:
        items = getattr(self, kind)
        if item not in items:
            items.append(item)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "rewound": list(self.rewound),
            "removed": list(self.removed),
            "dropped": list(self.dropped),
            "rerun": list(self.rerun),
        }


def locked(write: Callable[..., T]) -> Callable[..., T]:
    @wraps(write)
    def under_lock(g: Graph, *args: Any, **kwargs: Any) -> T:
        with g._lock:
            return write(g, *args, **kwargs)

    return under_lock


class Graph:
    """Agents, their nodes, and the edges between them."""

    def __init__(self, id: str | None = None) -> None:
        self.id = id or new_id("graph")
        self._agents: dict[str, Agent] = {}
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._out: defaultdict[str, list[str]] = defaultdict(list)
        self._in: defaultdict[str, list[str]] = defaultdict(list)
        self._parent: dict[str, str] = {}
        self._children: defaultdict[str, list[str]] = defaultdict(list)
        #: Fork agent ids by the node their history continues from.
        self._forks: defaultdict[str, list[str]] = defaultdict(list)
        self._seq = 0
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[str, Any], None]] = []
        #: Every edit, with what it invalidated.
        self.edits: list[dict[str, Any]] = []
        #: Bumped on every write; a checkpointer saves when it changes.
        self.revision = 0
        #: Agents with a step in flight; a Flow keeps this, and edits refuse them.
        self.busy: set[str] = set()

    def __repr__(self) -> str:
        return f"Graph({self.id!r}, agents={len(self._agents)}, nodes={len(self._nodes)})"

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, item: object) -> bool:
        key = getattr(item, "id", item)
        return isinstance(key, str) and (
            key in self._agents or key in self._nodes or key in self._edges
        )

    def __iter__(self) -> Iterator[Agent]:
        return iter(self.agents())

    # -- reads ----------------------------------------------------------------

    def agent(self, agent_id: str | Agent) -> Agent:
        key = agent_id.id if isinstance(agent_id, Agent) else agent_id
        try:
            return self._agents[key]
        except KeyError:
            raise KeyError(f"no agent {key!r} in this graph") from None

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id!r} in this graph") from None

    def owner(self, node: Node) -> Agent:
        """The agent whose history or inbox holds ``node``."""
        return self._agents[node.agent_id]

    def agents(self) -> list[Agent]:
        """Every agent, in creation order."""
        return list(self._agents.values())

    def roots(self) -> list[Agent]:
        """Agents nobody launched, in creation order."""
        return [agent for agent in self._agents.values() if agent.id not in self._parent]

    def nodes(self, type: str | None = None, *, agent: Agent | str | None = None) -> list[Node]:
        """Nodes in creation order: everywhere, or one agent's own history then its inbox."""
        if agent is not None:
            owner = self.agent(agent)
            found = [*owner.nodes, *owner.inbox]
        else:
            found = sorted(self._nodes.values(), key=lambda node: node.seq)
        return [node for node in found if type is None or node.type == type]

    def edges(
        self,
        *,
        kind: str | None = None,
        source: str | None = None,
        target: str | None = None,
        touching: str | None = None,
    ) -> list[Edge]:
        """Edges in creation order, filtered by kind and endpoint ids."""
        if touching is not None:
            ids = [*self._out.get(touching, ()), *self._in.get(touching, ())]
        elif source is not None:
            ids = list(self._out.get(source, ()))
        elif target is not None:
            ids = list(self._in.get(target, ()))
        else:
            ids = list(self._edges)
        found = [self._edges[edge_id] for edge_id in dict.fromkeys(ids)]
        return sorted(
            (
                edge
                for edge in found
                if (kind is None or edge.kind == kind)
                and (source is None or edge.source == source)
                and (target is None or edge.target == target)
            ),
            key=lambda edge: edge.seq,
        )

    def history(self, item: Agent | Node) -> list[Node]:
        """What an agent has seen: a fork's base history, then its own nodes (up to ``item`` for a node)."""
        if isinstance(item, Agent):
            prefix = self.history(item.base) if item.base is not None else []
            return [*prefix, *item.nodes]
        owner = self.owner(item)
        if item in owner.inbox:
            return [*self.history(owner), item]
        prefix = self.history(owner.base) if owner.base is not None else []
        return [*prefix, *owner.nodes[: owner.nodes.index(item) + 1]]

    def status(self, agent: Agent) -> str:
        """``"done"``, ``"idle"`` (a session waiting for messages), or ``"running"``.

        Read from the last node: ``done`` and ``idle`` stand until a message
        arrives; anything else, or a cell still to run, is ``running``.
        """
        last = _last(agent)
        if last is None or last.pending or agent.inbox:
            return "running"
        if last.type == DONE:
            return "done"
        if last.type == IDLE:
            return "idle"
        return "running"

    def result(self, agent: Agent) -> Any:
        """The agent's result, or ``None`` until it is done."""
        last = _last(agent)
        return last.data.get("result") if last is not None and last.type == DONE else None

    def parent(self, agent: Agent) -> Agent | None:
        """The agent that launched this one; ``None`` when top-level."""
        parent_id = self._parent.get(agent.id)
        return self._agents.get(parent_id) if parent_id else None

    def children(self, agent: Agent) -> list[Agent]:
        """Agents this one launched, in creation order."""
        return [self._agents[i] for i in self._children.get(agent.id, ()) if i in self._agents]

    def subtree(self, agent: Agent) -> list[Agent]:
        return queries.subtree(self, agent)

    def lineage(self, agent: Agent) -> list[Agent]:
        return queries.lineage(self, agent)

    def path(self, agent: Agent) -> str:
        return queries.path(self, agent)

    def usage(self, agents: Iterable[Agent] | None = None) -> LLMUsage:
        return queries.usage(self, agents)

    # -- writes ---------------------------------------------------------------

    def subscribe(self, fn: Callable[[str, Any], None]) -> Callable[[], None]:
        """Call ``fn(kind, item)`` after each write; returns the unsubscribe function.

        ``kind`` is ``"agent"``, ``"node"`` (also when a cell finishes),
        ``"edge"``, ``"deliver"`` or ``"edit"``.
        """
        self._subscribers.append(fn)
        return lambda: self._subscribers.remove(fn) if fn in self._subscribers else None

    def _emit(self, kind: str, item: Any) -> None:
        self.revision += 1
        for fn in list(self._subscribers):
            fn(kind, item)

    def _stamp(self, item: Agent | Node | Edge) -> None:
        item.seq = self._seq
        self._seq += 1

    @locked
    def add_agent(
        self,
        task: str = "",
        *,
        name: str | None = None,
        config: str | None = None,
        parent: Node | Agent | None = None,
        base: Node | None = None,
        **args: Any,
    ) -> Agent:
        """Add an agent; keyword arguments become its REPL variables. Nothing runs here.

        ``parent`` is the cell (or agent) that launched it, which gets a ``spawn``
        edge; without one it is top-level. ``config`` names one of the Flow's
        configs: ``"default"`` for a child, while a top-level agent with none runs
        with the run's ``root_config``. ``base`` makes it a fork whose history
        continues from that node.
        """
        launcher = None
        if parent is not None:
            launcher = parent if isinstance(parent, Agent) else self._agents.get(parent.agent_id)
            if launcher is None or parent.id not in self:
                raise GraphError(f"{parent!r} is not in this graph")
            config = config or "default"
        if base is not None and base not in [*self.owner(base).nodes]:
            raise GraphError(f"cannot fork at {base.type} {base.id!r}: it is not in a history")
        if config is not None:
            validate_agent_name(config)
        if not isinstance(task, str):
            raise TypeError("agent task must be a string")
        for arg in args:
            validate_argument_name(arg)
        name = name or self._default_name(config, launcher)
        validate_agent_name(name)
        agent = Agent(task or DEFAULT_QUERY, name=name, args=args, config=config)
        agent.base = base
        self._stamp(agent)
        self._agents[agent.id] = agent
        if base is not None:
            self._forks[base.id].append(agent.id)
        self._emit("agent", agent)
        if parent is not None:
            self._link(SPAWN, parent.id, agent.id)
        return agent

    def fork(
        self,
        at: Node | Agent,
        *,
        name: str | None = None,
        parent: Node | Agent | None = None,
        config: str | None = None,
    ) -> Agent:
        """A new agent with the same task and arguments whose history is everything up to ``at``.

        Nothing is copied. Forking an agent rather than a node starts it over.
        It runs with the owner's config unless ``config`` names another.
        """
        owner = at if isinstance(at, Agent) else self.owner(at)
        if name is None:
            taken = {agent.name for agent in self._agents.values()}
            index = 0
            while f"{owner.name}-fork{index}" in taken:
                index += 1
            name = f"{owner.name}-fork{index}"
        return self.add_agent(
            owner.task,
            name=name,
            config=config if config is not None else owner.config_name,
            parent=parent,
            base=at if isinstance(at, Node) else None,
            **deepcopy(owner.args),
        )

    @locked
    def append(self, agent: Agent, node: Node) -> Node:
        """Add ``node`` to the end of ``agent``'s history."""
        self.agent(agent)
        if node.id in self._nodes:
            raise GraphError(f"node {node.id!r} is already in the graph")
        node.agent_id = agent.id
        self._stamp(node)
        agent.nodes.append(node)
        self._nodes[node.id] = node
        self._emit("node", node)
        return node

    @locked
    def update(self, node: Node, **data: Any) -> Node:
        """Change a node's data in place, as when its cell finishes; no edit, nothing invalidated."""
        self.node(node.id)
        node.data.update(data)
        self._emit("node", node)
        return node

    @locked
    def send(
        self,
        agent: Agent,
        text: str,
        *,
        sender: str | None = None,
        by: Node | Agent | None = None,
        **data: Any,
    ) -> Node:
        """Queue a ``message`` in the agent's inbox; it is delivered at the agent's next turn.

        ``by`` is the cell (or agent) it comes from, which gets a ``message`` edge;
        ``sender`` is a label for host messages, such as ``"user"``.
        """
        self.agent(agent)
        if not isinstance(text, str):
            raise TypeError("message text must be a string")
        origin = None
        if by is not None:
            origin = by if isinstance(by, Agent) else self._agents.get(by.agent_id)
        label = sender or (origin.id if origin is not None else "user")
        message = Node(MESSAGE, text=text, sender=label, **data)
        message.agent_id = agent.id
        self._stamp(message)
        agent.inbox.append(message)
        self._nodes[message.id] = message
        self._emit("node", message)
        if by is not None:
            self._link(MESSAGE, by.id, message.id)
        return message

    @locked
    def deliver(self, agent: Agent) -> list[Node]:
        """Move queued messages into the agent's history, unless it is mid-action."""
        last = _last(agent)
        if not agent.inbox or (last is not None and (last.type == LLM or last.pending)):
            return []
        delivered, agent.inbox = agent.inbox, []
        agent.nodes.extend(delivered)
        self._emit("deliver", agent)
        return delivered

    @locked
    def add_edge(
        self, kind: str | Edge, source: Any = None, target: Any = None, **data: Any
    ) -> Edge:
        """Add an application edge between two agents or nodes; the engine never reads it."""
        edge = kind if isinstance(kind, Edge) else None
        if edge is None:
            edge = Edge(kind, getattr(source, "id", source), getattr(target, "id", target), data)
        if edge.structural:
            raise GraphError(f"{edge.kind!r} edges are written by the engine")
        if edge.id in self._edges:
            raise GraphError(f"edge {edge.id!r} is already in this graph")
        for end in (edge.source, edge.target):
            if end not in self._agents and end not in self._nodes:
                raise GraphError(f"edge endpoint {end!r} is not in this graph")
        edge.data = strict_json(edge.data, "edge data")
        return self._add_edge(edge)

    @locked
    def remove_edge(self, edge: Edge | str) -> None:
        found = self._edges.get(getattr(edge, "id", edge))
        if found is None:
            raise KeyError(f"no edge {edge!r}")
        if found.structural:
            raise GraphError(f"{found.kind!r} edges are written by the engine")
        self._unlink(found)
        self._emit("edge", found)

    @locked
    def link_return(self, done: Node, consumer: Node, **data: Any) -> Edge:
        """The ``return`` edge from a ``done`` to the cell or message its result reached."""
        for edge in self.edges(kind=RETURN, source=done.id, target=consumer.id):
            return edge
        return self._link(RETURN, done.id, consumer.id, **data)

    def _link(self, kind: str, source: str, target: str, **data: Any) -> Edge:
        return self._add_edge(Edge(kind, source, target, data))

    def _add_edge(self, edge: Edge) -> Edge:
        self._stamp(edge)
        self._index(edge)
        self._emit("edge", edge)
        return edge

    def _index(self, edge: Edge) -> None:
        self._edges[edge.id] = edge
        self._out[edge.source].append(edge.id)
        self._in[edge.target].append(edge.id)
        if edge.kind == SPAWN:
            source = (
                self._agents.get(edge.source) or self._agents[self._nodes[edge.source].agent_id]
            )
            self._parent[edge.target] = source.id
            self._children[source.id].append(edge.target)

    def _unlink(self, edge: Edge) -> None:
        if self._edges.pop(edge.id, None) is None:
            return
        self._out[edge.source].remove(edge.id)
        self._in[edge.target].remove(edge.id)
        if edge.kind == SPAWN and (parent := self._parent.pop(edge.target, None)):
            self._children[parent].remove(edge.target)

    def _default_name(self, config: str | None, parent: Agent | None) -> str:
        """The config name (``root`` for a root) for the first top-level agent of a kind, else ``<name>-<n>``."""
        base = config or "root"
        peers = self.children(parent) if parent is not None else self.roots()
        used = {peer.name for peer in peers}
        if parent is None and base not in used:
            return base
        index = 0
        while f"{base}-{index}" in used:
            index += 1
        return f"{base}-{index}"

    # -- edits ------------------------------------------------------------------

    def edit(self, node: Node, *, dry_run: bool = False, **data: Any) -> Invalidation:
        """Change a node's data. Its agent is rewound to it, and whatever used its old content redoes."""
        return self._apply("edit", node, data, dry_run, lambda g, n: g._edit(n, data))

    def delete(self, node: Node, *, dry_run: bool = False) -> Invalidation:
        """Remove a node and everything after it from its agent's history."""
        return self._apply("delete", node, {}, dry_run, lambda g, n: g._delete(n))

    def remove(self, agent: Agent, *, dry_run: bool = False) -> Invalidation:
        """Remove an agent, its subtree, and what they caused."""
        return self._apply(
            "remove", agent, {}, dry_run, lambda g, a: g._remove_agent(a, Invalidation())
        )

    def replace(
        self,
        cell: Node,
        index: int,
        value: Any = _MISSING,
        *,
        agent: Agent | None = None,
        dry_run: bool = False,
    ) -> Invalidation:
        """Bind the cell's ``index``-th saved call to ``value`` or another ``agent``; the cell re-runs.

        A launch takes another agent (the old child is removed). An ``await``
        takes a value or the agent to await instead. A tool call takes a value.
        """
        if (value is _MISSING) == (agent is None):
            raise TypeError("replace takes exactly one of a value or agent=")
        saved = cell.data.get("calls") or []
        details: dict[str, Any] = {"index": index}
        if 0 <= index < len(saved):
            details = {"tool": saved[index]["tool"], **details}
        if agent is not None:
            details["agent"] = agent.id
        else:
            details["value"] = strict_json(value, "replacement value")
        return self._apply(
            "replace",
            cell,
            details,
            dry_run,
            lambda g, c: g._replace(c, index, value, g.agent(agent.id) if agent else None),
        )

    def _apply(
        self,
        op: str,
        target: Agent | Node,
        details: dict[str, Any],
        dry_run: bool,
        change: Callable[[Graph, Any], Invalidation],
    ) -> Invalidation:
        with self._lock:
            if target.id not in self:
                raise KeyError(f"{target!r} is not in this graph")
            # Run it on a copy first, so an edit that fails part-way changes nothing.
            trial = self.copy()
            trial.busy = set(self.busy)
            invalidation = change(trial, trial._find(target.id))
            if dry_run:
                return invalidation
            invalidation = change(self, self._find(target.id))
            self.edits.append({"op": op, "target": target.id, **details, **invalidation.as_dict()})
            self._emit("edit", invalidation)
            return invalidation

    def _find(self, item_id: str) -> Agent | Node:
        return self._agents.get(item_id) or self._nodes[item_id]

    def _edit(self, node: Node, data: dict[str, Any]) -> Invalidation:
        inv = Invalidation()
        owner = self.owner(node)
        self._check_free(owner)
        node.data.update(strict_json(data, "node data"))
        if node in owner.inbox:
            return inv
        for edge in self.edges(kind=RETURN, source=node.id):
            self._unlink(edge)
            self._stale(self._nodes.get(edge.target), inv)
        self._cut(owner, owner.nodes.index(node) + 1, inv)
        for fork_id in list(self._forks.get(node.id, ())):
            if fork_id in self._agents:
                self._cut(self._agents[fork_id], 0, inv)
        return inv

    def _delete(self, node: Node) -> Invalidation:
        inv = Invalidation()
        owner = self.owner(node)
        self._check_free(owner)
        if node in owner.inbox:
            self._retract(node, inv)
        else:
            self._cut(owner, owner.nodes.index(node), inv)
        return inv

    def _replace(self, cell: Node, index: int, value: Any, agent: Agent | None) -> Invalidation:
        inv = Invalidation()
        owner = self.owner(cell)
        self._check_free(owner)
        saved = cell.data.get("calls") or []
        if not 0 <= index < len(saved):
            raise GraphError(f"cell {cell.id!r} saved no call {index}")
        entry = saved[index]
        tool, old = entry["tool"], self._agents.get(entry.get("agent", ""))
        if tool == calls.LAUNCH:
            if agent is None:
                raise GraphError("a launch can only be replaced by another agent")
            for edge in self.edges(kind=SPAWN, target=agent.id):
                self._unlink(edge)
            self._link(SPAWN, cell.id, agent.id)
            if old is not None and old is not agent:
                calls.rebind(cell, old.id, agent.id)
                self._remove_agent(old, inv)
            entry["agent"] = agent.id
        elif tool == calls.AWAIT:
            replaced = [
                edge.source
                for edge in self.edges(kind=RETURN, target=cell.id)
                if old is not None and self._nodes[edge.source].agent_id == old.id
            ]
            call_key = entry.get("key")
            entry.clear()
            entry.update(
                tool=tool, key=call_key, **({"agent": agent.id} if agent else {"value": value})
            )
            self._rerun(cell, inv)
            # The old agent's result reached this cell and was swapped out, so it isn't
            # delivered again as a message.
            for done in replaced:
                self._link(RETURN, done, cell.id, replaced=True)
            return inv
        else:
            if agent is not None:
                raise GraphError(f"a {tool} call can only be replaced by a value")
            call_key = entry.get("key")
            entry.clear()
            entry.update(tool=tool, key=call_key, value=strict_json(value, "value"))
        self._rerun(cell, inv)
        return inv

    def _check_free(self, agent: Agent) -> None:
        if agent.id in self.busy:
            raise GraphError(f"agent {agent.name!r} is stepping; edit it once its step lands")

    def _cut(self, agent: Agent, keep: int, inv: Invalidation) -> None:
        """Keep ``agent.nodes[:keep]``; drop the rest and everything it caused."""
        dropped = agent.nodes[keep:]
        if not dropped:
            return
        self._check_free(agent)
        del agent.nodes[keep:]
        inv._add("rewound", agent.id)
        for node in reversed(dropped):
            self._drop(node, inv)

    def _drop(self, node: Node, inv: Invalidation) -> None:
        """Forget a node, and undo what it caused: launches, messages, results it delivered, forks."""
        if self._nodes.pop(node.id, None) is None:
            return
        inv._add("dropped", node.id)
        for edge in self.edges(source=node.id):
            self._unlink(edge)
            if edge.kind == SPAWN and edge.target in self._agents:
                self._remove_agent(self._agents[edge.target], inv)
            elif edge.kind == MESSAGE and edge.target in self._nodes:
                self._retract(self._nodes[edge.target], inv)
            elif edge.kind == RETURN:
                self._stale(self._nodes.get(edge.target), inv)
        for edge in self.edges(target=node.id):
            self._unlink(edge)
        for fork_id in self._forks.pop(node.id, ()):
            if fork_id in self._agents:
                self._remove_agent(self._agents[fork_id], inv)

    def _retract(self, message: Node, inv: Invalidation) -> None:
        """Take back a message: from the inbox, or by rewinding its recipient to before it."""
        recipient = self._agents.get(message.agent_id)
        if recipient is None:
            return
        if message in recipient.inbox:
            recipient.inbox.remove(message)
            self._drop(message, inv)
        elif message in recipient.nodes:
            self._cut(recipient, recipient.nodes.index(message), inv)

    def _stale(self, consumer: Node | None, inv: Invalidation) -> None:
        """A result that ``consumer`` used changed: a message is taken back, a cell re-runs."""
        if consumer is None or consumer.id not in self._nodes:
            return
        if consumer.type == MESSAGE:
            self._retract(consumer, inv)
        else:
            self._rerun(consumer, inv)

    def _rerun(self, cell: Node, inv: Invalidation) -> None:
        """Rewind the cell's agent to it and mark it to run again; forks from it start over."""
        owner = self.owner(cell)
        self._check_free(owner)
        for edge in self.edges(kind=RETURN, target=cell.id):
            self._unlink(edge)
        for fork_id in list(self._forks.pop(cell.id, ())):
            if fork_id in self._agents:
                self._remove_agent(self._agents[fork_id], inv)
        self._cut(owner, owner.nodes.index(cell) + 1, inv)
        for key in CELL_RESULTS:
            cell.data.pop(key, None)
        cell.type = CODE
        cell.data[PENDING] = True
        inv._add("rewound", owner.id)
        inv._add("rerun", cell.id)

    def _remove_agent(self, agent: Agent, inv: Invalidation) -> Invalidation:
        if agent.id not in self._agents:
            return inv
        self._check_free(agent)
        self._cut(agent, 0, inv)
        for message in list(agent.inbox):
            agent.inbox.remove(message)
            self._drop(message, inv)
        for edge in self.edges(touching=agent.id):
            self._unlink(edge)
            if edge.kind == SPAWN and edge.source == agent.id and edge.target in self._agents:
                self._remove_agent(self._agents[edge.target], inv)
        if agent.base is not None and agent.id in self._forks.get(agent.base.id, ()):
            self._forks[agent.base.id].remove(agent.id)
        del self._agents[agent.id]
        self._children.pop(agent.id, None)
        inv._add("removed", agent.id)
        if agent.id in inv.rewound:
            inv.rewound.remove(agent.id)
        return inv

    # -- copies and files -------------------------------------------------------

    def copy(self) -> Graph:
        """A full, independent copy (same ids), for editing without losing the original."""
        from rlmflow.graph.persistence import from_records, to_records

        return from_records(to_records(self))

    def save(self, path: str | Path) -> Path:
        """Write the graph and its argument blobs as a run directory."""
        from rlmflow.graph import persistence

        return persistence.save_graph(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """Read a run directory written by ``save``."""
        from rlmflow.graph import persistence

        return persistence.load_graph(path)


def _last(agent: Agent) -> Node | None:
    """The agent's latest node: its own last node, else a fork's base."""
    return agent.nodes[-1] if agent.nodes else agent.base


__all__ = ["Graph", "GraphError", "Invalidation"]
