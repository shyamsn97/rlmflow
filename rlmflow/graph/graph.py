"""The run as a graph: agents' histories of nodes, and the edges between them.

An agent is its root node; ``root.root_data.history`` is its own nodes, root
first. A fork's history continues from the node it forked at. Edges say how
agents affect each other:

- ``spawn``: the node that launched a child -> the child's root
- ``fork``: the node a fork continues from -> the fork's root
- ``return``: a child's final node -> the node its result reached
- ``write``: the node that attached a node to another agent -> that node

The graph is mutable. ``edit``, ``delete``, ``remove`` and ``replace`` change it
and invalidate what depended on the change (see ``rlmflow.graph.edits``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from rlmflow.graph import edits, queries
from rlmflow.graph.config import (
    DEFAULT_QUERY,
    RECURSE,
    input_parts,
    transform_argument,
    validate_agent_name,
    validate_argument_name,
)
from rlmflow.graph.edits import GraphError, Invalidation
from rlmflow.graph.nodes import (
    DONE,
    FORK,
    READY,
    SELF,
    SPAWN,
    WRITE,
    Edge,
    LLMUsage,
    Node,
    RootData,
    new_id,
    strict_json,
)


class Graph:
    """Agents (root nodes), the nodes in their histories, and the edges between them."""

    def __init__(self, id: str | None = None) -> None:
        self.id = id or new_id("graph")
        self._roots: dict[str, Node] = {}
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._out: defaultdict[str, list[str]] = defaultdict(list)
        self._in: defaultdict[str, list[str]] = defaultdict(list)
        self._parent: dict[str, str] = {}
        self._children: defaultdict[str, list[str]] = defaultdict(list)
        self._seq = 0

    def __repr__(self) -> str:
        return f"Graph({self.id!r}, agents={len(self._roots)}, nodes={len(self._nodes)})"

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, item: object) -> bool:
        key = getattr(item, "id", item)
        return isinstance(key, str) and (key in self._nodes or key in self._edges)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.agents())

    # -- reads ----------------------------------------------------------------

    def agent(self, agent: str | Node) -> Node:
        """An agent's root node, by id or root node."""
        key = agent.id if isinstance(agent, Node) else agent
        try:
            return self._roots[key]
        except KeyError:
            raise KeyError(f"no agent {key!r} in this graph") from None

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id!r} in this graph") from None

    def owner(self, node: Node) -> Node:
        """The root of the agent whose history holds ``node``."""
        return self._roots[node.owner]

    def agents(self) -> list[Node]:
        """Every agent's root node, in creation order."""
        return list(self._roots.values())

    def roots(self) -> list[Node]:
        """Agents nobody launched, in creation order."""
        return [root for root in self._roots.values() if root.id not in self._parent]

    def nodes(self, *, agent: Node | str | None = None) -> list[Node]:
        """Nodes in creation order: everywhere, or one agent's own history."""
        if agent is not None:
            return list(self.agent(agent).root_data.history)
        return sorted(self._nodes.values(), key=lambda node: node.seq)

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

    def base(self, agent: Node) -> Node | None:
        """The node a fork's history continues from; ``None`` for an agent that isn't a fork."""
        for edge_id in self._in.get(agent.id, ()):
            edge = self._edges[edge_id]
            if edge.kind == FORK:
                return self._nodes.get(edge.source)
        return None

    def history(self, item: Node) -> list[Node]:
        """What an agent has seen: a fork's base history, then its own nodes (up to ``item`` for a node)."""
        root = self._roots[item.owner]
        if item is root:
            return [*self._prefix(root), *root.root_data.history]
        return self._upto(item)

    def _upto(self, node: Node) -> list[Node]:
        own = self._roots[node.owner].root_data.history
        return [*self._prefix(self._roots[node.owner]), *own[: own.index(node) + 1]]

    def _prefix(self, root: Node) -> list[Node]:
        base = self.base(root)
        return self._upto(base) if base is not None else []

    def tail(self, agent: Node) -> Node:
        """The last node of the agent's history."""
        return self.history(agent)[-1]

    def last_turn(self, agent: Node) -> Node | None:
        """The node that ended the agent's latest turn: the last one with ``seen``."""
        for node in reversed(self.history(agent)):
            if "seen" in node.data:
                return node
        return None

    def status(self, agent: Node) -> str:
        """``"done"``, ``"idle"`` (waiting for news), or ``"running"``.

        Unfinished nodes, or nodes from others that arrived after its last turn
        started, mean ``running``. Otherwise the last turn says: ``terminal`` is
        done, ``idle`` is idle, anything else is still working.
        """
        history = self.history(agent)
        if history[-1].state != DONE:
            return "running"
        turn = None
        for node in reversed(history):
            if "seen" in node.data:
                turn = node
                break
        if turn is None:
            return "running"
        if any(node.producer != SELF for node in history[turn.data["seen"] :]):
            return "running"
        if turn.data.get("terminal"):
            return "done"
        if turn.data.get("idle"):
            return "idle"
        return "running"

    def result(self, agent: Node) -> Any:
        """The agent's result, or ``None`` until it is done."""
        if self.status(agent) != "done":
            return None
        turn = self.last_turn(agent)
        return turn.result if turn is not None else None

    def parent(self, agent: Node) -> Node | None:
        """The agent that launched this one; ``None`` when top-level."""
        parent_id = self._parent.get(agent.id)
        return self._roots.get(parent_id) if parent_id else None

    def children(self, agent: Node) -> list[Node]:
        """Agents this one launched, in creation order."""
        return [self._roots[i] for i in self._children.get(agent.id, ()) if i in self._roots]

    def subtree(self, agent: Node) -> list[Node]:
        return queries.subtree(self, agent)

    def lineage(self, agent: Node) -> list[Node]:
        return queries.lineage(self, agent)

    def path(self, agent: Node) -> str:
        return queries.path(self, agent)

    def usage(self, agents: Iterable[Node] | None = None) -> LLMUsage:
        return queries.usage(self, agents)

    # -- writes ---------------------------------------------------------------

    def _stamp(self, item: Node | Edge) -> None:
        item.seq = self._seq
        self._seq += 1

    def add_agent(
        self,
        task: str = "",
        *,
        name: str | None = None,
        config: str | None = None,
        parent: Node | None = None,
        **args: Any,
    ) -> Node:
        """Add an agent and return its root node. Nothing runs here.

        Keyword arguments are the root's inputs, so they become REPL variables
        along with ``task``; wrap one in ``Input(value, description=...)`` to
        explain it. ``parent`` is the node that launched it, which gets a
        ``spawn`` edge; without one it is top-level. ``config`` names one of the
        Flow's configs: ``"default"`` for a child, while a top-level agent with
        none runs with the run's ``root_config``.
        """
        if parent is not None:
            if parent.id not in self._nodes:
                raise GraphError(f"{parent!r} is not in this graph")
            config = config or "default"
        if config is not None:
            validate_agent_name(config)
        if not isinstance(task, str):
            raise TypeError("agent task must be a string")
        for arg in args:
            validate_argument_name(arg)
        launcher = self._roots[parent.owner] if parent is not None else None
        name = name or self._default_name(config, launcher)
        validate_agent_name(name)
        task = task or DEFAULT_QUERY
        values = {arg: _unwrap(value, arg) for arg, value in args.items()}
        descriptions = {
            arg: wrapped[1] for arg, value in args.items() if (wrapped := input_parts(value))
        }
        root = Node("user", task, input={"task": task, **values})
        root.root_data = RootData(name, task, config_name=config, arg_descriptions=descriptions)
        self._add_root(root)
        if parent is not None:
            self.link(SPAWN, parent, root)
        return root

    def fork(
        self,
        at: Node,
        *,
        name: str | None = None,
        parent: Node | None = None,
        config: str | None = None,
    ) -> Node:
        """A new agent whose history continues from ``at``, through a ``fork`` edge.

        Nothing is copied. Forking at an agent's root starts it over. The fork
        runs with the owner's config unless ``config`` names another.
        """
        if at.id not in self._nodes:
            raise GraphError(f"cannot fork at {at!r}: it is not in this graph")
        owner = self._roots[at.owner].root_data
        if name is None:
            taken = {root.root_data.name for root in self._roots.values()}
            index = 0
            while f"{owner.name}-fork{index}" in taken:
                index += 1
            name = f"{owner.name}-fork{index}"
        validate_agent_name(name)
        root = Node("user", "", producer=SELF)
        root.root_data = RootData(
            name,
            owner.task,
            config_name=config if config is not None else owner.config_name,
            arg_descriptions=owner.arg_descriptions,
        )
        self._add_root(root)
        self.link(FORK, at, root)
        if parent is not None:
            self.link(SPAWN, parent, root)
        return root

    def _add_root(self, root: Node) -> None:
        root.owner = root.id
        root.root_data.history.append(root)
        self._stamp(root)
        self._roots[root.id] = root
        self._nodes[root.id] = root

    def append(self, agent: Node, node: Node) -> Node:
        """Add ``node`` to the end of ``agent``'s history; code that hasn't run is ready to run."""
        root = self.agent(agent)
        if node.id in self._nodes:
            raise GraphError(f"node {node.id!r} is already in the graph")
        if node.runs_code and node.finished_at is None and node.state == DONE:
            node.state = READY
        node.owner = root.id
        self._stamp(node)
        root.root_data.history.append(node)
        self._nodes[node.id] = node
        return node

    def send(
        self,
        agent: Node,
        text: str,
        *,
        by: Node | None = None,
        sender: str | None = None,
        **inputs: Any,
    ) -> Node:
        """Attach a message to the agent: a node with ``text``, binding ``inputs`` as variables.

        ``by`` is the node it comes from, which gets a ``write`` edge; ``sender``
        labels a message from the host, such as ``"user"``. The agent reads it
        at its next turn, which reopens it if it had finished.
        """
        if not isinstance(text, str):
            raise TypeError("message text must be a string")
        for name in inputs:
            validate_argument_name(name)
        if inputs:
            text += "\nSet: " + ", ".join(f"`{name}`" for name in inputs)
        producer = sender or (by.owner if by is not None else "user")
        node = Node("user", text, input=strict_json(inputs, "message inputs"), producer=producer)
        self.append(agent, node)
        if by is not None:
            self.link(WRITE, by, node)
        return node

    def link(self, kind: str, source: Any, target: Any, **data: Any) -> Edge:
        """Add an edge between two nodes; any kind but the engine's is the application's own."""
        edge = Edge(kind, getattr(source, "id", source), getattr(target, "id", target))
        for end in (edge.source, edge.target):
            if end not in self._nodes:
                raise GraphError(f"edge endpoint {end!r} is not in this graph")
        edge.data = strict_json(data, "edge data")
        self._stamp(edge)
        self._index(edge)
        return edge

    def unlink(self, edge: Edge | str) -> None:
        found = self._edges.get(getattr(edge, "id", edge))
        if found is None:
            raise KeyError(f"no edge {edge!r}")
        self._unlink(found)

    def _index(self, edge: Edge) -> None:
        self._edges[edge.id] = edge
        self._out[edge.source].append(edge.id)
        self._in[edge.target].append(edge.id)
        if edge.kind == SPAWN:
            source = self._nodes[edge.source].owner
            self._parent[edge.target] = source
            self._children[source].append(edge.target)

    def _unlink(self, edge: Edge) -> None:
        if self._edges.pop(edge.id, None) is None:
            return
        self._out[edge.source].remove(edge.id)
        self._in[edge.target].remove(edge.id)
        if edge.kind == SPAWN and (parent := self._parent.pop(edge.target, None)):
            self._children[parent].remove(edge.target)

    def _default_name(self, config: str | None, parent: Node | None) -> str:
        """The config name (``root`` for a root) for the first top-level agent of a kind, else ``<name>-<n>``."""
        base = config or "root"
        peers = self.children(parent) if parent is not None else self.roots()
        used = {peer.root_data.name for peer in peers}
        if parent is None and base not in used:
            return base
        index = 0
        while f"{base}-{index}" in used:
            index += 1
        return f"{base}-{index}"

    # -- edits ------------------------------------------------------------------

    def edit(self, node: Node, *, dry_run: bool = False, **fields: Any) -> Invalidation:
        """Change a node's ``text``, ``input``, ``code`` or ``result``; what used the old one redoes."""
        return self._apply(edits.edit, node, dry_run, **fields)

    def delete(self, node: Node, *, dry_run: bool = False) -> Invalidation:
        """Remove a node and everything after it from its agent's history."""
        return self._apply(edits.delete, node, dry_run)

    def remove(self, agent: Node, *, dry_run: bool = False) -> Invalidation:
        """Remove an agent, the agents it launched, and what they caused."""
        return self._apply(edits.remove, agent, dry_run)

    def replace(
        self,
        node: Node,
        index: int,
        value: Any = edits.MISSING,
        *,
        agent: Node | None = None,
        dry_run: bool = False,
    ) -> Invalidation:
        """Bind the node's ``index``-th saved call to ``value`` or another ``agent``; the node reruns."""
        return self._apply(edits.replace, node, dry_run, index, value, agent=agent)

    def _apply(
        self, edit: Any, target: Node, dry_run: bool, *args: Any, **kwargs: Any
    ) -> Invalidation:
        if target.id not in self._nodes:
            raise KeyError(f"{target!r} is not in this graph")
        if not dry_run:
            return edit(self, target, *args, **kwargs)
        trial = self.copy()
        agent = kwargs.get("agent")
        if agent is not None:
            kwargs["agent"] = trial.agent(agent.id)
        return edit(trial, trial.node(target.id), *args, **kwargs)

    # -- copies and files -------------------------------------------------------

    def copy(self) -> Graph:
        """A full, independent copy (same ids), for editing without losing the original."""
        from rlmflow.graph.persistence import from_records, to_records

        return from_records(to_records(self))

    def save(self, path: str | Path) -> Path:
        """Write the graph and its large inputs as a run directory."""
        from rlmflow.graph import persistence

        return persistence.save_graph(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """Read a run directory written by ``save``."""
        from rlmflow.graph import persistence

        return persistence.load_graph(path)


def _unwrap(value: Any, path: str) -> Any:
    """An argument with its ``Input`` description wrappers removed."""

    def unwrap(at: str, item: Any) -> Any:
        wrapped = input_parts(item)
        if wrapped is not None:
            return transform_argument(wrapped[0], unwrap, path=at)
        return RECURSE

    return transform_argument(value, unwrap, path=path)


__all__ = ["Graph", "GraphError", "Invalidation"]
