"""Agent-rooted node chains and the edges between them.

``node.root`` is the ``AgentRoot`` that starts its chain; ``root.nodes()`` is
the whole chain, and ``node.transcript()`` the part through ``node``. Edges say
how agent subtrees affect each other:

- ``spawn``: the node that launched a child -> the child's root
- ``return``: a child's final node -> the node its result reached
- ``write``: the node that attached a node to another agent -> that node

The graph is mutable. ``edit``, ``delete``, ``remove`` and ``replace`` change it
and invalidate what depended on the change (see ``rlmflow.graph.edits``).
``fork`` leaves it alone and returns a cut copy.
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
    READY,
    SPAWN,
    WRITE,
    AgentRoot,
    Edge,
    LLMUsage,
    Node,
    new_id,
    strict_json,
)


class Graph:
    """Agent roots, their nodes, and the edges between them."""

    def __init__(self, id: str | None = None) -> None:
        self.id = id or new_id("graph")
        self._roots: dict[str, AgentRoot] = {}
        self._nodes: dict[str, Node] = {}
        #: AgentRoot id -> its nodes in order; an index, not saved.
        self._chains: dict[str, list[Node]] = {}
        self._edges: dict[str, Edge] = {}
        self._out: defaultdict[str, list[str]] = defaultdict(list)
        self._in: defaultdict[str, list[str]] = defaultdict(list)
        self._parent: dict[str, str] = {}
        self._children: defaultdict[str, list[str]] = defaultdict(list)
        self._seq = 0
        #: The application's; ``fork`` records ``forked_from`` here. Saved with the graph.
        self.metadata: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Graph({self.id!r}, agents={len(self._roots)}, nodes={len(self._nodes)})"

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, item: object) -> bool:
        key = getattr(item, "id", item)
        return isinstance(key, str) and (key in self._nodes or key in self._edges)

    def __iter__(self) -> Iterator[AgentRoot]:
        return iter(self.agent_roots())

    def __getitem__(self, node_id: str) -> Node:
        return self.node(node_id)

    # -- reads ----------------------------------------------------------------

    def agent_root(self, node: str | Node) -> AgentRoot:
        """An agent root, by its id or any node in its chain."""
        if isinstance(node, Node):
            key = node.root_id if node.root_id in self._roots else node.id
        else:
            key = node
        try:
            return self._roots[key]
        except KeyError:
            raise KeyError(f"no agent root {key!r} in this graph") from None

    def node(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id!r} in this graph") from None

    def agent_roots(self) -> list[AgentRoot]:
        """Every agent root, including launched subagents, in creation order."""
        return list(self._roots.values())

    def top_level_roots(self) -> list[AgentRoot]:
        """Agent roots nobody launched, in creation order."""
        return [root for root in self._roots.values() if root.id not in self._parent]

    @property
    def root(self) -> AgentRoot:
        """The one top-level agent root; raises if there are none or several."""
        roots = self.top_level_roots()
        if len(roots) != 1:
            raise LookupError(f"this graph has {len(roots)} top-level agent roots, not one")
        return roots[0]

    def nodes(self) -> list[Node]:
        """Every node, in creation order."""
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

    def usage(self, roots: Iterable[AgentRoot] | None = None) -> LLMUsage:
        return queries.usage(self, roots)

    # -- writes ---------------------------------------------------------------

    def _stamp(self, item: Node | Edge) -> None:
        item.seq = self._seq
        self._seq += 1

    def start(
        self,
        task: str = "",
        *,
        agent: str = "default",
        name: str | None = None,
        parent: Node | None = None,
        **args: Any,
    ) -> AgentRoot:
        """Add and return the ``AgentRoot`` for ``agent`` acting on ``task``.

        Keyword arguments are the root's inputs, so they become REPL
        variables along with ``task``; wrap one in ``Input(value,
        description=...)`` to explain it. ``parent`` is the node that launched
        it, which gets a ``spawn`` edge; without one it is top-level. ``agent``
        names one of the Flow's agents.
        """
        if parent is not None and parent.id not in self._nodes:
            raise GraphError(f"{parent!r} is not in this graph")
        validate_agent_name(agent)
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        for arg in args:
            validate_argument_name(arg)
        launcher = self._roots[parent.root_id] if parent is not None else None
        name = name or self._default_name(agent, launcher)
        validate_agent_name(name)
        task = task or DEFAULT_QUERY
        values = {arg: _unwrap(value, arg) for arg, value in args.items()}
        descriptions = {
            arg: wrapped[1] for arg, value in args.items() if (wrapped := input_parts(value))
        }
        root = AgentRoot(
            "user",
            task,
            input={"task": task, **values},
            name=name,
            task=task,
            agent=agent,
            arg_descriptions=descriptions,
        )
        self._add_root(root)
        if parent is not None:
            self.link(SPAWN, parent, root)
        return root

    def _add_root(self, root: AgentRoot) -> None:
        root.root_id = root.id
        root.repl_id = root.repl_id or root.id
        root.graph = self
        self._stamp(root)
        self._roots[root.id] = root
        self._nodes[root.id] = root
        self._chains[root.id] = [root]

    def append(self, at: Node, node: Node) -> Node:
        """Add ``node`` to ``at``'s root; code that hasn't run is ready to run."""
        root = self.agent_root(at)
        if isinstance(node, AgentRoot):
            raise GraphError("an agent root is added with start, not append")
        if node.id in self._nodes:
            raise GraphError(f"node {node.id!r} is already in the graph")
        if node.terminal and root.status() == "done":
            raise GraphError(
                f"agent {root.name!r} already finished; send it a message to reopen it"
            )
        if node.runs_code and node.finished_at is None and node.state == DONE:
            node.state = READY
        node.root_id = root.id
        node.graph = self
        self._stamp(node)
        self._chains[root.id].append(node)
        self._nodes[node.id] = node
        return node

    def send(
        self,
        at: Node,
        text: str,
        *,
        by: Node | None = None,
        sender: str | None = None,
        **inputs: Any,
    ) -> Node:
        """Attach a message to ``at``'s root: a node with ``text``, binding ``inputs`` as variables.

        ``by`` is the node it comes from, which gets a ``write`` edge; ``sender``
        labels a message from the host, such as ``"user"``. The run reads it
        at its next step, which reopens it if it had finished.
        """
        if not isinstance(text, str):
            raise TypeError("message text must be a string")
        for name in inputs:
            validate_argument_name(name)
        if inputs:
            text += "\nSet: " + ", ".join(f"`{name}`" for name in inputs)
        producer = sender or (by.root_id if by is not None else "user")
        node = Node("user", text, input=strict_json(inputs, "message inputs"), producer=producer)
        self.append(at, node)
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
            source = self._nodes[edge.source].root_id
            self._parent[edge.target] = source
            self._children[source].append(edge.target)

    def _unlink(self, edge: Edge) -> None:
        if self._edges.pop(edge.id, None) is None:
            return
        self._out[edge.source].remove(edge.id)
        self._in[edge.target].remove(edge.id)
        if edge.kind == SPAWN and (parent := self._parent.pop(edge.target, None)):
            self._children[parent].remove(edge.target)

    def _default_name(self, agent: str, parent: AgentRoot | None) -> str:
        """The first root's agent name, then ``<agent>-<n>`` for its peers."""
        base = "root" if parent is None and agent == "default" else agent
        peers = parent.children() if parent is not None else self.top_level_roots()
        used = {peer.name for peer in peers}
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
        """Remove a node and its agent's nodes after it."""
        return self._apply(edits.delete, node, dry_run)

    def remove(self, root: Node, *, dry_run: bool = False) -> Invalidation:
        """Remove an agent root, its subtree, and what they caused."""
        return self._apply(edits.remove, root, dry_run)

    def replace(
        self,
        node: Node,
        index: int,
        value: Any = edits.MISSING,
        *,
        run: Node | None = None,
        dry_run: bool = False,
    ) -> Invalidation:
        """Bind the node's ``index``-th saved call to ``value`` or another ``run``; the node reruns."""
        return self._apply(edits.replace, node, dry_run, index, value, run=run)

    def _apply(
        self, edit: Any, target: Node, dry_run: bool, *args: Any, **kwargs: Any
    ) -> Invalidation:
        if target.id not in self._nodes:
            raise KeyError(f"{target!r} is not in this graph")
        if not dry_run:
            return edit(self, target, *args, **kwargs)
        trial = self.copy()
        run = kwargs.get("run")
        if run is not None:
            kwargs["run"] = trial.agent_root(run.id)
        return edit(trial, trial.node(target.id), *args, **kwargs)

    # -- copies and files -------------------------------------------------------

    def fork(self, at: Node, *, repl: str = "copy") -> Graph:
        """A new graph: a copy of this one, with the same ids, cut right after ``at``.

        This graph is left as it is. ``at``'s run ends at ``at``; what came
        after it is gone, and so is what that caused (runs launched, messages
        sent, results delivered). If ``at``'s code launched runs or sent
        messages, those are gone too and ``at`` is set to run again. Every other
        run is copied as it is, finished or not.

        ``repl="copy"`` gives every run a new REPL, rebuilt by running its
        nodes again, with its own workspace and session. ``repl="same"``
        keeps the runs' REPLs, workspaces and sessions, so the copy carries on
        where this graph is; runs the cut rewound still get new ones. Don't
        run this graph and a ``"same"`` copy at once.
        """
        if at.id not in self._nodes:
            raise GraphError(f"cannot fork at {at!r}: it is not in this graph")
        if repl not in ("copy", "same"):
            raise ValueError(f'repl must be "copy" or "same", not {repl!r}')
        branch = self.copy()
        branch.id = new_id("graph")
        branch.metadata["forked_from"] = {"graph": self.id, "node": at.id}
        cut = edits.fork(branch, branch.node(at.id))
        for root in branch.agent_roots():
            if repl == "copy" or root.id in cut.rewound:
                root.repl_id = new_id("repl")
        return branch

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
