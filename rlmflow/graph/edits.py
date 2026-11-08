"""Edits: change a node, delete it, remove an agent, or replace a saved call.

An edit rewinds the agent that owns the node to it, then follows the node's
outgoing edges. ``RULES`` says, for each engine edge kind, what happens to the
edge's target when its source is dropped and when its source changes:

- ``remove_root``: the target agent is removed.
- ``restart_root``: the target fork starts over from its base.
- ``rerun_target``: the target runs again (a node with no code is taken back,
  so its agent gets the new value).
- ``retract_target``: the target is taken back from its agent's history.
- ``match_calls``: nothing yet; the source runs again, and saved calls it
  doesn't make again are undone when it finishes.

Nothing here knows about running agents: a ``Flow`` cancels the affected
agents' tasks before it applies an edit, and node versions discard the result
of any run that started before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rlmflow.graph import calls
from rlmflow.graph.nodes import (
    FORK,
    IMAGES_KEY,
    READY,
    RETURN,
    SPAWN,
    WRITE,
    Node,
    strict_json,
)

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph

REMOVE_ROOT = "remove_root"
RESTART_ROOT = "restart_root"
RERUN_TARGET = "rerun_target"
RETRACT_TARGET = "retract_target"
MATCH_CALLS = "match_calls"

#: ``kind: (when the source is dropped, when the source changes)``.
RULES: dict[str, tuple[str, str]] = {
    SPAWN: (REMOVE_ROOT, MATCH_CALLS),
    FORK: (REMOVE_ROOT, RESTART_ROOT),
    RETURN: (RERUN_TARGET, RERUN_TARGET),
    WRITE: (RETRACT_TARGET, MATCH_CALLS),
}

#: What a node's run wrote into ``data``; a node set to run again loses them.
RUN_DATA = ("dead", IMAGES_KEY, "workspace", "terminal", "idle")

#: The fields ``edit`` changes.
EDITABLE = ("text", "input", "code", "result")

MISSING: Any = object()


class GraphError(Exception):
    """A write that would leave the graph inconsistent."""


@dataclass
class Invalidation:
    """What an edit changed: agents rewound or removed, nodes dropped, nodes set to run again."""

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


class _Edit:
    """One edit in progress: the graph, and what it has invalidated so far."""

    def __init__(self, g: Graph) -> None:
        self.g = g
        self.inv = Invalidation()

    def follow(self, node: Node, *, changed: bool) -> None:
        """Apply ``RULES`` to the edges out of ``node``, which was dropped or changed."""
        g = self.g
        for edge in g.edges(source=node.id):
            target = g._nodes.get(edge.target)
            rule = RULES.get(edge.kind)
            if rule is None:
                if not changed:
                    g._unlink(edge)
                continue
            action = rule[1 if changed else 0]
            if not changed or action == RERUN_TARGET:
                g._unlink(edge)
            if target is None:
                continue
            if action == REMOVE_ROOT:
                self.remove_root(target)
            elif action == RESTART_ROOT:
                self.cut(target, 1)
            elif action == RERUN_TARGET:
                self.rerun(target) if target.runs_code else self.retract(target)
            elif action == RETRACT_TARGET:
                self.retract(target)

    def cut(self, root: Node, keep: int) -> None:
        """Keep ``root``'s first ``keep`` own nodes; drop the rest and what they caused."""
        history = root.root_data.history
        dropped = history[keep:]
        if not dropped:
            return
        del history[keep:]
        self.inv._add("rewound", root.id)
        for node in reversed(dropped):
            self.drop(node)

    def drop(self, node: Node) -> None:
        if self.g._nodes.pop(node.id, None) is None:
            return
        self.inv._add("dropped", node.id)
        self.follow(node, changed=False)
        for edge in self.g.edges(target=node.id):
            self.g._unlink(edge)

    def retract(self, node: Node) -> None:
        """Take a node back: rewind its agent to before it."""
        root = self.g._roots.get(node.owner)
        if root is not None and node in root.root_data.history:
            self.cut(root, root.root_data.history.index(node))

    def rerun(self, node: Node) -> None:
        """Rewind the node's agent to it and set it to run again."""
        root = self.g._roots[node.owner]
        for edge in self.g.edges(kind=RETURN, target=node.id):
            self.g._unlink(edge)
        self.follow(node, changed=True)
        self.cut(root, root.root_data.history.index(node) + 1)
        node.output, node.result, node.error, node.state = "", None, False, READY
        for key in RUN_DATA:
            node.data.pop(key, None)
        node.version += 1
        self.inv._add("rewound", root.id)
        self.inv._add("rerun", node.id)

    def remove_root(self, root: Node) -> None:
        g = self.g
        if root.id not in g._roots:
            return
        self.cut(root, 1)
        self.drop(root)
        del g._roots[root.id]
        g._children.pop(root.id, None)
        self.inv._add("removed", root.id)
        if root.id in self.inv.rewound:
            self.inv.rewound.remove(root.id)


def edit(g: Graph, node: Node, **fields: Any) -> Invalidation:
    """Change ``node``. Its agent is rewound to it; changed code or inputs run again."""
    unknown = set(fields) - set(EDITABLE)
    if unknown:
        raise TypeError(f"edit takes {', '.join(EDITABLE)}; not {', '.join(sorted(unknown))}")
    if "input" in fields:
        fields["input"] = strict_json(fields["input"] or {}, "node input")
    if "result" in fields:
        fields["result"] = strict_json(fields["result"], "node result")
    for name in ("text", "code"):
        if name in fields and not isinstance(fields[name], str):
            raise TypeError(f"node {name} must be a string")
    for name, value in fields.items():
        setattr(node, name, value)
    if node.root_data is not None and "text" in fields:
        node.root_data.task = node.text
    node.version += 1
    change = _Edit(g)
    if node.runs_code and ("code" in fields or "input" in fields):
        change.rerun(node)
    else:
        change.follow(node, changed=True)
        root = g._roots[node.owner]
        change.cut(root, root.root_data.history.index(node) + 1)
        change.inv._add("rewound", root.id)
    return change.inv


def delete(g: Graph, node: Node) -> Invalidation:
    """Remove a node and everything after it; deleting a root removes its agent."""
    change = _Edit(g)
    if node.root_data is not None:
        change.remove_root(node)
    else:
        change.retract(node)
    return change.inv


def remove(g: Graph, agent: Node) -> Invalidation:
    """Remove an agent, the agents it launched, and what they caused."""
    change = _Edit(g)
    change.remove_root(g.agent(agent))
    return change.inv


def replace(
    g: Graph, node: Node, index: int, value: Any = MISSING, *, agent: Node | None = None
) -> Invalidation:
    """Bind the node's ``index``-th saved call to ``value`` or another ``agent``; the node reruns.

    A launch takes another agent (the old child is removed). An ``await`` takes
    a value or the agent to await instead. A tool call takes a value.
    """
    if (value is MISSING) == (agent is None):
        raise TypeError("replace takes exactly one of a value or agent=")
    if not 0 <= index < len(node.calls):
        raise GraphError(f"node {node.id!r} saved no call {index}")
    entry = node.calls[index]
    tool, old = entry["tool"], g._roots.get(entry.get("agent", ""))
    change = _Edit(g)
    if tool == calls.LAUNCH:
        if agent is None:
            raise GraphError("a launch can only be replaced by another agent")
        for edge in g.edges(kind=SPAWN, target=agent.id):
            g._unlink(edge)
        g.link(SPAWN, node, agent)
        if old is not None and old is not agent:
            calls.rebind(node, old.id, agent.id)
            change.remove_root(old)
        entry["agent"] = agent.id
        change.rerun(node)
        return change.inv
    if agent is not None and tool != calls.AWAIT:
        raise GraphError(f"a {tool} call can only be replaced by a value")
    replaced = [
        edge.source
        for edge in g.edges(kind=RETURN, target=node.id)
        if old is not None and g._nodes[edge.source].owner == old.id
    ]
    call_key = entry.get("key")
    bound = (
        {"agent": agent.id}
        if agent is not None
        else {"value": strict_json(value, "replacement value")}
    )
    entry.clear()
    entry.update(tool=tool, key=call_key, **bound)
    change.rerun(node)
    # The old agent's result reached this node and was swapped out, so it isn't
    # delivered again as a result node.
    for source in replaced:
        g.link(RETURN, source, node, replaced=True)
    return change.inv


__all__ = [
    "EDITABLE",
    "MATCH_CALLS",
    "REMOVE_ROOT",
    "RERUN_TARGET",
    "RESTART_ROOT",
    "RETRACT_TARGET",
    "RULES",
    "GraphError",
    "Invalidation",
    "delete",
    "edit",
    "remove",
    "replace",
]
