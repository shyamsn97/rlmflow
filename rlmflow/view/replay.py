"""Walk a finished run one step at a time.

A saved run holds every node with the stamp of when it was made, so the run can
be replayed by handing back the tree as it stood at each step. ``replay`` yields
those snapshots as real trees, which means anything that reads a tree — the ASCII
renderer, the SVG figure, your own code — works on a replay without knowing it is
one.

Each snapshot is rebuilt rather than mutated in place, so they can be collected,
compared, or rendered out of order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import copy
from pathlib import Path

from rlmflow.consumers.ui import render_tree
from rlmflow.graph.nodes import AgentStart, Node, _rebuild_index
from rlmflow.view.steps import timeline


def as_root(source: AgentStart | str | Path) -> AgentStart:
    """A root from a graph, a checkpoint directory, or a path to one."""
    if isinstance(source, AgentStart):
        return source
    from rlmflow.graph import persistence

    return persistence.load(source)


def snapshot(nodes: Sequence[Node]) -> AgentStart:
    """Rebuild ``nodes`` into a standalone tree.

    The nodes are copied shallowly and relinked, so a snapshot shares content with
    the run it came from but not its shape: growing one leaves the original alone.
    ``nodes`` must start at a root and hold every node's parent before the node.
    """
    if not nodes:
        raise ValueError("a snapshot needs at least a root")
    if not isinstance(nodes[0], AgentStart):
        raise ValueError(f"a snapshot has to start at an agent, not {nodes[0].type}")

    clones: dict[str, Node] = {}
    root: AgentStart | None = None
    for node in nodes:
        clone = copy(node)
        clone.children = []
        if isinstance(clone, AgentStart):
            clone.sub_agents = []
            clone.frontier = clone
        clones[node.id] = clone

        if root is None:
            root = clone  # type: ignore[assignment]
        clone.root = root

        parent = clones.get(node.parent.id) if node.parent is not None else None
        if parent is not None:
            clone.parent = parent
            parent.children.append(clone)
        elif clone is not root:
            clone.parent = None

        if isinstance(clone, AgentStart):
            clone.parent_agent = clone
            if parent is not None and isinstance(parent.parent_agent, AgentStart):
                parent.parent_agent.sub_agents.append(clone)
        elif node.parent_agent is not None:
            agent = clones.get(node.parent_agent.id)
            if isinstance(agent, AgentStart):
                clone.parent_agent = agent
                agent.frontier = clone

    assert root is not None
    _rebuild_index(root)
    return root


def replay(source: AgentStart | str | Path) -> Iterator[AgentStart]:
    """The run as a tree per step, from the root alone to the whole thing.

    ::

        for snap in replay("runs/coding/graph"):
            print(render_tree(snap))
    """
    ordered = timeline(as_root(source))
    for cut in range(1, len(ordered) + 1):
        yield snapshot(ordered[:cut])


def render_steps(source: AgentStart | str | Path) -> Iterator[str]:
    """Each step as the ASCII tree, ready to print."""
    for snap in replay(source):
        yield render_tree(snap)
