"""Questions about a graph that only read it: family, paths, usage, unseen results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from rlmflow.graph.nodes import RETURN, LLMUsage, Node

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph


def subtree(g: Graph, agent: Node) -> list[Node]:
    """``agent`` and every agent it launched, transitively, depth first."""
    found: list[Node] = []
    stack = [agent]
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(reversed(g.children(current)))
    return found


def lineage(g: Graph, agent: Node) -> list[Node]:
    """Its top-level ancestor first, ending with ``agent``."""
    chain = [agent]
    while (parent := g.parent(chain[-1])) is not None and parent not in chain:
        chain.append(parent)
    return chain[::-1]


def path(g: Graph, agent: Node) -> str:
    """Agent names from the top, joined with dots: ``root.worker-0``."""
    return ".".join(item.root_data.name for item in lineage(g, agent))


def unseen_results(g: Graph, agent: Node) -> list[tuple[Node, Node]]:
    """``(child, final node)`` for each finished child whose result reached no node yet."""
    found = []
    for child in g.children(agent):
        if g.status(child) == "done":
            final = g.last_turn(child)
            if final is not None and not g.edges(kind=RETURN, source=final.id):
                found.append((child, final))
    return found


def usage(g: Graph, agents: Iterable[Node] | None = None) -> LLMUsage:
    """Model usage recorded on ``agents``' own nodes, or anywhere in the graph."""
    nodes = (
        g.nodes()
        if agents is None
        else [node for agent in agents for node in agent.root_data.history]
    )
    total = LLMUsage()
    for node in nodes:
        if node.data.get("usage"):
            total += LLMUsage.of(node.data["usage"])
        if node.data.get("cost_usd"):
            total.cost_usd += float(node.data["cost_usd"])
    return total


__all__ = ["lineage", "path", "subtree", "unseen_results", "usage"]
