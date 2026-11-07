"""Questions about a graph that only read it: family, paths, cells, usage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from rlmflow.graph.nodes import CODE, RETURN, Agent, LLMUsage, Node

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph


def subtree(g: Graph, agent: Agent) -> list[Agent]:
    """``agent`` and every agent it launched, transitively, depth first."""
    found: list[Agent] = []
    stack = [agent]
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(reversed(g.children(current)))
    return found


def lineage(g: Graph, agent: Agent) -> list[Agent]:
    """Its top-level ancestor first, ending with ``agent``."""
    chain = [agent]
    while (parent := g.parent(chain[-1])) is not None and parent not in chain:
        chain.append(parent)
    return chain[::-1]


def path(g: Graph, agent: Agent) -> str:
    """Agent names from the top, joined with dots: ``root.worker-0``."""
    return ".".join(item.name for item in lineage(g, agent))


def cells(g: Graph, agent: Agent) -> list[Node]:
    """Every cell in the agent's history that ran code, in order."""
    return [node for node in g.history(agent) if node.type == CODE and node.data.get("code")]


def unseen_results(g: Graph, agent: Agent) -> list[tuple[Agent, Node]]:
    """``(child, done)`` for each finished child whose result no cell awaited and no message carried."""
    found = []
    for child in g.children(agent):
        if g.status(child) == "done":
            done = g.history(child)[-1]
            if not g.edges(kind=RETURN, source=done.id):
                found.append((child, done))
    return found


def usage(g: Graph, agents: Iterable[Agent] | None = None) -> LLMUsage:
    """Model usage recorded on ``agents``' nodes, or anywhere in the graph."""
    nodes = g.nodes() if agents is None else [node for agent in agents for node in agent.nodes]
    total = LLMUsage()
    for node in nodes:
        if node.data.get("usage"):
            total += LLMUsage.of(node.data["usage"])
        if node.data.get("cost_usd"):
            total.cost_usd += float(node.data["cost_usd"])
    return total


__all__ = ["cells", "lineage", "path", "subtree", "unseen_results", "usage"]
