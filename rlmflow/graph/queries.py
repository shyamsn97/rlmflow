"""Questions about a graph that only read it: usage and unseen results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from rlmflow.graph.nodes import RETURN, Agent, LLMUsage, Node

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph


def unseen_results(g: Graph, agent: Agent) -> list[tuple[Agent, Node]]:
    """``(child, final node)`` for each finished child whose result reached no node yet."""
    found = []
    for child in agent.children():
        if child.status() == "done":
            final = child.last_turn()
            if final is not None and not g.edges(kind=RETURN, source=final.id):
                found.append((child, final))
    return found


def usage(g: Graph, agents: Iterable[Agent] | None = None) -> LLMUsage:
    """Model usage recorded on ``agents``' own nodes, or anywhere in the graph."""
    nodes = g.nodes() if agents is None else [node for agent in agents for node in agent.nodes()]
    total = LLMUsage()
    for node in nodes:
        if node.data.get("usage"):
            total += LLMUsage.of(node.data["usage"])
        if node.data.get("cost_usd"):
            total.cost_usd += float(node.data["cost_usd"])
    return total


__all__ = ["unseen_results", "usage"]
