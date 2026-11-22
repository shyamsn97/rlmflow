"""Questions about a graph that only read it: usage, unseen results, readiness, workspace snapshots."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import DONE, RETURN, Agent, LLMUsage, Node

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


def has_work(g: Graph, agent: Agent) -> bool:
    """Whether the graph holds something for ``agent`` to do.

    A running agent has work. So does an idle one whose finished children's
    results reached no node yet: they don't change its status until they are
    attached.
    """
    status = agent.status()
    if status == "running":
        return True
    if status == "idle":
        return all(node.state == DONE for node in agent.nodes()) and bool(unseen_results(g, agent))
    return False


def ready_agents(g: Graph, busy: Collection[str] = ()) -> list[Agent]:
    """Agents with work whose ids are not in ``busy`` (already taking a turn)."""
    return [agent for agent in g.agents() if agent.id not in busy and has_work(g, agent)]


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


def last_workspace(nodes: Iterable[Node]) -> Any:
    """The latest workspace snapshot among ``nodes``, if any has one."""
    return next((node.workspace for node in reversed(list(nodes)) if node.workspace), None)


__all__ = ["has_work", "last_workspace", "ready_agents", "unseen_results", "usage"]
