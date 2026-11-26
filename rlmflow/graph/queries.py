"""Questions about a graph that only read it: usage, unseen results, readiness, workspace snapshots."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import DONE, RETURN, AgentRoot, LLMUsage, Node

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph


def unseen_results(g: Graph, run: AgentRoot) -> list[tuple[AgentRoot, Node]]:
    """``(child, final node)`` for each finished child run whose result reached no node yet."""
    found = []
    for child in run.children():
        if child.status() == "done":
            final = child.last_turn()
            if final is not None and not g.edges(kind=RETURN, source=final.id):
                found.append((child, final))
    return found


def has_work(g: Graph, run: AgentRoot) -> bool:
    """Whether the graph holds something for ``run`` to do.

    A running run has work. So does an idle one whose finished children's
    results reached no node yet: they don't change its status until they are
    attached.
    """
    status = run.status()
    if status == "running":
        return True
    if status == "idle":
        return all(node.state == DONE for node in run.nodes()) and bool(unseen_results(g, run))
    return False


def ready_runs(g: Graph, busy: Collection[str] = ()) -> list[AgentRoot]:
    """Runs with work whose ids are not in ``busy`` (already taking a step)."""
    return [run for run in g.agent_roots() if run.id not in busy and has_work(g, run)]


def usage(g: Graph, runs: Iterable[AgentRoot] | None = None) -> LLMUsage:
    """Model usage recorded on ``runs``' own nodes, or anywhere in the graph."""
    nodes = g.nodes() if runs is None else [node for run in runs for node in run.nodes()]
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


__all__ = ["has_work", "last_workspace", "ready_runs", "unseen_results", "usage"]
