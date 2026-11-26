"""Hooks that come with rlmflow: ``Budget`` (what one stream may spend) and ``Checkpoint`` (saving it as it runs)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import FINAL_KEY, Node
from rlmflow.hooks import Stop

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph

#: The share of a time budget after which agents are asked for their last turn.
FINAL_TURN_AT = 0.8

BUDGET_FINAL_TEXT = """\
This run has nearly used its budget, so this is your last turn. Give the best \
supported final result now and don't start new work."""

LAUNCH_REFUSED = "cannot start an agent: the run's budget is spent"


@dataclass(frozen=True)
class Budget:
    """A limit on one stream, enforced as a hook.

    Near the limit, launches fail and every unfinished run gets one last step
    to give its best answer. At the limit, a token or dollar budget stops each
    run after that step; a time budget stops whatever is still running.

    ``usd`` counts the dollars harnesses report (Claude Code does; REPL model
    calls report only tokens). Spend lands when a step ends, so steps already
    running can go over.
    """

    tokens: int | None = None
    seconds: float | None = None
    usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("tokens", "seconds", "usd"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Budget.{name} must be positive")

    # -- hook --------------------------------------------------------------------

    def start(self, flow: Flow, g: Graph) -> None:
        if self.seconds is not None:
            flow.call_later(g, FINAL_TURN_AT * self.seconds, lambda: None)
            flow.call_later(g, self.seconds, lambda: self._time_up(flow, g))

    def before_step(self, flow: Flow, run: Node) -> Node | Stop | None:
        if _final_turn_taken(run):
            return Stop("budget spent")
        if self.nearly_spent(flow, run._graph()) and not _final_pending(run):
            return Node("user", BUDGET_FINAL_TEXT, producer="rlmflow", data={FINAL_KEY: True})
        return None

    def before_launch(self, flow: Flow, node: Node, task: str, agent: str) -> str | None:
        return LAUNCH_REFUSED if self.spent(flow, node._graph()) else None

    # -- accounting ----------------------------------------------------------------

    def spent(self, flow: Flow, g: Graph) -> bool:
        """Whether ``g``'s stream used up the budget: no more launches, no turns past the final one."""
        usage = g.usage()
        if self.tokens is not None and usage.total >= self.tokens:
            return True
        if self.usd is not None and usage.cost_usd >= self.usd:
            return True
        return self.seconds is not None and flow.elapsed(g) >= self.seconds

    def nearly_spent(self, flow: Flow, g: Graph) -> bool:
        """Whether agents should be asked for their last turn now."""
        if self.spent(flow, g):
            return True
        if self.seconds is not None and flow.elapsed(g) >= FINAL_TURN_AT * self.seconds:
            return True
        nodes = g.nodes()
        usage = g.usage()
        # Another turn as costly as the average one so far would reach the limit.
        for limit, spent, key in (
            (self.tokens, usage.total, "usage"),
            (self.usd, usage.cost_usd, "cost_usd"),
        ):
            turns = sum(1 for node in nodes if node.data.get(key))
            if limit is not None and turns > 0 and spent + spent / turns >= limit:
                return True
        return False

    async def _time_up(self, flow: Flow, g: Graph) -> None:
        for root in g.top_level_roots():
            if root.status() != "done":
                await flow.stop(root, reason="time budget spent")


def _final_pending(run: Node) -> bool:
    """A final-step note was sent since the run last finished or went idle."""
    for node in reversed(run.nodes()):
        if node.final:
            return True
        if node.terminal or node.idle:
            return False
    return False


def _final_turn_taken(run: Node) -> bool:
    """The run took a step after its final-step note and still didn't finish."""
    took_turn = False
    for node in reversed(run.nodes()):
        if node.final:
            return took_turn
        if node.terminal or node.idle:
            return False
        if node.ends_turn:
            took_turn = True
    return False


class Checkpoint:
    """Save one streamed graph every so often, on every terminal node, and when the stream ends.

    Pass it as a hook, or call it on each node yourself and ``close()`` it after.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        interval_s: float | None = 2.0,
        interval_nodes: int | None = 50,
    ) -> None:
        if interval_s is not None and interval_s <= 0:
            raise ValueError("interval_s must be positive or None")
        if interval_nodes is not None and interval_nodes <= 0:
            raise ValueError("interval_nodes must be positive or None")
        self.path = Path(path)
        self.interval_s = interval_s
        self.interval_nodes = interval_nodes
        self.last_flush_at = time.monotonic()
        self.latest: Graph | None = None
        self.dirty_nodes = 0

    def __call__(self, node: Node) -> None:
        graph = node._graph()
        if self.latest is not None and graph is not self.latest:
            raise ValueError("one Checkpoint cannot write multiple graphs")
        self.latest = graph
        self.dirty_nodes += 1
        if node.terminal or self._due():
            self.flush()

    def on_node(self, flow: Any, node: Node) -> None:
        self(node)

    def end(self, flow: Any, g: Graph) -> None:
        self.latest = self.latest or g
        self.close()

    def _due(self) -> bool:
        if self.interval_nodes is not None and self.dirty_nodes >= self.interval_nodes:
            return True
        return (
            self.interval_s is not None and time.monotonic() - self.last_flush_at >= self.interval_s
        )

    def flush(self) -> None:
        graph = self.latest
        if graph is None:
            return
        graph.save(self.path)
        self.dirty_nodes = 0
        self.last_flush_at = time.monotonic()

    def close(self) -> None:
        self.flush()


__all__ = ["BUDGET_FINAL_TEXT", "FINAL_TURN_AT", "LAUNCH_REFUSED", "Budget", "Checkpoint"]
