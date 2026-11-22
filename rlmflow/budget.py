"""What one ``run_streaming`` may spend: tokens and dollars across the whole graph, and seconds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlmflow.graph.nodes import FINAL_KEY, Node
from rlmflow.hooks import Stop

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph
    from rlmflow.graph.nodes import Agent, Step

#: The share of a time budget after which agents are asked for their last turn.
FINAL_TURN_AT = 0.8

BUDGET_FINAL_TEXT = """\
This run has nearly used its budget, so this is your last turn. Give the best \
supported final result now and don't start new work."""

LAUNCH_REFUSED = "cannot start an agent: the run's budget is spent"


@dataclass(frozen=True)
class Budget:
    """A limit on one stream, enforced as a hook.

    Near the limit, launches fail and every unfinished agent gets one last turn
    to give its best answer. At the limit, a token or dollar budget stops each
    agent after that turn; a time budget stops whatever is still running.

    ``usd`` counts the dollars harnesses report (Claude Code does; REPL model
    calls report only tokens). Spend lands when a turn ends, so turns already
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

    def before_turn(self, flow: Flow, agent: Agent) -> Node | Stop | None:
        if _final_turn_taken(agent):
            return Stop("budget spent")
        if self.nearly_spent(flow, agent.graph) and not _final_pending(agent):
            return Node("user", BUDGET_FINAL_TEXT, producer="rlmflow", data={FINAL_KEY: True})
        return None

    def before_launch(self, flow: Flow, step: Step, task: str, config: str) -> str | None:
        return LAUNCH_REFUSED if self.spent(flow, step.graph) else None

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
        for root in g.roots():
            if root.status() != "done":
                await flow.stop(g, root, reason="time budget spent")


def _final_pending(agent: Agent) -> bool:
    """A final-turn note was sent since the agent last finished or went idle."""
    for node in reversed(agent.nodes()):
        if node.final:
            return True
        if node.terminal or node.idle:
            return False
    return False


def _final_turn_taken(agent: Agent) -> bool:
    """The agent took a turn after its final-turn note and still didn't finish."""
    took_turn = False
    for node in reversed(agent.nodes()):
        if node.final:
            return took_turn
        if node.terminal or node.idle:
            return False
        if node.ends_turn:
            took_turn = True
    return False


__all__ = ["BUDGET_FINAL_TEXT", "FINAL_TURN_AT", "LAUNCH_REFUSED", "Budget"]
