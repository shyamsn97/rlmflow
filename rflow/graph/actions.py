"""Action values — the *intent* half of one advance.

The engine advances an agent by **one observation-to-observation step**: the
agent rests at an observation, the engine decides what to do next, runs it, and
records the resulting observation. This module holds the pure "decide" output —
small value objects naming the transition to take — kept separate from the
side-effectful handlers on :class:`~rflow.engine.Flow` so the policy that picks
them (:meth:`Flow.plan`) stays pure, inspectable, and overridable.

Each action carries only intent not recoverable from the graph; the persisted
:class:`~rflow.graph.ActionNode` records what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CallLLM:
    """Call the LLM. Materialized as ``LLMAction -> LLMOutput``.

    ``force_final`` carries the policy decision to force a terminal answer this
    turn (iteration budget exhausted).
    """

    agent_id: str
    force_final: bool = False


@dataclass(frozen=True, slots=True)
class Exec:
    """Run the code from the preceding ``LLMOutput``. ``ExecAction -> CodeObservation``."""

    agent_id: str


@dataclass(frozen=True, slots=True)
class Recover:
    """Inject a recovery observation for a stranded supervisor."""

    agent_id: str
    launch_id: str


Action = CallLLM | Exec | Recover
ActionPlan = dict[str, Action]


__all__ = ["Action", "ActionPlan", "CallLLM", "Exec", "Recover"]
