"""Outcome-gated scoring for frozen human or judge trajectory labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmarks.eval.delegation.annotations import TaskAnnotation
from rlmflow import AgentStart


@dataclass(frozen=True)
class ChildAlignment:
    agent_id: str
    obligations: tuple[str, ...]
    result_used: bool
    redundant: bool = False


def score_alignment(
    root: AgentStart,
    annotation: TaskAnnotation,
    labels: Iterable[ChildAlignment],
    *,
    outcome: float,
) -> dict[str, Any]:
    """Score externally labeled semantics without rewarding raw child count."""

    children = {agent.id: agent for agent in list(root.iter_agents())[1:]}
    labels = list(labels)
    unknown_agents = {label.agent_id for label in labels} - children.keys()
    if unknown_agents:
        raise ValueError(f"alignment labels reference unknown agents: {sorted(unknown_agents)}")

    obligation_ids = {obligation.id for obligation in annotation.obligations}
    covered = {
        obligation
        for label in labels
        for obligation in label.obligations
        if obligation in obligation_ids
    }
    unknown_obligations = {
        obligation
        for label in labels
        for obligation in label.obligations
        if obligation not in obligation_ids
    }
    if unknown_obligations:
        raise ValueError(
            f"alignment labels reference unknown obligations: {sorted(unknown_obligations)}"
        )

    useful = [label for label in labels if label.obligations and label.result_used]
    redundant = [label for label in labels if label.redundant]
    coverage = len(covered) / len(obligation_ids) if obligation_ids else 1.0
    precision = len(useful) / len(labels) if labels else float(not obligation_ids)
    utilization = sum(label.result_used for label in labels) / len(labels) if labels else 1.0
    local_restraint = float(annotation.preferred_local and not children)
    trajectory = coverage * precision * utilization
    if annotation.preferred_local and not children:
        trajectory = 1.0

    return {
        "outcome": outcome,
        "coverage": coverage,
        "precision": precision,
        "result_utilization": utilization,
        "redundant_children": len(redundant),
        "local_restraint": local_restraint,
        "trajectory": trajectory,
        "outcome_gated_trajectory": outcome * trajectory,
    }


__all__ = ["ChildAlignment", "score_alignment"]
