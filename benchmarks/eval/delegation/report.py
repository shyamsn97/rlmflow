"""Aggregation helpers that preserve problem-level delegation results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from benchmarks.eval.types import Row


def delegation_report(rows: Iterable[Row]) -> dict[str, Any]:
    rows = list(rows)
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        condition = str(row.prediction.metrics.get("delegation", {}).get("condition", row.runner))
        groups[(condition, row.example_id)].append(row)

    problems = []
    for (condition, example_id), attempts in sorted(groups.items()):
        scores = [attempt.score.value for attempt in attempts]
        delegation = [attempt.prediction.metrics.get("delegation", {}) for attempt in attempts]
        problems.append(
            {
                "condition": condition,
                "example_id": example_id,
                "attempts": len(attempts),
                "mean_score": fmean(scores),
                "pass_at_1": sum(bool(attempt.score.correct) for attempt in attempts)
                / len(attempts),
                "mean_children": fmean(float(metric.get("children", 0)) for metric in delegation),
                "delegation_rate": sum(
                    float(metric.get("children", 0)) > 0 for metric in delegation
                )
                / len(delegation),
                "mean_retrieved_child_results": fmean(
                    float(metric.get("retrieved_child_results", 0)) for metric in delegation
                ),
                "mean_unretrieved_child_results": fmean(
                    float(metric.get("unretrieved_child_results", 0)) for metric in delegation
                ),
                "mean_model_calls": fmean(
                    float(metric.get("model_calls", 0)) for metric in delegation
                ),
                "unfinished_attempts": sum(
                    bool(metric.get("root_finished_with_unfinished_children"))
                    for metric in delegation
                ),
            }
        )

    conditions = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in problems:
        by_condition[problem["condition"]].append(problem)
    for condition, condition_problems in sorted(by_condition.items()):
        conditions.append(
            {
                "condition": condition,
                "problems": len(condition_problems),
                "macro_score": fmean(problem["mean_score"] for problem in condition_problems),
                "macro_pass_at_1": fmean(problem["pass_at_1"] for problem in condition_problems),
                "mean_children": fmean(problem["mean_children"] for problem in condition_problems),
                "delegation_rate": fmean(
                    problem["delegation_rate"] for problem in condition_problems
                ),
                "mean_retrieved_child_results": fmean(
                    problem["mean_retrieved_child_results"] for problem in condition_problems
                ),
                "mean_unretrieved_child_results": fmean(
                    problem["mean_unretrieved_child_results"] for problem in condition_problems
                ),
            }
        )
    return {"rows": len(rows), "conditions": conditions, "problems": problems}


__all__ = ["delegation_report"]
