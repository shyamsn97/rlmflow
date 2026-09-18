"""Canonical read-only views over stable benchmark row payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.eval.types import Prediction, Row

INFRA_ERROR_MARKERS = (
    "APIConnectionError",
    "ConnectionError",
    "RateLimit",
    "Timeout",
    "Traceback",
    "budget exceeded",
)


def is_infra_error(value: object) -> bool:
    text = str(value or "")
    return any(marker in text for marker in INFRA_ERROR_MARKERS)


def expected_from_row(row: Row) -> Any:
    return row.score.details.get("expected", row.metadata.get("expected"))


def canonical_delegation_metrics(
    *,
    graph: dict[str, Any],
    runtime_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Normalize backend telemetry once at the row boundary."""
    recursive = int(runtime_metrics.get("recursive_subcalls", 0))
    children = int(runtime_metrics.get("subcalls", recursive))
    return {
        "children": children,
        "dependency_edge_count": 0,
        "dependency_chain_depth": 0,
        "recursive_launch_count": recursive,
        "retrieved_child_results": recursive,
        "retrieval_ratio": float(recursive > 0),
        "max_launch_batch": int(graph.get("max_branching", 0)),
        "max_ready_launch_batch": int(graph.get("max_branching", 0)),
        "concurrent_launch_batches": 0,
        "query_calls": max(0, children - recursive),
    }


@dataclass(frozen=True, slots=True)
class EnvironmentMetricsView:
    kind: str
    primary: dict[str, Any]
    candidate_search: dict[str, Any]
    accepted_actions: int
    primary_actions: int
    probe_actions: int
    instances_forked: int

    @classmethod
    def from_prediction(cls, prediction: Prediction) -> EnvironmentMetricsView:
        raw = prediction.metrics.get("environment", {})
        environment = raw if isinstance(raw, dict) else {}
        raw_primary = environment.get("primary", environment)
        primary = raw_primary if isinstance(raw_primary, dict) else {}
        raw_search = environment.get("candidate_search", {})
        search = raw_search if isinstance(raw_search, dict) else {}
        accepted = environment.get("total_actions", environment.get("accepted_actions"))
        if not isinstance(accepted, (int, float)):
            accepted = len(environment.get("actions", []))
        return cls(
            kind=str(environment.get("kind", "")),
            primary=dict(primary),
            candidate_search=dict(search),
            accepted_actions=max(0, int(accepted or 0)),
            primary_actions=max(0, int(environment.get("primary_actions", 0) or 0)),
            probe_actions=max(0, int(environment.get("probe_actions", 0) or 0)),
            instances_forked=max(0, int(environment.get("instances_forked", 0) or 0)),
        )

    @property
    def selection_required(self) -> bool:
        return bool(self.candidate_search.get("selection_required"))

    @property
    def selection_made(self) -> bool:
        return bool(
            self.candidate_search.get("selection_made")
            or self.candidate_search.get("selected_candidate_id")
        )

    @property
    def selection_compliance(self) -> float | None:
        return float(self.selection_made) if self.selection_required else None

    def native_float(self, *keys: str, bounded: bool = False) -> float:
        raw = next((self.primary[key] for key in keys if key in self.primary), 0.0)
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0.0
        return min(1.0, max(0.0, value)) if bounded else value

    def common_details(self) -> dict[str, Any]:
        return {
            "accepted_actions": self.accepted_actions,
            "primary_actions": self.primary_actions,
            "probe_actions": self.probe_actions,
            "instances_forked": self.instances_forked,
            "selected_candidate_id": self.candidate_search.get("selected_candidate_id"),
            "selection_regret": self.candidate_search.get("selection_regret"),
            "selection_required": self.selection_required,
            "selection_made": self.selection_made,
            "selection_compliance": (
                float(self.selection_made) if self.selection_required else 1.0
            ),
        }


@dataclass(frozen=True, slots=True)
class RowTelemetry:
    runner: str
    score: float
    correct: bool | None
    error: str | None
    infra_error: bool
    input_tokens: int
    output_tokens: int
    time_seconds: float
    graph: dict[str, Any]
    delegation: dict[str, Any]
    environment: EnvironmentMetricsView
    artifacts: dict[str, str]

    @classmethod
    def from_row(cls, row: Row) -> RowTelemetry:
        metrics = row.prediction.metrics
        graph = dict(metrics.get("graph", {}))
        delegation = dict(metrics.get("delegation", {}))
        if not delegation and isinstance(metrics.get("rlm"), dict):
            delegation = canonical_delegation_metrics(
                graph=graph,
                runtime_metrics=dict(metrics["rlm"]),
            )
        error = row.prediction.error or row.score.details.get("error")
        return cls(
            runner=row.runner,
            score=float(row.score.value),
            correct=row.score.correct,
            error=str(error) if error else None,
            infra_error=is_infra_error(error) or is_infra_error(row.prediction.answer),
            input_tokens=int(
                row.metadata.get(
                    "aggregate_input_tokens",
                    row.prediction.usage.get("input_tokens", 0),
                )
            ),
            output_tokens=int(
                row.metadata.get(
                    "aggregate_output_tokens",
                    row.prediction.usage.get("output_tokens", 0),
                )
            ),
            time_seconds=float(
                row.metadata.get(
                    "oracle_wall_time_seconds",
                    metrics.get("time_seconds", 0.0),
                )
            ),
            graph=graph,
            delegation=delegation,
            environment=EnvironmentMetricsView.from_prediction(row.prediction),
            artifacts=dict(row.prediction.artifacts),
        )

    @property
    def topology(self) -> str:
        if self.delegation.get("dependency_edge_count", 0):
            return "dependency-dag"
        if self.delegation.get("children", 0):
            return "delegation-tree"
        return "single-agent"


__all__ = [
    "EnvironmentMetricsView",
    "INFRA_ERROR_MARKERS",
    "RowTelemetry",
    "canonical_delegation_metrics",
    "expected_from_row",
    "is_infra_error",
]
