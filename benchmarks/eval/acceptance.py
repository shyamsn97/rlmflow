"""One source of truth for conformance and depth-claim thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median
from typing import Any

from benchmarks.eval.suite import DEPTH_COMPARISON


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    required_runners: frozenset[str] = frozenset(DEPTH_COMPARISON)
    required_domains: frozenset[str] = frozenset({"minimaze", "sudoku_search"})
    required_seeds: frozenset[str] = frozenset({"0", "1", "2"})
    rows_per_runner_domain: int = 6
    minimum_selection_compliance: float = 0.8
    maximum_row_seconds: float = 300.0
    depth0_minimum: float = 0.3
    depth0_maximum: float = 0.7
    maximum_floor_share: float = 0.7
    maximum_ceiling_share: float = 0.7
    baseline_runner: str = "rlmflow-depth-0"
    treatment_runner: str = "rlmflow-depth-2"

    def assess_scores(
        self,
        baseline: list[float],
        treatment: list[float],
        *,
        candidate_variation: bool,
    ) -> dict[str, Any]:
        baseline_median = median(baseline) if baseline else None
        baseline_mean = fmean(baseline) if baseline else None
        treatment_mean = fmean(treatment) if treatment else None
        floor_share = sum(score <= 0 for score in baseline) / len(baseline) if baseline else None
        ceiling_share = (
            sum(score >= 0.999 for score in baseline) / len(baseline) if baseline else None
        )
        target = (
            baseline_median is not None
            and self.depth0_minimum <= baseline_median <= self.depth0_maximum
        )
        return {
            "depth0_median": baseline_median,
            "depth0_nonzero_median": baseline_median is not None and baseline_median > 0,
            "depth0_target_30_70": target,
            "floor_share": floor_share,
            "ceiling_share": ceiling_share,
            "multiple_candidate_outcomes": candidate_variation,
            "depth2_lift_over_depth0": (
                treatment_mean is not None
                and baseline_mean is not None
                and treatment_mean > baseline_mean
            ),
            "claim_delegation_benefit": (
                treatment_mean is not None
                and baseline_mean is not None
                and treatment_mean > baseline_mean
                and candidate_variation
            ),
        }


DEFAULT_ACCEPTANCE_POLICY = AcceptancePolicy()

__all__ = ["AcceptancePolicy", "DEFAULT_ACCEPTANCE_POLICY"]
