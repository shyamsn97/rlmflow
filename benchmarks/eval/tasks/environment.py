"""Shared scoring primitives for environment-backed tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.eval.telemetry import EnvironmentMetricsView
from benchmarks.eval.types import Prediction


@dataclass(frozen=True, slots=True)
class EnvironmentGrade:
    view: EnvironmentMetricsView
    native_value: float
    value: float
    selected: bool

    def details(self, *, error: str | None, **domain: Any) -> dict[str, Any]:
        return {
            **domain,
            **self.view.common_details(),
            "native_progress_score": self.native_value,
            "error": error,
        }


def grade_environment(
    prediction: Prediction,
    *,
    native_keys: tuple[str, ...],
    bounded: bool = True,
) -> EnvironmentGrade:
    view = EnvironmentMetricsView.from_prediction(prediction)
    native = view.native_float(*native_keys, bounded=bounded)
    selected = not view.selection_required or view.selection_made
    return EnvironmentGrade(
        view=view,
        native_value=native,
        value=native if selected else 0.0,
        selected=selected,
    )


__all__ = ["EnvironmentGrade", "grade_environment"]
