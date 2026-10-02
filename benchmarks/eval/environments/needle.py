"""Needle-haystack benchmark environments."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from benchmarks.eval.environments.environment import (
    Environment,
    Observation,
)
from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics
from benchmarks.eval.tasks.synthetic_needle import SyntheticNeedleDataset, _normalize
from benchmarks.eval.types import Example, Prediction, Score


@dataclass(frozen=True, slots=True)
class NeedleObservation(Observation):
    haystack: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {"haystack": self.haystack}


class NeedleEnvironment(Environment[NeedleObservation, dict]):
    def __init__(self, example: Example) -> None:
        self.id = example.id
        self.description = example.prompt
        self._observation = NeedleObservation(haystack=str(example.context["haystack"]))
        self._expected = str(example.expected)
        self._metadata = {
            key: value
            for key, value in example.metadata.items()
            if key not in {"needle_index", "expected"}
        }

    def start(self, instance_id: str) -> NeedleObservation:
        del instance_id
        return self._observation

    def metrics(self, instance_id: str) -> dict:
        del instance_id
        return {
            "records": self._metadata["records"],
            "context_chars": self._metadata["context_chars"],
        }

    def report(
        self,
        prediction: Prediction,
        runtime: RuntimeMetrics,
    ) -> EnvironmentReport[dict]:
        normalized = _normalize(prediction.answer)
        correct = normalized == self._expected
        score = Score(
            value=float(correct),
            correct=correct,
            details={"normalized_answer": normalized},
        )
        return EnvironmentReport(
            benchmark="",
            environment_id=self.id,
            runner="",
            model="",
            seed=None,
            answer=prediction.answer,
            score=score,
            description_sha256="",
            observation_sha256="",
            runtime=runtime,
            environment=self.metrics(""),
            artifacts=prediction.artifacts,
            error=prediction.error,
        )


def needle_environments(*, split: str, limit: int | None, seed: int) -> list[NeedleEnvironment]:
    examples = SyntheticNeedleDataset().examples(split=split, limit=limit, seed=seed)
    return [NeedleEnvironment(example) for example in examples]


__all__ = ["NeedleEnvironment", "NeedleObservation", "needle_environments"]
