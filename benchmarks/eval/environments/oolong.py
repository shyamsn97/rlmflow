"""OOLONG long-context environments."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from benchmarks.eval.environments.environment import Environment, Observation
from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics
from benchmarks.eval.tasks.oolong import OolongTrecCoarse128KDataset, _synth_score
from benchmarks.eval.types import Example, Prediction, Score

SPLIT_SIZES = {"pilot": 2, "full": 25}


@dataclass(frozen=True, slots=True)
class OolongObservation(Observation):
    context: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {"context": self.context}


class OolongEnvironment(Environment[OolongObservation, dict]):
    def __init__(self, example: Example) -> None:
        self.id = example.id
        self.description = example.prompt
        self._observation = OolongObservation(context=str(example.context["context"]))
        self._expected = example.expected
        self._answer_type = str(example.metadata.get("answer_type") or "")
        self._context_len = example.metadata.get("context_len")

    def start(self, instance_id: str) -> OolongObservation:
        del instance_id
        return self._observation

    def metrics(self, instance_id: str) -> dict:
        del instance_id
        return {
            "context_chars": len(self._observation.context),
            "context_len": self._context_len,
            "answer_type": self._answer_type,
        }

    def report(
        self,
        prediction: Prediction,
        runtime: RuntimeMetrics,
    ) -> EnvironmentReport[dict]:
        score = self._score(prediction.answer)
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

    def _score(self, answer: str) -> Score:
        best = max(
            (
                _synth_score(expected, answer, answer_type=self._answer_type)
                for expected in self._expected or []
            ),
            default=0.0,
        )
        return Score(
            value=best,
            correct=best >= 1.0,
            details={"answer_type": self._answer_type},
        )


def oolong_environments(
    *,
    split: str,
    limit: int | None,
    seed: int,
) -> list[OolongEnvironment]:
    try:
        split_size = SPLIT_SIZES[split]
    except KeyError as exc:
        raise ValueError(f"unknown OOLONG split {split!r}") from exc
    if limit is not None:
        split_size = min(split_size, limit)
    examples = OolongTrecCoarse128KDataset().examples(
        split=split,
        limit=split_size,
        seed=seed,
    )
    return [OolongEnvironment(example) for example in examples]


__all__ = ["OolongEnvironment", "OolongObservation", "oolong_environments"]
