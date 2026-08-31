"""Small core types for the benchmark harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    expected: Any = None
    context: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def inputs(self) -> dict[str, str]:
        if self.context is None:
            return {}
        if isinstance(self.context, dict):
            return {str(key): str(value) for key, value in self.context.items()}
        return {"context": str(self.context)}


@dataclass(frozen=True)
class Score:
    value: float
    correct: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    answer: str
    usage: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class Row:
    run_id: str
    dataset: str
    example_id: str
    runner: str
    model: str
    seed: int | None
    prediction: Prediction
    score: Score
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Row:
        return cls(
            **{
                **data,
                "prediction": Prediction(**data["prediction"]),
                "score": Score(**data["score"]),
            }
        )


@dataclass(frozen=True)
class RunContext:
    run_id: str
    root: Path
    artifact_dir: Path
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name if self.provider == "fake" else f"{self.provider}:{self.name}"


@dataclass(frozen=True)
class SuiteConfig:
    run_id: str
    datasets: list[ComponentSpec]
    runners: list[ComponentSpec]
    model: ModelSpec
    loggers: list[ComponentSpec]
    seeds: list[int]
    split: str = "test"
    limit: int | None = None
    output_root: Path = Path("benchmarks/runs")
    resume: bool = False
    executor: str = "local"
    parallelism: int = 1
    best_of_n: int = 1
    attempt_timeout: int = 900
    modal_app_name: str = "rlmflow-benchmarks"
    modal_cpu: float = 1.0
    modal_timeout: int = 3600

    @property
    def root(self) -> Path:
        return self.output_root / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "datasets": [asdict(spec) for spec in self.datasets],
            "runners": [asdict(spec) for spec in self.runners],
            "model": asdict(self.model),
            "loggers": [asdict(spec) for spec in self.loggers],
            "seeds": self.seeds,
            "split": self.split,
            "limit": self.limit,
            "output_root": str(self.output_root),
            "resume": self.resume,
            "executor": self.executor,
            "parallelism": self.parallelism,
            "best_of_n": self.best_of_n,
            "attempt_timeout": self.attempt_timeout,
            "modal_app_name": self.modal_app_name,
            "modal_cpu": self.modal_cpu,
            "modal_timeout": self.modal_timeout,
        }


class Dataset:
    name = ""

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        raise NotImplementedError

    def score(self, example: Example, prediction: Prediction) -> Score:
        raise NotImplementedError


class Model:
    provider = ""
    name = ""

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    def usage(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        return None


class Runner:
    name = ""

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        raise NotImplementedError


class Logger:
    def start(self, config: dict[str, Any]) -> None:
        return None

    def example_start(self, example: Example, *, runner: str, model: str) -> None:
        return None

    def row(self, row: Row) -> None:
        return None

    def summary(self, rows: list[Row]) -> None:
        return None

    def finish(self) -> None:
        return None


__all__ = [
    "ComponentSpec",
    "Dataset",
    "Example",
    "Logger",
    "Model",
    "ModelSpec",
    "Prediction",
    "Row",
    "RunContext",
    "Runner",
    "Score",
    "SuiteConfig",
]
