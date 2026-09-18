"""Shared runner-neutral benchmark behavior."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.telemetry import canonical_delegation_metrics
from benchmarks.eval.types import Example, RunContext

FACT_LOOKUP_DESCRIPTION = (
    "Required task-data source: call once for each exact entity named in the "
    "question and preserve the returned numeric precision."
)


def example_inputs(example: Example) -> dict[str, str]:
    """Return the exact public input mapping supplied to every runner."""
    return example.inputs()


def fact_lookup_data(example: Example) -> dict[str, Any] | None:
    """Return canonical lookup data when the task declares that tool."""
    if example.metadata.get("tool") != "fact_lookup":
        return None
    return {str(key).casefold(): value for key, value in example.metadata.get("facts", {}).items()}


def materialize_fixtures(example: Example, work_dir: Path) -> None:
    """Copy the same declared filesystem fixtures into one runner workdir."""
    raw = example.metadata.get("fixture_paths", [])
    paths = [raw] if isinstance(raw, str) else list(raw)
    for value in paths:
        source = Path(value).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"benchmark fixture does not exist: {source}")
        target = work_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """Runner-neutral task bytes and filesystem layout for one row."""

    inputs: dict[str, str]
    facts: dict[str, Any] | None
    environment: dict[str, Any] | None
    work_dir: Path
    episode_id: str


def prepare_run(example: Example, ctx: RunContext, *, runtime: str) -> PreparedRun:
    work_dir = ctx.artifact_dir / "workdir"
    work_dir.mkdir(parents=True, exist_ok=True)
    materialize_fixtures(example, work_dir)
    raw_environment = example.metadata.get("environment")
    environment = dict(raw_environment) if isinstance(raw_environment, dict) else None
    suffix = "" if runtime == "rlmflow" else f":{runtime}"
    return PreparedRun(
        inputs=example_inputs(example),
        facts=fact_lookup_data(example),
        environment=environment,
        work_dir=work_dir,
        episode_id=f"{ctx.run_id}:{example.id}:{ctx.params.get('seed', 0)}{suffix}",
    )


__all__ = [
    "FACT_LOOKUP_DESCRIPTION",
    "PreparedRun",
    "canonical_delegation_metrics",
    "example_inputs",
    "fact_lookup_data",
    "materialize_fixtures",
    "prepare_run",
]
