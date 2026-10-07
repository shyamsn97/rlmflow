"""The four public benchmark collections."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from benchmarks.eval.benchmark import benchmark
from benchmarks.eval.environments.environment import Environment
from benchmarks.eval.environments.fanout import fanout_environments
from benchmarks.eval.environments.needle import needle_environments
from benchmarks.eval.environments.oolong import oolong_environments
from benchmarks.eval.reporting import EnvironmentReport
from benchmarks.eval.types import ComponentSpec


class Split(StrEnum):
    PILOT = "pilot"
    FULL = "full"


RUNNER_CAPABILITIES = {
    "fake": {"smoke"},
    "prime-agent": {"delegation"},
    "jaz": {"delegation"},
    "official-rlm": {"smoke", "delegation", "long-context", "interactive"},
    "rlmflow-depth-0": {"smoke", "long-context", "interactive"},
    "rlmflow-depth-1": {"smoke", "delegation", "long-context", "interactive"},
    "rlmflow-depth-2": {"smoke", "delegation", "long-context", "interactive"},
}


def smoke(
    *,
    runners: list[str | ComponentSpec],
    model: str,
    output_dir: str | Path,
    split: Split | str = Split.PILOT,
    seed: int = 0,
    **execution: Any,
) -> list[EnvironmentReport]:
    split = Split(split)
    _validate_runners("smoke", runners)
    return benchmark(
        name="smoke",
        description="Can each method recover a needle from a synthetic haystack?",
        environments=needle_environments(split=split.value, limit=1, seed=seed),
        runners=runners,
        model=model,
        output_dir=output_dir,
        seed=seed,
        **execution,
    )


def delegation(
    *,
    runners: list[str | ComponentSpec],
    model: str,
    output_dir: str | Path,
    split: Split | str = Split.PILOT,
    seed: int = 0,
    **execution: Any,
) -> list[EnvironmentReport]:
    split = Split(split)
    _validate_runners("delegation", runners)
    return benchmark(
        name="delegation",
        description="Can each method fan out work without fabricating or dropping results?",
        environments=fanout_environments(split=split.value, limit=None, seed=seed),
        runners=runners,
        model=model,
        output_dir=output_dir,
        seed=seed,
        **execution,
    )


def long_context(
    *,
    runners: list[str | ComponentSpec],
    model: str,
    output_dir: str | Path,
    split: Split | str = Split.PILOT,
    seed: int = 0,
    **execution: Any,
) -> list[EnvironmentReport]:
    split = Split(split)
    _validate_runners("long-context", runners)
    return benchmark(
        name="long-context",
        description="How do the methods handle paper-aligned long-context tasks?",
        environments=oolong_environments(split=split.value, limit=None, seed=seed),
        runners=runners,
        model=model,
        output_dir=output_dir,
        seed=seed,
        **execution,
    )


def interactive(
    *,
    runners: list[str | ComponentSpec],
    model: str,
    output_dir: str | Path,
    split: Split | str = Split.PILOT,
    seed: int = 0,
    **execution: Any,
) -> list[EnvironmentReport]:
    from benchmarks.eval.environments.catalog import interactive_environments

    split = Split(split)
    _validate_runners("interactive", runners)
    return benchmark(
        name="interactive",
        description="How do the methods search and act in persistent environments?",
        environments=interactive_environments(
            split=split.value,
            seed=seed,
            artifact_root=Path(output_dir) / "environment-work",
        ),
        runners=runners,
        model=model,
        output_dir=output_dir,
        seed=seed,
        **execution,
    )


BENCHMARKS = {
    "smoke": smoke,
    "delegation": delegation,
    "long-context": long_context,
    "interactive": interactive,
}


def environment_by_id(
    benchmark_name: str,
    environment_id: str,
    *,
    seed: int,
    artifact_dir: str | Path,
) -> Environment:
    if benchmark_name == "smoke":
        candidates = needle_environments(split="full", limit=1, seed=seed)
    elif benchmark_name == "delegation":
        candidates = fanout_environments(split="full", limit=None, seed=seed)
    elif benchmark_name == "long-context":
        candidates = oolong_environments(split="full", limit=None, seed=seed)
    elif benchmark_name == "interactive":
        from benchmarks.eval.environments.catalog import interactive_environments

        candidates = [
            *interactive_environments(
                split="pilot",
                seed=seed,
                artifact_root=artifact_dir,
            ),
            *interactive_environments(
                split="full",
                seed=seed,
                artifact_root=artifact_dir,
            ),
        ]
    else:
        raise ValueError(f"unknown benchmark {benchmark_name!r}")
    try:
        return next(item for item in candidates if item.id == environment_id)
    except StopIteration as exc:
        raise ValueError(
            f"{benchmark_name}: unknown environment {environment_id!r}"
        ) from exc


def _validate_runners(benchmark_name: str, runners: list[str | ComponentSpec]) -> None:
    runners = [runner if isinstance(runner, str) else runner.name for runner in runners]
    unknown = [runner for runner in runners if runner not in RUNNER_CAPABILITIES]
    if unknown:
        raise ValueError(f"unknown runner(s): {', '.join(unknown)}")
    incompatible = [
        runner
        for runner in runners
        if benchmark_name not in RUNNER_CAPABILITIES[runner]
    ]
    if incompatible:
        raise ValueError(
            f"{benchmark_name} does not support runner(s): {', '.join(incompatible)}"
        )


__all__ = [
    "BENCHMARKS",
    "Split",
    "RUNNER_CAPABILITIES",
    "delegation",
    "environment_by_id",
    "interactive",
    "long_context",
    "smoke",
]
