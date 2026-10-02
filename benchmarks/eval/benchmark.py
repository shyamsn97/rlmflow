"""Environment-centered benchmark execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.environment import (
    Environment,
    canonical_observation_json,
    observation_sha256,
)
from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics, write_reports
from benchmarks.eval.run import _run_jobs, parse_model_spec
from benchmarks.eval.types import (
    ComponentSpec,
    Example,
    ModelSpec,
    Prediction,
    Score,
    SuiteConfig,
)


@dataclass(frozen=True)
class _EnvironmentScorer:
    """Private bridge while runners still speak the legacy Example protocol."""

    name: str
    environment: Environment

    def score(self, example: Example, prediction: Prediction) -> Score:
        del example
        if prediction.error:
            return Score(value=0.0, correct=False, details={"error": prediction.error})
        return self.environment.report(prediction, RuntimeMetrics()).score


def benchmark(
    *,
    name: str,
    description: str,
    environments: list[Environment],
    runners: list[str | ComponentSpec],
    model: str | ModelSpec,
    output_dir: str | Path,
    seed: int = 0,
    executor: str = "local",
    parallelism: int = 1,
    resume: bool = False,
    attempt_timeout: int = 900,
    docker_image: str = "rlmflow-eval:local",
    docker_cpus: float = 2.0,
    docker_memory: str = "4g",
    modal_app_name: str = "rlmflow-benchmarks",
    modal_cpu: float = 1.0,
    modal_timeout: int = 3600,
) -> list[EnvironmentReport]:
    """Run an environment × runner matrix and write its complete report."""
    if not environments:
        raise ValueError("benchmark requires at least one environment")
    if not runners:
        raise ValueError("benchmark requires at least one runner")
    if len({environment.id for environment in environments}) != len(environments):
        raise ValueError("environment ids must be unique within a benchmark")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_spec = parse_model_spec(model) if isinstance(model, str) else model
    runner_specs = [
        ComponentSpec(name=runner) if isinstance(runner, str) else runner for runner in runners
    ]
    run_id = root.name
    config = SuiteConfig(
        run_id=run_id,
        datasets=[],
        runners=runner_specs,
        model=model_spec,
        loggers=[],
        seeds=[seed],
        output_root=root.parent,
        resume=resume,
        executor=executor,
        parallelism=parallelism,
        attempt_timeout=attempt_timeout,
        docker_image=docker_image,
        docker_cpus=docker_cpus,
        docker_memory=docker_memory,
        modal_app_name=modal_app_name,
        modal_cpu=modal_cpu,
        modal_timeout=modal_timeout,
    )

    completed = _read_completed(root / "rows.jsonl") if resume else {}
    jobs: list[dict[str, Any]] = []
    contracts: dict[str, tuple[str, str]] = {}
    environments_by_id = {environment.id: environment for environment in environments}
    for environment in environments:
        observation = environment.start(f"preflight-{environment.id}")
        observation_json = canonical_observation_json(observation)
        observation_hash = observation_sha256(observation)
        description_hash = hashlib.sha256(environment.description.encode()).hexdigest()
        example = example_for_environment(
            benchmark_name=name,
            environment=environment,
            observation=observation,
        )
        _validate_public_example(environment, example, observation_json, observation_hash)
        contracts[environment.id] = (description_hash, observation_hash)
        scorer = _EnvironmentScorer(name=name, environment=environment)
        for runner_spec in runner_specs:
            key = (environment.id, runner_spec.name, model_spec.name, seed)
            if key in completed:
                continue
            job_index = len(jobs)
            payload = {
                "run_id": run_id,
                "root": str(config.root),
                "dataset_name": name,
                "runner": asdict(runner_spec),
                "model": asdict(model_spec),
                "seed": seed,
                "benchmark_name": name,
                "environment_id": environment.id,
                "example": asdict(example),
                "job_index": job_index,
                "attempt": 0,
                "best_of_n": 1,
                "attempt_timeout": attempt_timeout,
            }
            jobs.append(
                {
                    "dataset": scorer,
                    "example": example,
                    "payloads": [payload],
                }
            )

    reports = list(completed.values())
    for row in _run_jobs(config, jobs):
        description_hash, observation_hash = contracts[row.example_id]
        row = _verify_row_contract(
            row,
            description_sha256=description_hash,
            observation_sha256=observation_hash,
        )
        normalized = EnvironmentReport.from_row(
            benchmark=name,
            row=row,
            description_sha256=description_hash,
            observation_sha256=observation_hash,
        )
        report = environments_by_id[row.example_id].report(row.prediction, normalized.runtime)
        reports.append(
            replace(
                report,
                benchmark=name,
                environment_id=row.example_id,
                runner=row.runner,
                model=row.model,
                seed=row.seed,
                score=row.score if row.prediction.error else report.score,
                description_sha256=description_hash,
                observation_sha256=observation_hash,
                timestamp=row.timestamp,
            )
        )
        reports.sort(key=lambda item: (item.environment_id, item.runner, item.seed or 0))
        write_reports(root=root, name=name, description=description, reports=reports)
    reports.sort(key=lambda item: (item.environment_id, item.runner, item.seed or 0))
    write_reports(root=root, name=name, description=description, reports=reports)
    return reports


def _verify_row_contract(
    row,
    *,
    description_sha256: str,
    observation_sha256: str,
):
    if row.prediction.error:
        return row
    contract = row.prediction.metrics.get("benchmark_task", {})
    actual_description = contract.get(
        "description_sha256",
        contract.get("task_prompt_sha256"),
    )
    actual_observation = contract.get(
        "observation_sha256",
        contract.get("inputs_sha256"),
    )
    if (
        actual_description == description_sha256
        and actual_observation == observation_sha256
    ):
        return row
    error = (
        "ObservationParityError: runner task bytes differ from the canonical "
        f"environment (description={actual_description}, observation={actual_observation})"
    )
    prediction = replace(row.prediction, error=error)
    return replace(
        row,
        prediction=prediction,
        score=Score(value=0.0, correct=False, details={"error": error}),
    )


def example_for_environment(
    *,
    benchmark_name: str,
    environment: Environment,
    observation=None,
) -> Example:
    observation = observation or environment.start(f"delivery-{environment.id}")
    observation_json = canonical_observation_json(observation)
    return Example(
        id=environment.id,
        prompt=environment.description,
        context={"observation": observation_json},
        expected=None,
        metadata={
            "benchmark": benchmark_name,
            "environment_id": environment.id,
            "environment_family": type(observation).__name__.removesuffix(
                "Observation"
            ).lower(),
            "observation_json": observation_json,
            "observation_sha256": observation_sha256(observation),
        },
    )


def _validate_public_example(
    environment: Environment,
    example: Example,
    observation_json: str,
    observation_hash: str,
) -> None:
    if example.id != environment.id:
        raise ValueError(f"{environment.id}: public example id does not match environment id")
    if example.prompt != environment.description:
        raise ValueError(f"{environment.id}: runner description differs from environment description")
    if example.expected is not None:
        raise ValueError(f"{environment.id}: hidden expected answer leaked into runner payload")
    metadata_hash = example.metadata.get("observation_sha256")
    if metadata_hash != observation_hash:
        raise ValueError(f"{environment.id}: public observation hash mismatch")
    if example.metadata.get("observation_json") != observation_json:
        raise ValueError(f"{environment.id}: public observation bytes mismatch")


def _read_completed(path: Path) -> dict[tuple[str, str, str, int | None], EnvironmentReport]:
    if not path.exists():
        return {}
    reports: dict[tuple[str, str, str, int | None], EnvironmentReport] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        report = EnvironmentReport.from_dict(json.loads(line))
        reports[(report.environment_id, report.runner, report.model, report.seed)] = report
    return reports


__all__ = ["benchmark", "example_for_environment"]
