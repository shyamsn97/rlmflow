"""Deterministic preflight checks for the benchmark architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.eval.benchmark import example_for_environment
from benchmarks.eval.benchmarks import RUNNER_CAPABILITIES
from benchmarks.eval.environments.catalog import interactive_environments
from benchmarks.eval.environments.environment import (
    Environment,
    Observation,
    canonical_observation_bytes,
    observation_sha256,
)
from benchmarks.eval.environments.fanout import fanout_environments
from benchmarks.eval.environments.needle import needle_environments
from benchmarks.eval.reporting import RuntimeMetrics
from benchmarks.eval.runners.shared import benchmark_task_contract
from benchmarks.eval.types import Prediction


def validate(*, include_mazebench: bool = False) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    environments: list[tuple[str, Environment]] = [
        *[("smoke", item) for item in needle_environments(split="full", limit=1, seed=0)],
        *[
            ("delegation", item)
            for item in fanout_environments(split="full", limit=None, seed=0)
        ],
    ]
    with TemporaryDirectory(prefix="rlmflow-benchmark-preflight-") as directory:
        interactive = interactive_environments(
            split="pilot",
            seed=0,
            artifact_root=Path(directory),
        )
        environments.extend(
            ("interactive", item)
            for item in interactive
            if include_mazebench or item.kind != "mazebench"
        )
        ids: set[tuple[str, str]] = set()
        for benchmark_name, environment in environments:
            key = (benchmark_name, environment.id)
            if key in ids:
                raise ValueError(f"duplicate environment id: {key}")
            ids.add(key)
            instance_id = f"validate-{environment.id}"
            observation = environment.start(instance_id)
            _validate_observation(environment, observation)
            example = example_for_environment(
                benchmark_name=benchmark_name,
                environment=environment,
                observation=observation,
            )
            contract = benchmark_task_contract(example)
            runner_hashes = {
                runner: benchmark_task_contract(example)["observation_sha256"]
                for runner, capabilities in RUNNER_CAPABILITIES.items()
                if benchmark_name in capabilities
            }
            if len(set(runner_hashes.values())) != 1:
                raise ValueError(f"{environment.id}: runner observation hashes differ")
            if contract["observation_sha256"] != observation_sha256(observation):
                raise ValueError(f"{environment.id}: observation contract mismatch")
            if benchmark_name == "interactive":
                _validate_interactive(environment, instance_id, observation)
            report = environment.report(Prediction(answer=""), RuntimeMetrics())
            json.dumps(report.to_dict(), sort_keys=True)
            checked.append(
                {
                    "benchmark": benchmark_name,
                    "environment_id": environment.id,
                    "observation_type": type(observation).__name__,
                    "description_sha256": contract["description_sha256"],
                    "observation_sha256": contract["observation_sha256"],
                    "runner_hashes": runner_hashes,
                }
            )
    return {"checked": checked, "count": len(checked)}


def _validate_observation(environment: Environment, observation: Observation) -> None:
    first = canonical_observation_bytes(observation)
    second = canonical_observation_bytes(observation)
    if first != second:
        raise ValueError(f"{environment.id}: observation serialization is nondeterministic")
    decoded = json.loads(first)
    if not isinstance(decoded, dict):
        raise TypeError(f"{environment.id}: observation must serialize to an object")
    _reject_hidden_keys(environment.id, decoded)


def _validate_interactive(
    environment: Environment,
    instance_id: str,
    observation: Observation,
) -> None:
    observed = environment.observe(instance_id)
    if type(observed) is not type(observation):
        raise TypeError(f"{environment.id}: observe changed observation type")
    checked = environment.execute(instance_id, "observe", {}, max_actions=1)
    if type(checked) is not type(observation):
        raise TypeError(f"{environment.id}: execute changed observation type")
    child = environment.fork(
        f"{instance_id}-child",
        parent_instance_id=instance_id,
        command_prefix=environment.accepted_prefix(instance_id),
    )
    if type(child) is not type(observation):
        raise TypeError(f"{environment.id}: fork changed observation type")


def _reject_hidden_keys(environment_id: str, value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            location = f"{path}.{key}" if path else str(key)
            if str(key).casefold() in {"expected", "gold", "solution", "private_state"}:
                raise ValueError(f"{environment_id}: hidden key leaked at {location}")
            _reject_hidden_keys(environment_id, nested, location)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_hidden_keys(environment_id, nested, f"{path}[{index}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-mazebench", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(validate(include_mazebench=args.include_mazebench), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate"]
