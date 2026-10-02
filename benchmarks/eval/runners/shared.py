"""Shared runner-neutral benchmark behavior."""

from __future__ import annotations

import hashlib
import json
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
    inputs = example.inputs()
    observation_json = example.metadata.get("observation_json")
    if isinstance(observation_json, str) and inputs != {"observation": observation_json}:
        raise ValueError("runner input differs from the canonical observation bytes")
    return inputs


def benchmark_task_contract(example: Example) -> dict[str, Any]:
    """Fingerprint the runner-neutral task, inputs, and declared fixture bytes."""
    inputs = example_inputs(example)
    raw_observation = example.metadata.get("observation_json")
    actual_observation_hash = hashlib.sha256(
        (
            raw_observation
            if isinstance(raw_observation, str)
            else json.dumps(
                inputs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ).encode("utf-8")
    ).hexdigest()
    declared_observation_hash = example.metadata.get("observation_sha256")
    if (
        declared_observation_hash is not None
        and declared_observation_hash != actual_observation_hash
    ):
        raise ValueError("declared observation hash differs from canonical bytes")
    fixture_hashes: dict[str, str] = {}
    raw_paths = example.metadata.get("fixture_paths", [])
    paths = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        for file_path in files:
            name = f"{path.name}/{file_path.relative_to(path)}" if path.is_dir() else path.name
            fixture_hashes[name] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return {
        "version": "canonical-observation-v1",
        "description_sha256": hashlib.sha256(example.prompt.encode()).hexdigest(),
        "observation_sha256": actual_observation_hash,
        "task_prompt_sha256": hashlib.sha256(example.prompt.encode()).hexdigest(),
        "inputs_sha256": hashlib.sha256(
            json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_names": sorted(inputs),
        "fixture_sha256": fixture_hashes,
    }


def canonical_observation(example: Example) -> dict[str, Any] | None:
    raw = example.metadata.get("observation_json")
    if not isinstance(raw, str):
        return None
    observation = json.loads(raw)
    if not isinstance(observation, dict):
        raise ValueError("canonical observation must be a JSON object")
    return observation


def fanout_payloads_from_example(
    example: Example,
) -> dict[str, dict[str, str]] | None:
    observation = canonical_observation(example)
    if observation is None or example.metadata.get("environment_family") != "fanout":
        return None
    raw_payloads = observation.get("payloads")
    if not isinstance(raw_payloads, dict):
        raise ValueError("fanout observation is missing payloads")
    return {
        str(filename): {
            "worker": str(payload["worker"]),
            "filename": str(payload["filename"]),
            "task": str(payload["task"]),
        }
        for filename, payload in raw_payloads.items()
    }


def fanout_shards_from_payloads(
    payloads: dict[str, dict[str, str]] | None,
) -> list[dict[str, str]] | None:
    if payloads is None:
        return None
    return [
        {
            "worker": payload["worker"],
            "file": filename,
        }
        for filename, payload in payloads.items()
    ]


def fact_lookup_data(example: Example) -> dict[str, Any] | None:
    """Return canonical lookup data when the task declares that tool."""
    if example.metadata.get("tool") != "fact_lookup":
        return None
    return {str(key).casefold(): value for key, value in example.metadata.get("facts", {}).items()}


def runtime_prompt(task_prompt: str, guidance: str | None = None) -> str:
    """Attach framework mechanics without changing the benchmark-owned task."""
    if not guidance:
        return task_prompt
    return (
        "<benchmark-task>\n"
        f"{task_prompt.rstrip()}\n"
        "</benchmark-task>\n\n"
        "<runtime-interface>\n"
        f"{guidance.strip()}\n"
        "</runtime-interface>\n"
    )


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
    if environment is None and example.metadata.get("benchmark") == "interactive":
        family = str(example.metadata.get("environment_family", ""))
        environment = {
            "kind": {
                "sudoku": "sudoku_search",
                "minimaze": "minimaze",
                "mazebench": "mazebench",
            }.get(family, family),
            "environment_id": str(example.metadata["environment_id"]),
            "observation_sha256": str(example.metadata["observation_sha256"]),
        }
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
    "benchmark_task_contract",
    "canonical_observation",
    "canonical_delegation_metrics",
    "example_inputs",
    "fact_lookup_data",
    "fanout_payloads_from_example",
    "fanout_shards_from_payloads",
    "materialize_fixtures",
    "prepare_run",
    "runtime_prompt",
]
