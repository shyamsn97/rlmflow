"""Swarm fan-out environments and runner-neutral scoring."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from benchmarks.eval.delegation.fanout import score_orchestration
from benchmarks.eval.environments.environment import Environment, Observation
from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics
from benchmarks.eval.tasks.swarm_fanout import (
    SwarmFanoutDataset,
    fanout_payloads,
)
from benchmarks.eval.types import Example, Prediction, Score

ANSWER_LINE = re.compile(r"^\s*[-*]\s+(.+?):\s*(.*?)\s*$")


@dataclass(frozen=True, slots=True)
class FanoutObservation(Observation):
    payloads: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, JsonValue]:
        return {"payloads": self.payloads}


class FanoutEnvironment(Environment[FanoutObservation, dict]):
    def __init__(self, example: Example) -> None:
        self.id = example.id
        self.description = example.prompt
        self._shards = tuple(dict(shard) for shard in example.metadata["shards"])
        shard_dir = Path(example.metadata["fixture_paths"][0])
        self._observation = FanoutObservation(
            payloads=fanout_payloads(list(self._shards), shard_dir)
        )

    def start(self, instance_id: str) -> FanoutObservation:
        del instance_id
        return self._observation

    def metrics(self, instance_id: str) -> dict:
        del instance_id
        return {
            "shards": len(self._shards),
            "payload_chars": sum(
                len(payload["task"]) for payload in self._observation.payloads.values()
            ),
        }

    def report(
        self,
        prediction: Prediction,
        runtime: RuntimeMetrics,
    ) -> EnvironmentReport[dict]:
        answers = _parse_answers(prediction.answer)
        coverage = _score_coverage(self._shards, answers)
        expected_prompt_hashes = {
            filename: hashlib.sha256(payload["task"].encode()).hexdigest()
            for filename, payload in self._observation.payloads.items()
        }
        orchestration = score_orchestration(
            list(self._shards),
            prediction.metrics.get("fanout", {}),
            expected_prompt_hashes=expected_prompt_hashes,
        )
        runtime_delegation = prediction.metrics.get("delegation", {})
        if not isinstance(runtime_delegation, dict):
            runtime_delegation = {}
        delegation = {
            **runtime_delegation,
            **orchestration,
            "children": int(orchestration.get("direct_children", 0)),
            "query_calls": int(runtime_delegation.get("query_calls", 0)),
        }
        checks = {
            "coverage": coverage["coverage"],
            "delegation_evidence": orchestration["delegation_evidence"],
            "dedup": orchestration["dedup"],
            "receipts": orchestration["receipts"],
            "prompt_parity": orchestration["prompt_parity"],
        }
        resolved = all(checks.values())
        score = Score(
            value=sum(checks.values()) / len(checks),
            correct=resolved,
            details={**checks, "resolved": resolved},
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
            environment={
                **self.metrics(""),
                **coverage,
                "delegation": delegation,
                "per_shard_answers": answers,
            },
            artifacts=prediction.artifacts,
            error=prediction.error,
        )


def fanout_environments(*, split: str, limit: int | None, seed: int) -> list[FanoutEnvironment]:
    examples = SwarmFanoutDataset().examples(split=split, limit=limit, seed=seed)
    return [FanoutEnvironment(example) for example in examples]


def _parse_answers(text: str) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = ANSWER_LINE.match(line)
        if match is not None:
            filename = match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
            answers.setdefault(filename, []).append(match.group(2))
    return answers


def _score_coverage(
    shards: tuple[dict[str, Any], ...],
    answers: dict[str, list[str]],
) -> dict[str, Any]:
    missing: list[str] = []
    wrong: dict[str, dict[str, Any]] = {}
    for shard in shards:
        filename = str(shard["file"])
        expected = str(shard["expected"])
        found = answers.get(filename)
        if not found:
            missing.append(filename)
        elif any(answer != expected for answer in found):
            wrong[filename] = {"found": found}
    return {
        "coverage": not missing and not wrong,
        "missing_shards": missing,
        "wrong_answers": wrong,
    }


__all__ = ["FanoutEnvironment", "FanoutObservation", "fanout_environments"]
