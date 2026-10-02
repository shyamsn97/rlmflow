"""Graph-native adaptation of Prime Agent's swarm-fanout evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.delegation.fanout import score_orchestration
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "swarm_fanout"
ANSWER_LINE = re.compile(r"^\s*[-*]\s+(.+?):\s*(.*?)\s*$")
CHILD_INSTRUCTION = (
    "Answer the question using only the supplied content. Return only the number."
)


@dataset("swarm-fanout", tags=["delegation", "orchestration", "artifact-scored"])
class SwarmFanoutDataset(Dataset):
    """Two eight-way fan-out tasks with deterministic artifact and graph scoring."""

    fixture_names = ("json-events", "csv-metrics")

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del seed
        if split not in {"test", "pilot", "full"}:
            raise ValueError("swarm-fanout exposes only pilot and full suite splits")
        names = self.fixture_names[:1] if split in {"test", "pilot"} else self.fixture_names
        if limit is not None:
            names = names[:limit]
        return [self._example(name) for name in names]

    def score(self, example: Example, prediction: Prediction) -> Score:
        shards = list(example.metadata["shards"])
        shard_dir = Path(example.metadata["fixture_paths"][0])
        expected_prompt_hashes = {
            filename: hashlib.sha256(payload["task"].encode()).hexdigest()
            for filename, payload in fanout_payloads(shards, shard_dir).items()
        }
        coverage = _score_coverage(shards, _parse_answers(prediction.answer))
        orchestration = score_orchestration(
            shards,
            prediction.metrics.get("fanout", {}),
            expected_prompt_hashes=expected_prompt_hashes,
        )
        checks = {
            "coverage": coverage["coverage"],
            "delegation_evidence": orchestration["delegation_evidence"],
            "dedup": orchestration["dedup"],
            "receipts": orchestration["receipts"],
            "prompt_parity": orchestration["prompt_parity"],
        }
        resolved = all(checks.values())
        return Score(
            value=sum(checks.values()) / len(checks),
            correct=resolved,
            details={
                **checks,
                **coverage,
                **orchestration,
                "resolved": resolved,
                "scorer": "rlmflow:runner-neutral-fanout-v1",
            },
        )

    def _example(self, name: str) -> Example:
        fixture_dir = FIXTURES / name
        manifest = json.loads((fixture_dir / "fixture.json").read_text())
        shards = list(manifest["shards"])
        return Example(
            id=f"swarm_fanout_{name.replace('-', '_')}",
            prompt=fanout_prompt(shards),
            expected={shard["file"]: str(shard["expected"]) for shard in shards},
            metadata={
                "fixture": name,
                "fixture_paths": [str(fixture_dir / "shards")],
                "shards": shards,
            },
        )


def fanout_payload(shard: dict[str, Any], content: str) -> dict[str, str]:
    """Return the worker identity and its complete shared prompt."""
    worker = str(shard["worker"])
    filename = str(shard["file"])
    question = str(shard["question"])
    task = (
        f"Worker: {worker}\n"
        f"Filename: {filename}\n"
        f"Question: {question}\n"
        "<content>\n"
        f"{content}"
        "</content>\n"
        f"Instruction: {CHILD_INSTRUCTION}"
    )
    return {
        "worker": worker,
        "filename": filename,
        "task": task,
    }


def fanout_payloads(
    shards: list[dict[str, Any]],
    shard_dir: Path,
) -> dict[str, dict[str, str]]:
    """Build the identical complete child prompts consumed by every runner."""
    return {
        str(shard["file"]): fanout_payload(
            shard,
            (shard_dir / str(shard["file"])).read_text(),
        )
        for shard in shards
    }


def fanout_prompt(shards: list[dict[str, Any]]) -> str:
    """Return the benchmark-owned task shared byte-for-byte by every runner."""
    questions = "\n".join(
        f"- shards/{shard['file']}: {shard['question']} (child name: {shard['worker']})"
        for shard in shards
    )
    return f"""\
You are the orchestrator of an eight-shard data-analysis fan-out.
Delegate each question to exactly one independent direct child execution.
Use the declared child name for that shard, pass the corresponding complete
child task unchanged, and do not compute shard answers in the parent. Launch
each shard exactly once, wait for every child, and copy every answer from the
corresponding child result.

Shard questions:
{questions}

Return exactly one bullet per shard, with no extra shard bullets:
- <shard-file-name>: <answer>
Use only the file basename in each bullet (for example, `events-01.jsonl`,
not `shards/events-01.jsonl`).
"""


def fanout_runtime_guidance(runtime: str) -> str:
    """Describe only how one runtime exposes the shared payload and child API."""
    instructions = {
        "rlmflow": (
            "The exact child payloads are in `fanout_payloads`. For each "
            "basename, set `p = fanout_payloads[basename]`, then call "
            "`launch_subagent(name=p['worker'], task=p['task'])` once, concurrently. "
            "Await every handle."
        ),
        "official-rlm": (
            "The exact child payloads are in `context['fanout_payloads']`. Use exactly "
            "`llm_query_batched([p['task'] for p in "
            "context['fanout_payloads'].values()])` to invoke the children."
        ),
        "jaz": (
            "The exact child payloads are in `fanout_payloads`. For each basename, set "
            "`p = fanout_payloads[basename]`, then call `invoke(task=p['task'])` "
            "exactly once."
        ),
        "prime-agent": (
            "The exact child payloads are in `fanout-payloads.json`. From the Python "
            "kernel, load that mapping and spawn exactly one RLM child per shard with "
            "`await rlm.spawn(p['task'], name=p['worker'])`. Keep the run active until "
            "all replies arrive."
        ),
    }
    if runtime not in instructions:
        raise ValueError(f"unknown fanout runtime {runtime!r}")
    return instructions[runtime]


def _parse_answers(text: str) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = ANSWER_LINE.match(line)
        if match is not None:
            filename = match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
            answers.setdefault(filename, []).append(match.group(2))
    return answers


def _score_coverage(shards: list[dict[str, Any]], answers: dict[str, list[str]]) -> dict[str, Any]:
    missing: list[str] = []
    wrong: dict[str, dict[str, Any]] = {}
    for shard in shards:
        filename = str(shard["file"])
        expected = str(shard["expected"])
        found = answers.get(filename)
        if not found:
            missing.append(filename)
        elif any(answer != expected for answer in found):
            wrong[filename] = {"expected": expected, "found": found}
    return {
        "coverage": not missing and not wrong,
        "missing_shards": missing,
        "wrong_answers": wrong,
    }


__all__ = [
    "CHILD_INSTRUCTION",
    "SwarmFanoutDataset",
    "fanout_payload",
    "fanout_payloads",
    "fanout_prompt",
    "fanout_runtime_guidance",
]
