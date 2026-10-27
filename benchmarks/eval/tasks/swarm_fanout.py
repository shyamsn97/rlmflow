"""Graph-native adaptation of Prime Agent's swarm-fanout evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.delegation.fanout import score_orchestration
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "swarm_fanout"
ANSWER_LINE = re.compile(r"^\s*[-*]\s+(.+?):\s*(.*?)\s*$")
CHILD_INSTRUCTION = "Answer the question using only the supplied content. Return only the number."


@dataclass(frozen=True, slots=True)
class GeneratedFixture:
    name: str
    format: str
    shard_count: int
    rows: int
    operation: str


GENERATED_FIXTURES = (
    GeneratedFixture("csv-4-max", "csv", 4, 12, "max"),
    GeneratedFixture("jsonl-4-sum", "jsonl", 4, 10, "sum"),
    GeneratedFixture("tsv-6-min", "tsv", 6, 14, "min"),
    GeneratedFixture("kv-6-count", "kv", 6, 16, "count"),
    GeneratedFixture("logs-8-max", "log", 8, 18, "max"),
    GeneratedFixture("markdown-8-sum", "markdown", 8, 9, "sum"),
    GeneratedFixture("nested-json-8-min", "nested-json", 8, 15, "min"),
    GeneratedFixture("text-8-count", "text", 8, 20, "count"),
    GeneratedFixture("csv-12-sum", "csv", 12, 13, "sum"),
    GeneratedFixture("jsonl-12-count", "jsonl", 12, 17, "count"),
    GeneratedFixture("logs-12-min", "log", 12, 19, "min"),
    GeneratedFixture("nested-json-16-max", "nested-json", 16, 12, "max"),
    GeneratedFixture("markdown-16-count", "markdown", 16, 16, "count"),
    GeneratedFixture("text-16-sum", "text", 16, 11, "sum"),
)


@dataset("swarm-fanout", tags=["delegation", "orchestration", "artifact-scored"])
class SwarmFanoutDataset(Dataset):
    """Frozen fan-out matrix with deterministic artifact and graph scoring."""

    fixture_names = (
        "json-events",
        "csv-metrics",
        *(fixture.name for fixture in GENERATED_FIXTURES),
    )

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
        expected_prompt_hashes = {
            filename: hashlib.sha256(payload["task"].encode()).hexdigest()
            for filename, payload in fanout_payloads_for_example(example).items()
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
        if fixture_dir.is_dir():
            manifest = json.loads((fixture_dir / "fixture.json").read_text())
            shards = list(manifest["shards"])
            metadata = {
                "fixture_paths": [str(fixture_dir / "shards")],
                "shards": shards,
            }
        else:
            spec = next(fixture for fixture in GENERATED_FIXTURES if fixture.name == name)
            shards, contents = _generated_fixture(spec)
            metadata = {
                "generated_contents": contents,
                "generator": "deterministic-fanout-v1",
                "shards": shards,
            }
        return Example(
            id=f"swarm_fanout_{name.replace('-', '_')}",
            prompt=fanout_prompt(shards),
            expected={shard["file"]: str(shard["expected"]) for shard in shards},
            metadata={
                "fixture": name,
                **metadata,
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


def fanout_payloads_for_example(example: Example) -> dict[str, dict[str, str]]:
    """Build canonical payloads from either checked-in or generated fixture content."""
    shards = list(example.metadata["shards"])
    generated = example.metadata.get("generated_contents")
    if isinstance(generated, dict):
        return {
            str(shard["file"]): fanout_payload(
                shard,
                str(generated[str(shard["file"])]),
            )
            for shard in shards
        }
    return fanout_payloads(shards, Path(example.metadata["fixture_paths"][0]))


def fanout_prompt(shards: list[dict[str, Any]]) -> str:
    """Return the benchmark-owned task shared byte-for-byte by every runner."""
    questions = "\n".join(
        f"- shards/{shard['file']}: {shard['question']} (child name: {shard['worker']})"
        for shard in shards
    )
    return f"""\
You are the orchestrator of a {len(shards)}-shard data-analysis fan-out.
Delegate each question to exactly one independent direct child execution.
Use the declared child name for that shard, pass the corresponding complete
child task unchanged, and do not compute shard answers in the parent. Launch
each shard exactly once, wait for every child, and copy every answer from the
corresponding child result.

Shard questions:
{questions}

Return exactly one bullet per shard, with no extra shard bullets:
- <shard-file-name>: <answer>
Use only the file basename in each bullet (for example, `{shards[0]["file"]}`,
not `shards/{shards[0]["file"]}`).
"""


def _generated_fixture(
    spec: GeneratedFixture,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    shards: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    for shard_index in range(1, spec.shard_count + 1):
        values = [
            ((shard_index * 37 + row_index * 19 + len(spec.name) * 11) % 97) + 3
            for row_index in range(spec.rows)
        ]
        threshold = 50
        expected = {
            "max": max(values),
            "min": min(values),
            "sum": sum(values),
            "count": sum(value > threshold for value in values),
        }[spec.operation]
        filename = f"{spec.name}-{shard_index:02d}{_extension(spec.format)}"
        worker = f"worker-{spec.name}-{shard_index:02d}"
        question = {
            "max": "What is the maximum value of the value field?",
            "min": "What is the minimum value of the value field?",
            "sum": "What is the sum of the value field?",
            "count": f"How many records have a value greater than {threshold}?",
        }[spec.operation]
        shards.append(
            {
                "worker": worker,
                "file": filename,
                "question": question,
                "expected": expected,
            }
        )
        contents[filename] = _format_records(spec.format, values)
    return shards, contents


def _extension(format_name: str) -> str:
    return {
        "csv": ".csv",
        "jsonl": ".jsonl",
        "tsv": ".tsv",
        "kv": ".txt",
        "log": ".log",
        "markdown": ".md",
        "nested-json": ".json",
        "text": ".txt",
    }[format_name]


def _format_records(format_name: str, values: list[int]) -> str:
    records = [
        {
            "id": f"item-{index:02d}",
            "value": value,
            "noise": (value * 7 + index * 3) % 101,
        }
        for index, value in enumerate(values, start=1)
    ]
    if format_name == "csv":
        return "id,value,noise\n" + "\n".join(
            f"{record['id']},{record['value']},{record['noise']}" for record in records
        )
    if format_name == "jsonl":
        return "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
    if format_name == "tsv":
        return "id\tvalue\tnoise\n" + "\n".join(
            f"{record['id']}\t{record['value']}\t{record['noise']}" for record in records
        )
    if format_name == "kv":
        return "\n".join(
            f"id={record['id']} value={record['value']} noise={record['noise']}"
            for record in records
        )
    if format_name == "log":
        return "\n".join(
            f"2026-01-01T00:{index:02d}:00Z metric id={record['id']} "
            f"value={record['value']} noise={record['noise']}"
            for index, record in enumerate(records)
        )
    if format_name == "markdown":
        return "| id | value | noise |\n|---|---:|---:|\n" + "\n".join(
            f"| {record['id']} | {record['value']} | {record['noise']} |" for record in records
        )
    if format_name == "nested-json":
        return json.dumps({"metadata": {"version": 1}, "records": records}, indent=2)
    if format_name == "text":
        return "\n".join(
            f"Record {record['id']} has value {record['value']} and noise {record['noise']}."
            for record in records
        )
    raise ValueError(f"unknown generated fan-out format {format_name!r}")


def fanout_runtime_guidance(runtime: str) -> str:
    """Describe only how one runtime exposes the shared payload and child API."""
    instructions = {
        "rlmflow": (
            "The exact child payloads are in `fanout_payloads`. For each "
            "basename, set `p = fanout_payloads[basename]`, then call "
            "`launch_agent(p['task'], name=p['worker'])` once, concurrently. "
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
    "fanout_payloads_for_example",
    "fanout_prompt",
    "fanout_runtime_guidance",
]
