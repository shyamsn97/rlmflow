"""Needle in a haystack: find one record among 2,000 by its marker.

A control: every system should solve it, so a miss points at the system, not the
task. Records are generated from the seed, so every run sees the same file.
"""

from __future__ import annotations

import random
from pathlib import Path

from benchmarks.table.systems import Limits, Outcome

NAME = "needle"
SIZE = 10
RECORDS = 2_000
FILLER_WORDS = 16
VOCABULARY = (
    "harbor lantern quartz meadow orbit cobalt archive cedar "
    "signal vector delta ember garden matrix notebook prairie"
).split()

TASK = """\
Find the record whose marker is exactly `{marker}` and report its `secret` value.

The records are in `context.txt` in this directory. Write only the secret to
`answer.txt`."""


def example(seed: int) -> dict:
    rng = random.Random(seed)
    needle = rng.randrange(RECORDS)
    marker = f"needle-marker-{seed:04d}-{rng.randrange(10**8):08d}"
    secret = f"SECRET-{seed:04d}-{rng.randrange(10**10):010d}"
    blocks = []
    for i in range(RECORDS):
        words = random.Random(seed * 1_000_003 + i).choices(VOCABULARY, k=FILLER_WORDS)
        hit = i == needle
        blocks.append(
            f"RECORD {i:04d}\n"
            f"title: synthetic benchmark record {i}\n"
            f"marker: {marker if hit else f'decoy-{seed:04d}-{i:04d}'}\n"
            f"body: {' '.join(words)}\n"
            f"secret: {secret if hit else f'DECOY-{seed:04d}-{i:04d}'}\n"
            "END_RECORD"
        )
    rng.shuffle(blocks)
    return {"marker": marker, "secret": secret, "context": "\n\n".join(blocks)}


def ids() -> list[str]:
    return [f"needle_{seed:04d}" for seed in range(SIZE)]


def prepare(task_id: str, workdir: Path, limits: Limits) -> str:
    row = example(int(task_id.split("_")[1]))
    (workdir / "context.txt").write_text(row["context"])
    return TASK.format(marker=row["marker"])


def score(task_id: str, workdir: Path, outcome: Outcome) -> dict:
    expected = example(int(task_id.split("_")[1]))["secret"]
    path = workdir / "answer.txt"
    answer = path.read_text().strip().strip("`'\"").strip() if path.exists() else ""
    return {"score": float(answer == expected), "answer": answer, "expected": expected}
