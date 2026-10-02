"""Deterministic synthetic needle-in-haystack dataset."""

from __future__ import annotations

import random
import re

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().strip("`'\"").strip()


@dataset("synthetic_needle", aliases=["needle"], tags=["smoke", "synthetic"])
class SyntheticNeedleDataset(Dataset):
    """Single synthetic needle retrieval with deterministic generation."""

    def __init__(self, records: int = 120, filler_words: int = 16) -> None:
        self.records = records
        self.filler_words = filler_words

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        count = limit or 1
        return [self._example(seed + i) for i in range(count)]

    def _example(self, seed: int) -> Example:
        rng = random.Random(seed)
        needle_index = rng.randrange(self.records)
        marker = f"needle-marker-{seed:04d}-{rng.randrange(10**8):08d}"
        expected = f"SECRET-{seed:04d}-{rng.randrange(10**10):010d}"
        vocabulary = [
            "harbor",
            "lantern",
            "quartz",
            "meadow",
            "orbit",
            "cobalt",
            "archive",
            "cedar",
            "signal",
            "vector",
            "delta",
            "ember",
            "garden",
            "matrix",
            "notebook",
            "prairie",
        ]
        blocks: list[str] = []
        for i in range(self.records):
            local_rng = random.Random((seed * 1_000_003) + i)
            words = [local_rng.choice(vocabulary) for _ in range(self.filler_words)]
            record_marker = marker if i == needle_index else f"decoy-{seed:04d}-{i:04d}"
            record_secret = expected if i == needle_index else f"DECOY-{seed:04d}-{i:04d}"
            blocks.append(
                "\n".join(
                    [
                        f"RECORD {i:04d}",
                        f"title: synthetic benchmark record {i}",
                        f"marker: {record_marker}",
                        f"body: {' '.join(words)}",
                        f"secret: {record_secret}",
                        "END_RECORD",
                    ]
                )
            )
        rng.shuffle(blocks)
        haystack = "\n\n".join(blocks)
        prompt = (
            "Find the record whose marker is exactly "
            f"`{marker}`. Return only that record's `secret` value, with no "
            "explanation and no extra text. Parse the canonical observation JSON; "
            "the records are plain text in its `haystack` string field."
        )
        return Example(
            id=f"synthetic_needle_{seed:04d}",
            prompt=prompt,
            context={"haystack": haystack},
            expected=expected,
            metadata={
                "records": self.records,
                "filler_words": self.filler_words,
                "needle_index": needle_index,
                "marker": marker,
                "context_chars": len(haystack),
            },
        )

    def score(self, example: Example, prediction: Prediction) -> Score:
        normalized = _normalize(prediction.answer)
        expected = str(example.expected)
        correct = normalized == expected
        return Score(
            value=1.0 if correct else 0.0,
            correct=correct,
            details={
                "expected": expected,
                "normalized_answer": normalized,
                "contains_expected": expected in normalized,
            },
        )


__all__ = ["SyntheticNeedleDataset"]
