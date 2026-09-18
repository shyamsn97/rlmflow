"""OOLONG-Pairs benchmark from the Recursive Language Models paper."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score

PAIR_DATASET = "mit-oasys/oolong-pairs"
PAIR_REVISION = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
CONTEXT_DATASET = "lsteno/RLM-Evals"
CONTEXT_CONFIG = "oolong_pairs_contexts"
CONTEXT_REVISION = "a6aea6d06da9f08d701038b64195049cf71e1997"
CONTEXT_LENGTHS = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
PAPER_CONTEXT_LEN = 32768
PAPER_QUESTION_IDS = tuple(str(index) for index in range(1, 21))
_PARENTHESIZED_PAIR_PATTERN = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_BARE_PAIR_PATTERN = re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)(?!\d)")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class OolongPairsDataset(Dataset):
    """The 20 official pair-aggregation queries at one context length.

    Gold answers come from ``mit-oasys/oolong-pairs``. Contexts are a compact
    mirror of the corresponding unlabeled ``oolongbench/oolong-synth``
    ``trec_coarse`` validation windows.
    """

    def __init__(
        self,
        data_dir: str = "evals/data",
        context_len: int = 32768,
        max_samples: int | None = 20,
        question_ids: tuple[str, ...] | None = None,
    ) -> None:
        if context_len not in CONTEXT_LENGTHS:
            choices = ", ".join(str(value) for value in CONTEXT_LENGTHS)
            raise ValueError(f"context_len must be one of: {choices}")
        self.data_dir = Path(data_dir)
        self.context_len = context_len
        self.max_samples = max_samples
        self.question_ids = tuple(str(item) for item in question_ids) if question_ids else None
        self._rows: list[dict[str, Any]] | None = None
        self._context: str | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        rows = self._load_rows()
        context = self._load_context()
        if self.question_ids is not None:
            by_id = {str(row.get("id", "")).strip(): row for row in rows}
            missing = [question_id for question_id in self.question_ids if question_id not in by_id]
            if missing:
                raise ValueError(f"OOLONG-Pairs is missing question ids: {missing}")
            selected = [by_id[question_id] for question_id in self.question_ids]
            if limit is not None:
                selected = selected[:limit]
            return [self._example(row, context) for row in selected]
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        count = len(indices)
        if self.max_samples is not None:
            count = min(count, self.max_samples)
        if limit is not None:
            count = min(count, limit)
        return [self._example(rows[index], context) for index in indices[:count]]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = _extract_pairs(example.expected or [])
        predicted = _extract_pairs(prediction.answer)
        true_positives = len(expected & predicted)
        precision = true_positives / len(predicted) if predicted else float(not expected)
        recall = true_positives / len(expected) if expected else float(not predicted)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return Score(
            value=f1,
            correct=f1 == 1.0,
            details={
                "precision": precision,
                "recall": recall,
                "expected_pairs": len(expected),
                "predicted_pairs": len(predicted),
                "true_positives": true_positives,
            },
        )

    def _example(self, row: dict[str, Any], context: str) -> Example:
        question_id = str(row.get("id", "")).strip()
        return Example(
            id=f"oolong_pairs_{self.context_len}_{int(question_id):02d}",
            prompt=str(row.get("question", "")).strip(),
            context={"context": context},
            expected=list(row.get("answer") or []),
            metadata={
                "question_id": question_id,
                "context_len": self.context_len,
                "answer_type": row.get("type"),
                "expected_pairs": len(row.get("answer") or []),
            },
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        filename = f"oolong-pairs-{self.context_len}.json"
        local_candidates = (
            self.data_dir / "oolong_pairs" / filename,
            self.data_dir / filename,
        )
        path = next((candidate for candidate in local_candidates if candidate.exists()), None)
        if path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError(
                    "OOLONG-Pairs requires the eval extra: pip install -e '.[eval]'"
                ) from exc
            path = Path(
                hf_hub_download(
                    repo_id=PAIR_DATASET,
                    filename=f"data/{filename}",
                    repo_type="dataset",
                    revision=PAIR_REVISION,
                )
            )
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != 20:
            raise ValueError(f"Expected 20 OOLONG-Pairs queries in {path}.")
        self._rows = [dict(row) for row in rows]
        return self._rows

    def _load_context(self) -> str:
        if self._context is not None:
            return self._context
        try:
            from datasets import load_dataset, load_from_disk
            from datasets.utils.logging import disable_progress_bar
        except ImportError as exc:
            raise RuntimeError(
                "OOLONG-Pairs requires the eval extra: pip install -e '.[eval]'"
            ) from exc

        disable_progress_bar()
        local = self.data_dir / CONTEXT_CONFIG
        if local.exists():
            loaded = load_from_disk(str(local))
            contexts = loaded["eval"] if hasattr(loaded, "keys") and "eval" in loaded else loaded
        else:
            contexts = load_dataset(
                CONTEXT_DATASET,
                CONTEXT_CONFIG,
                split="eval",
                revision=CONTEXT_REVISION,
            )
        context = _context_at_length(contexts, self.context_len)
        self._context = context
        return context


@dataset("oolong-pairs-32k", tags=["core", "paper-aligned", "long-context"])
class OolongPairsPaperDataset(OolongPairsDataset):
    """The paper's 20 OOLONG-Pairs queries at 32K, official order.

    This is the single frozen 32K registration, not the configurable loader.
    """

    def __init__(self, data_dir: str = "evals/data", **_ignored: Any) -> None:
        super().__init__(
            data_dir=data_dir,
            context_len=PAPER_CONTEXT_LEN,
            question_ids=PAPER_QUESTION_IDS,
            max_samples=20,
        )


def _context_at_length(rows: Any, context_len: int) -> str:
    matches = [row for row in rows if int(row.get("context_len") or 0) == context_len]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one OOLONG-Pairs context at length {context_len}, found {len(matches)}."
        )
    context = str(matches[0].get("context_window_text") or "")
    if not context:
        raise ValueError(f"OOLONG-Pairs context {context_len} is empty.")
    return context


def _extract_pairs(value: Any) -> set[tuple[int, int]]:
    if isinstance(value, (list, tuple, set)):
        text = "\n".join(str(item) for item in value)
    else:
        text = str(value)
    text = _THINK_PATTERN.sub("", text)
    matches = _PARENTHESIZED_PAIR_PATTERN.findall(text)
    if not matches:
        matches = _BARE_PAIR_PATTERN.findall(text)
    pairs: set[tuple[int, int]] = set()
    for left_raw, right_raw in matches:
        left, right = int(left_raw), int(right_raw)
        pairs.add((min(left, right), max(left, right)))
    return pairs


__all__ = [
    "OolongPairsDataset",
    "OolongPairsPaperDataset",
    "PAPER_CONTEXT_LEN",
    "PAPER_QUESTION_IDS",
]
