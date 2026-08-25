"""AIME 2025 benchmark adapter."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("official_aime_2025", tags=["rlm-comparison", "reasoning"])
class AIME2025Dataset(Dataset):
    """All 30 AIME 2025 competition-math problems."""

    dataset_name = "MathArena/aime_2025"

    def __init__(
        self,
        data_dir: str = "evals/data",
        split: str = "train",
        max_samples: int | None = None,
        problem_type_filter: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.problem_type_filter = problem_type_filter
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        rows = _select_rows(self._load(), limit=limit, seed=seed)
        return [self._example(row) for row in rows]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = str(example.expected or "").strip()
        numbers = re.findall(r"-?\d+", prediction.answer)
        correct = bool(expected) and any(_same_integer(value, expected) for value in numbers)
        return Score(
            value=1.0 if correct else 0.0,
            correct=correct,
            details={"expected": expected, "extracted_integers": numbers},
        )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        try:
            from datasets import load_dataset, load_from_disk
            from datasets.utils.logging import disable_progress_bar
        except ImportError as exc:
            raise RuntimeError(
                "AIME 2025 requires the eval extra: pip install -e '.[eval]'"
            ) from exc

        disable_progress_bar()
        local = self.data_dir / "aime_2025"
        if local.exists():
            rows = load_from_disk(str(local))
            if hasattr(rows, "keys") and self.split in rows:
                rows = rows[self.split]
        else:
            rows = load_dataset(self.dataset_name, split=self.split)

        selected: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            row = dict(raw)
            problem_types = row.get("problem_type") or []
            if isinstance(problem_types, str):
                problem_types = [problem_types]
            if self.problem_type_filter and not any(
                self.problem_type_filter.lower() in str(value).lower() for value in problem_types
            ):
                continue
            selected.append({**row, "_source_index": index})
            if self.max_samples is not None and len(selected) >= self.max_samples:
                break
        if not selected:
            raise ValueError("No AIME 2025 examples fit the configured filters.")
        self._rows = selected
        return selected

    def _example(self, row: dict[str, Any]) -> Example:
        index = int(row.get("_source_index") or 0)
        problem = str(row.get("problem", "")).strip()
        expected = str(row.get("answer", "")).strip()
        return Example(
            id=f"official_aime_2025_{index:03d}",
            prompt=(
                "Solve this AIME competition math problem. Return ONLY the final "
                f"integer answer, without any explanation.\n\n{problem}"
            ),
            expected=expected,
            metadata={
                "problem_idx": row.get("problem_idx"),
                "problem_types": row.get("problem_type") or [],
            },
        )


def _select_rows(
    rows: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return [rows[index] for index in indices]


def _same_integer(actual: str, expected: str) -> bool:
    try:
        return int(actual) == int(expected)
    except ValueError:
        return False


__all__ = ["AIME2025Dataset"]
