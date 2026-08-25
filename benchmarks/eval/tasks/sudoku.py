"""Sudoku benchmark dataset adapters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("official_sudoku_extreme", tags=["official", "reasoning"])
class SudokuExtremeDataset(Dataset):
    """Sudoku Extreme puzzles from sapientinc/sudoku-extreme."""

    dataset_name = "sapientinc/sudoku-extreme"

    def __init__(
        self,
        data_dir: str = "evals/data",
        max_samples: int | None = 50,
        min_rating: int | None = None,
        max_rating: int | None = None,
        sample_window: int = 4096,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.sample_window = sample_window
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        rows = _select_rows(self._load(split), limit=limit, seed=seed)
        return [self._example(row) for row in rows]

    def score(self, example: Example, prediction: Prediction) -> Score:
        digits = re.sub(r"[^1-9]", "", prediction.answer)
        candidate = digits[:81]
        expected = str(example.expected or "")
        correct = len(candidate) == 81 and candidate == expected
        matching = sum(1 for actual, wanted in zip(candidate, expected) if actual == wanted)
        return Score(
            value=matching / 81.0 if expected else 0.0,
            correct=correct,
            details={"expected": expected, "matching_digits": matching},
        )

    def _load(self, split: str) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        local = self.data_dir / "sudoku_extreme" / f"{split}.parquet"
        if local.exists():
            import pandas as pd

            rows = pd.read_parquet(local).to_dict("records")
        else:
            try:
                from datasets import load_dataset
                from datasets.utils.logging import disable_progress_bar
            except ImportError as exc:
                raise RuntimeError(
                    "Sudoku Extreme requires the eval extra: pip install -e '.[eval]'"
                ) from exc
            disable_progress_bar()
            stream = load_dataset(self.dataset_name, split=split, streaming=True)
            rows = [dict(row) for _, row in zip(range(self.sample_window), stream)]
        filtered = [row for row in rows if self._matches_rating(row)]
        if self.max_samples:
            filtered = filtered[: self.max_samples]
        if not filtered:
            raise ValueError("No Sudoku Extreme examples fit the configured limits.")
        self._rows = [{**row, "_source_index": index} for index, row in enumerate(filtered)]
        return self._rows

    def _matches_rating(self, row: dict[str, Any]) -> bool:
        rating = row.get("rating")
        if (
            self.min_rating is not None
            and isinstance(rating, (int, float))
            and rating < self.min_rating
        ):
            return False
        if (
            self.max_rating is not None
            and isinstance(rating, (int, float))
            and rating > self.max_rating
        ):
            return False
        return True

    def _example(self, row: dict[str, Any]) -> Example:
        puzzle = str(row.get("question") or row.get("puzzle") or "").strip()
        answer = str(row.get("answer") or row.get("solution") or "").strip()
        index = int(row.get("_source_index") or 0)
        prompt = (
            "Solve this Sudoku puzzle. Each row, column, and 3x3 box must contain "
            f"digits 1-9 exactly once.\n\n{_format_sudoku_grid(puzzle)}\n\n"
            f"Raw puzzle: {puzzle}\n\n"
            "Return ONLY the completed 81-digit solution string (no spaces, no newlines)."
        )
        return Example(
            id=f"official_sudoku_extreme_{index:05d}",
            prompt=prompt,
            expected=answer,
            metadata={
                "source": row.get("source"),
                "rating": row.get("rating"),
                "puzzle": puzzle,
            },
        )


def _select_rows(
    rows: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    import random

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return [rows[index] for index in indices]


def _format_sudoku_grid(puzzle: str) -> str:
    lines = []
    for row_index in range(9):
        row = puzzle[row_index * 9 : (row_index + 1) * 9]
        cells = [(row[col] if col < len(row) and row[col] != "." else "_") for col in range(9)]
        lines.append(
            " ".join(cells[0:3]) + " | " + " ".join(cells[3:6]) + " | " + " ".join(cells[6:9])
        )
        if row_index in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)


__all__ = ["SudokuExtremeDataset"]
