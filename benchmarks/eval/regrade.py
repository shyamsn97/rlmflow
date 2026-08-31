"""Re-score a finished run from its saved answers, without calling a model again.

A scorer fix would otherwise require re-spending every token in a run to learn what
the corrected numbers are. `rows.jsonl` already holds each prediction, so the graders
can simply run again over it.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmarks.eval import DATASETS
from benchmarks.eval.loggers.jsonl import load_rows, write_rows
from benchmarks.eval.types import Dataset, Example, Row, Score

# Datasets register themselves on import.
importlib.import_module("benchmarks.eval.tasks")


def regrade_rows(
    rows: Sequence[Row],
    *,
    datasets: dict[str, Dataset] | None = None,
    split: str = "test",
) -> tuple[list[Row], list[dict[str, Any]]]:
    """Return the rows with fresh scores, plus one record per changed score."""
    resolved = datasets if datasets is not None else _load_datasets(rows, split=split)
    examples = _load_examples(resolved, rows, split=split)

    regraded: list[Row] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        dataset = resolved.get(row.dataset)
        example = examples.get((row.dataset, row.example_id))
        if dataset is None or example is None or row.prediction.error:
            regraded.append(row)
            continue
        try:
            score = dataset.score(example, row.prediction)
        except Exception as exc:  # noqa: BLE001 - matches the harness: a raising scorer is a zero
            score = Score(
                value=0.0,
                correct=False,
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        if score.value != row.score.value:
            changes.append(
                {
                    "dataset": row.dataset,
                    "example_id": row.example_id,
                    "runner": row.runner,
                    "seed": row.seed,
                    "answer": row.prediction.answer,
                    "was": row.score.value,
                    "now": score.value,
                }
            )
        regraded.append(replace(row, score=score))
    return regraded, changes


def _load_datasets(rows: Sequence[Row], *, split: str) -> dict[str, Dataset]:
    return {name: DATASETS.make(name) for name in sorted({row.dataset for row in rows})}


def _load_examples(
    datasets: dict[str, Dataset],
    rows: Sequence[Row],
    *,
    split: str,
) -> dict[tuple[str, str], Example]:
    seeds = sorted({row.seed for row in rows if row.seed is not None}) or [0]
    examples: dict[tuple[str, str], Example] = {}
    for name, dataset in datasets.items():
        for seed in seeds:
            for example in dataset.examples(split=split, limit=None, seed=seed):
                examples.setdefault((name, example.id), example)
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or rows.jsonl path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write regraded rows; defaults to rows-regraded.jsonl beside the input.",
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    path = args.run.expanduser()
    rows_path = path if path.suffix == ".jsonl" else path / "rows.jsonl"
    rows = load_rows(rows_path)
    if not rows:
        parser.error(f"run has no rows: {rows_path}")

    regraded, changes = regrade_rows(rows, split=args.split)
    output = args.output or rows_path.with_name("rows-regraded.jsonl")
    write_rows(output, regraded)

    print(f"{len(changes)} of {len(rows)} scores changed")
    for change in changes:
        print(
            f"  {change['example_id']} {change['runner']} s{change['seed']}: "
            f"{change['was']} -> {change['now']} ({change['answer'][:40]})"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["regrade_rows"]
