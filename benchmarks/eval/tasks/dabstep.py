"""Two frozen DABstep data-workflow tasks."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import load_hf_rows, normalized
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FROZEN_IDS = ("2536", "2769")
REVISION = "9cef9a2976ccce4d306bf220604597788b090d43"


@dataset("delegation_dabstep", tags=["delegation", "task-graph", "data"])
class DABstepTaskGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "evals/data",
        dataset_name: str = "adyen/DABstep",
        materialize_context: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.materialize_context = materialize_context
        self._rows: list[dict[str, Any]] | None = None
        self._context_dir: Path | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = {str(row.get("task_id")): row for row in self._load()}
        missing = set(FROZEN_IDS) - rows.keys()
        if missing:
            raise ValueError(f"DABstep source is missing frozen IDs: {sorted(missing)}")
        selected = [rows[source_id] for source_id in FROZEN_IDS]
        if limit is not None:
            selected = selected[:limit]
        fixture_paths = [str(self._context())] if self.materialize_context else []
        return [
            Example(
                id=f"delegation_dabstep_{problem:02d}_{source_id}",
                prompt=(
                    f"{row.get('question', '')}\n\n"
                    f"{row.get('guidelines', '')}\n\n"
                    "The benchmark data files and documentation are in the `context/` "
                    "directory. Return only the requested factoid."
                ),
                expected=str(row.get("answer", "")),
                metadata={
                    "source_id": source_id,
                    "problem": problem,
                    "level": row.get("level"),
                    "native_scorer": "dabstep_type_aware_exact",
                    "fixture_paths": fixture_paths,
                },
            )
            for source_id, row, problem in zip(FROZEN_IDS, selected, (11, 12))
        ]

    def score(self, example: Example, prediction: Prediction) -> Score:
        correct = _matches(prediction.answer, str(example.expected))
        return Score(
            value=float(correct),
            correct=correct,
            details={"expected": example.expected},
        )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = load_hf_rows(
                self.dataset_name,
                config="tasks",
                split="default",
                data_dir=self.data_dir,
                revision=REVISION,
            )
        return self._rows

    def _context(self) -> Path:
        if self._context_dir is not None:
            return self._context_dir
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("DABstep fixtures require huggingface_hub") from exc
        snapshot = Path(
            snapshot_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                allow_patterns=["data/context/*"],
                revision=REVISION,
            )
        )
        context = snapshot / "data" / "context"
        if not context.exists():
            raise FileNotFoundError(f"DABstep context was not found under {snapshot}")
        self._context_dir = context
        return context


def _matches(answer: str, expected: str) -> bool:
    actual = normalized(answer)
    gold = normalized(expected)
    try:
        return abs(Decimal(actual) - Decimal(gold)) <= Decimal("0.000001")
    except InvalidOperation:
        pass
    if "," in actual or "," in gold:
        return {normalized(item) for item in answer.split(",")} == {
            normalized(item) for item in expected.split(",")
        }
    return actual == gold


__all__ = ["DABstepTaskGraphDataset", "FROZEN_IDS", "REVISION"]
