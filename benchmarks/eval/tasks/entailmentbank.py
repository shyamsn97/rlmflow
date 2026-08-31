"""EntailmentBank proof-tree delegation problem."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import load_hf_rows, normalized, set_f1
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FROZEN_ID = "Mercury_SC_416126"
REVISION = "8c6c148f7a21c21ff037a42d9a22446c9d42debc"


@dataset("delegation_entailmentbank", tags=["delegation", "task-graph", "proof"])
class EntailmentBankTaskGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "evals/data",
        dataset_name: str = "sxiong/entailmentbank",
        split: str = "test",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = [row for row in self._load() if str(row.get("id")) == FROZEN_ID]
        if not rows:
            raise ValueError(f"EntailmentBank source is missing frozen ID {FROZEN_ID!r}")
        row = rows[0]
        return [] if limit == 0 else [self._example(row)]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        actual_steps = _proof_steps(prediction.answer)
        expected_steps = _proof_steps(expected["proof"])
        step_f1 = set_f1(actual_steps, expected_steps)
        exact = actual_steps == expected_steps
        return Score(
            value=step_f1,
            correct=exact,
            details={
                "proof_exact": exact,
                "proof_step_f1": step_f1,
                "expected_steps": len(expected_steps),
            },
        )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = load_hf_rows(
                self.dataset_name,
                config="task2",
                split=self.split,
                data_dir=self.data_dir,
                revision=REVISION,
            )
        return self._rows

    def _example(self, row: dict[str, Any]) -> Example:
        source_id = str(row.get("id", "selected"))
        return Example(
            id=f"delegation_entailmentbank_10_{source_id}",
            prompt=(
                "Construct a valid entailment proof for the hypothesis below using only "
                "the supplied sentence IDs. Return only semicolon-separated proof steps "
                "in EntailmentBank form (`sent1 & sent2 -> int1: conclusion; ...`).\n\n"
                f"Hypothesis: {row.get('hypothesis', '')}"
            ),
            context={"context": str(row.get("context", ""))},
            expected={"proof": str(row.get("proof", ""))},
            metadata={
                "source_id": source_id,
                "problem": 10,
                "proof_depth": int(row.get("depth_of_proof", 0)),
                "proof_length": int(row.get("length_of_proof", 0)),
                "native_scorer": "proof_step_f1",
            },
        )


def _proof_steps(value: str) -> set[str]:
    text = re.sub(r"```.*?```", "", value, flags=re.DOTALL)
    return {normalized(step) for step in text.split(";") if normalized(step)}


__all__ = ["EntailmentBankTaskGraphDataset", "FROZEN_ID", "REVISION"]
