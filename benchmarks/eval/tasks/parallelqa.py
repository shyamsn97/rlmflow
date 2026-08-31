"""Four frozen ParallelQA dependency-graph problems."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import exact_or_alias
from benchmarks.eval.types import Dataset, Example, Prediction, Score

SOURCE_URL = (
    "https://raw.githubusercontent.com/SqueezeAILab/LLMCompiler/"
    "dca77b2197f89a20e505545a1458ae4a2ff04571/datasets/parallelqa_dataset.json"
)
FROZEN_IDS = ("11", "22", "63", "94")
FROZEN_FACTS = {
    "Mariana Trench": 10984.869565217392,
    "Puerto Rico Trench": 8376.0,
    "Sunda Trench": 7288.211857707474,
    "South Sandwich Trench": 8202.952766798458,
    "Peru-Chile Trench": 8065.0,
}


@dataset("delegation_parallelqa", tags=["delegation", "task-graph", "qa"])
class ParallelQATaskGraphDataset(Dataset):
    def __init__(self, data_dir: str = "evals/data", source_url: str = SOURCE_URL) -> None:
        self.path = Path(data_dir) / "delegation" / "parallelqa_dataset.json"
        self.source_url = source_url
        self._rows: list[dict] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = {str(row["id"]): row for row in self._load()}
        selected = [rows[source_id] for source_id in FROZEN_IDS]
        if limit is not None:
            selected = selected[:limit]
        return [
            Example(
                id=f"delegation_parallelqa_{row['id']}",
                prompt=(f"{row['question']}\n\nReturn only the requested number or entity name."),
                expected=row["answer"],
                metadata={
                    "source_id": str(row["id"]),
                    "branch": int(row["branch"]),
                    "native_scorer": "normalized_exact",
                    "tool": "fact_lookup",
                    "facts": FROZEN_FACTS,
                },
            )
            for row in selected
        ]

    def score(self, example: Example, prediction: Prediction) -> Score:
        correct = exact_or_alias(prediction.answer, example.expected)
        return Score(
            value=float(correct),
            correct=correct,
            details={"expected": example.expected},
        )

    def _load(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(self.source_url, timeout=30) as response:
                self.path.write_bytes(response.read())
        self._rows = json.loads(self.path.read_text())
        missing = set(FROZEN_IDS) - {str(row.get("id")) for row in self._rows}
        if missing:
            raise ValueError(f"ParallelQA source is missing frozen IDs: {sorted(missing)}")
        return self._rows


__all__ = ["FROZEN_FACTS", "FROZEN_IDS", "ParallelQATaskGraphDataset"]
