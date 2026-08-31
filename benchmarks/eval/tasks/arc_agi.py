"""Two frozen ARC-AGI-2 compositional transformation problems."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import parse_json_answer
from benchmarks.eval.types import Dataset, Example, Prediction, Score

REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
FROZEN_IDS = ("1ae2feb7", "2ba387bc")


@dataset("delegation_arc_agi", tags=["delegation", "task-graph", "reasoning"])
class ArcAgiTaskGraphDataset(Dataset):
    def __init__(self, data_dir: str = "evals/data") -> None:
        self.data_dir = Path(data_dir) / "delegation" / "arc_agi_2"

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        examples = [
            self._example(source_id, problem) for source_id, problem in zip(FROZEN_IDS, (19, 20))
        ]
        return examples if limit is None else examples[:limit]

    def score(self, example: Example, prediction: Prediction) -> Score:
        try:
            actual = parse_json_answer(prediction.answer)
        except (TypeError, ValueError):
            actual = None
        expected = example.expected
        if (
            isinstance(actual, list)
            and len(expected) == 1
            and actual
            and isinstance(actual[0], list)
            and actual[0]
            and isinstance(actual[0][0], int)
        ):
            actual = [actual]
        exact = actual == expected
        return Score(
            value=float(exact),
            correct=exact,
            details={"exact": exact, "test_outputs": len(expected)},
        )

    def _example(self, source_id: str, problem: int) -> Example:
        task = self._load(source_id)
        public_task = {
            "train": task["train"],
            "test": [{"input": item["input"]} for item in task["test"]],
        }
        expected = [item["output"] for item in task["test"]]
        return Example(
            id=f"delegation_arc_agi_{problem:02d}_{source_id}",
            prompt=(
                "Infer the transformation from every training pair in `INPUTS['task']` "
                "and apply it to each test input. Return only a JSON list containing one "
                "output grid per test input."
            ),
            context={"task": json.dumps(public_task, separators=(",", ":"))},
            expected=expected,
            metadata={
                "source_id": source_id,
                "problem": problem,
                "native_scorer": "arc_exact_grid",
            },
        )

    def _load(self, source_id: str) -> dict:
        path = self.data_dir / f"{source_id}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/"
                f"{REVISION}/data/evaluation/{source_id}.json"
            )
            with urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        return json.loads(path.read_text())


__all__ = ["ArcAgiTaskGraphDataset", "FROZEN_IDS"]
