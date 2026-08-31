"""One deterministic LongBench-v2 CodeQA task-graph problem."""

from __future__ import annotations

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import load_hf_rows
from benchmarks.eval.tasks.longbench import CodeQADataset
from benchmarks.eval.types import Example

FROZEN_INDEX = 72
REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"


@dataset("delegation_codeqa", tags=["delegation", "task-graph", "code"])
class DelegationCodeQADataset(CodeQADataset):
    def _load(self, split: str) -> list[dict]:
        if self._rows is None:
            source = load_hf_rows(
                self.dataset_name,
                split=split,
                data_dir=self.data_dir,
                revision=REVISION,
            )
            self._rows = [
                {**row, "_source_index": index}
                for index, row in enumerate(source)
                if self._include_row(row)
            ]
        return self._rows

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = [
            row
            for row in self._load(self.split)
            if int(row.get("_source_index", -1)) == FROZEN_INDEX
        ]
        if not rows:
            raise ValueError(f"LongBench-v2 CodeQA is missing frozen index {FROZEN_INDEX}")
        row = rows[0]
        example = self._example(row)
        return (
            []
            if limit == 0
            else [
                Example(
                    id=f"delegation_codeqa_17_{example.id}",
                    prompt=example.prompt,
                    context=example.context,
                    expected=example.expected,
                    metadata={
                        **example.metadata,
                        "source_id": f"train:{FROZEN_INDEX}",
                        "problem": 17,
                        "native_scorer": "longbench_exact_choice",
                    },
                )
            ]
        )


__all__ = ["DelegationCodeQADataset", "FROZEN_INDEX", "REVISION"]
