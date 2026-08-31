"""MuSiQue task-graph adapters with hidden hop annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import (
    exact_or_alias,
    parse_json_answer,
    set_f1,
    token_f1,
)
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FROZEN_ID = "4hop1__38130_8966_31714_79432"
REVISION = "a7d9f9adf6191604fc67cde318ee1a86fcf7babc"


@dataset("delegation_musique", tags=["delegation", "task-graph", "qa"])
class MuSiQueTaskGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "evals/data",
        dataset_name: str = "voidful/MuSiQue",
        split: str = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = self._load()
        answerable = [
            row
            for row in rows
            if bool(row.get("answerable", True)) and len(row.get("question_decomposition", [])) == 4
        ]
        if not answerable:
            raise ValueError("MuSiQue source has no answerable four-hop example")
        positive = answerable[0]
        paired = [
            row
            for row in rows
            if not bool(row.get("answerable", True))
            and str(row.get("question")) == str(positive.get("question"))
        ]
        negative_pool = paired or [row for row in rows if not bool(row.get("answerable", True))]
        if not negative_pool:
            raise ValueError("MuSiQue source has no unanswerable example")
        selected = [positive, negative_pool[0]]
        if limit is not None:
            selected = selected[:limit]
        return [self._example(row, index) for index, row in enumerate(selected, start=5)]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        try:
            answer = parse_json_answer(prediction.answer)
        except (TypeError, ValueError):
            answer = {"answer": prediction.answer, "supporting_paragraphs": []}
        if not isinstance(answer, dict):
            answer = {"answer": str(answer), "supporting_paragraphs": []}
        actual_answer = str(answer.get("answer", ""))
        answer_f1 = max(
            token_f1(actual_answer, candidate)
            for candidate in [expected["answer"], *expected["aliases"]]
        )
        answer_exact = exact_or_alias(
            actual_answer,
            expected["answer"],
            expected["aliases"],
        )
        support_f1 = set_f1(
            answer.get("supporting_paragraphs", []),
            expected["supporting_paragraphs"],
        )
        joint = answer_f1 * support_f1
        return Score(
            value=joint,
            correct=answer_exact and support_f1 == 1.0,
            details={
                "answer_exact": answer_exact,
                "answer_f1": answer_f1,
                "support_f1": support_f1,
                "joint_f1": joint,
            },
        )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        cache = self.data_dir / "delegation" / "musique_frozen.json"
        if cache.exists():
            self._rows = json.loads(cache.read_text())
            return self._rows
        try:
            from datasets import load_dataset  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "Install benchmark dependencies with: pip install -e '.[eval]'"
            ) from exc
        selected_by_answerable: dict[bool, dict[str, Any]] = {}
        for row in load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            revision=REVISION,
        ):
            if str(row.get("id")) == FROZEN_ID:
                answerable = bool(row.get("answerable", True))
                selected_by_answerable.setdefault(answerable, dict(row))
                if selected_by_answerable.keys() == {False, True}:
                    break
        if selected_by_answerable.keys() != {False, True}:
            raise ValueError(f"MuSiQue source is missing the frozen pair {FROZEN_ID!r}")
        selected = [selected_by_answerable[True], selected_by_answerable[False]]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(selected))
        self._rows = selected
        return self._rows

    def _example(self, row: dict[str, Any], problem: int) -> Example:
        paragraphs = row.get("paragraphs", [])
        context = "\n\n".join(
            f"[{item.get('idx', index)}] {item.get('title', '')}\n"
            f"{item.get('paragraph_text', item.get('text', ''))}"
            for index, item in enumerate(paragraphs)
        )
        decomposition = row.get("question_decomposition", [])
        support = [
            str(item.get("paragraph_support_idx"))
            for item in decomposition
            if item.get("paragraph_support_idx") is not None
        ]
        answerable = bool(row.get("answerable", True))
        expected_answer = str(row.get("answer", "")) if answerable else "UNANSWERABLE"
        return Example(
            id=f"delegation_musique_{problem:02d}_{row.get('id', problem)}",
            prompt=(
                f"{row.get('question', '')}\n\n"
                "Return JSON with `answer` and `supporting_paragraphs`, where the latter "
                "is a list of paragraph IDs. If the supplied evidence cannot answer the "
                "question, use `UNANSWERABLE`."
            ),
            context={"context": context},
            expected={
                "answer": expected_answer,
                "aliases": list(row.get("answer_aliases", [])) if answerable else [],
                "supporting_paragraphs": support,
            },
            metadata={
                "source_id": str(row.get("id", "")),
                "problem": problem,
                "answerable": answerable,
                "native_scorer": "answer_support_joint_f1",
            },
        )


__all__ = ["FROZEN_ID", "MuSiQueTaskGraphDataset", "REVISION"]
