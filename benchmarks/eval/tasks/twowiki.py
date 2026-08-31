"""Two frozen 2WikiMultiHopQA graph shapes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import (
    exact_or_alias,
    load_hf_rows,
    parse_json_answer,
    set_f1,
    token_f1,
)
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FROZEN_IDS = {
    "inference": "948c33ea0baf11ebab90acde48001122",
    "bridge_comparison": "d594f50208c111ebbd8bac1f6bf848b6",
}
REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"


@dataset("delegation_twowiki", tags=["delegation", "task-graph", "qa"])
class TwoWikiTaskGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "evals/data",
        dataset_name: str = "framolfese/2WikiMultihopQA",
        split: str = "validation",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        rows = self._load()
        selected = []
        for kind, source_id in FROZEN_IDS.items():
            candidates = [
                row
                for row in rows
                if str(row.get("type")) == kind
                and str(row.get("id") or row.get("_id")) == source_id
            ]
            if not candidates:
                raise ValueError(f"2WikiMultiHopQA source is missing frozen ID {source_id!r}")
            selected.append(candidates[0])
        if limit is not None:
            selected = selected[:limit]
        return [self._example(row, problem) for row, problem in zip(selected, (7, 8))]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        try:
            parsed = parse_json_answer(prediction.answer)
        except (TypeError, ValueError):
            parsed = {"answer": prediction.answer}
        if not isinstance(parsed, dict):
            parsed = {"answer": str(parsed)}
        answer = str(parsed.get("answer", ""))
        answer_exact = exact_or_alias(answer, expected["answer"])
        answer_f1 = token_f1(answer, expected["answer"])
        support_f1 = set_f1(parsed.get("supporting_facts", []), expected["supporting_facts"])
        evidence_f1 = set_f1(parsed.get("evidence", []), expected["evidence"])
        joint = answer_f1 * support_f1 * evidence_f1
        return Score(
            value=joint,
            correct=answer_exact and support_f1 == 1.0 and evidence_f1 == 1.0,
            details={
                "answer_exact": answer_exact,
                "answer_f1": answer_f1,
                "support_f1": support_f1,
                "evidence_f1": evidence_f1,
                "joint_f1": joint,
            },
        )

    def _load(self) -> list[dict[str, Any]]:
        if self._rows is None:
            self._rows = load_hf_rows(
                self.dataset_name,
                split=self.split,
                data_dir=self.data_dir,
                revision=REVISION,
            )
        return self._rows

    def _example(self, row: dict[str, Any], problem: int) -> Example:
        contexts = _contexts(row.get("context", []))
        context = "\n\n".join(
            f"[{index}] {title}\n{''.join(sentences)}"
            for index, (title, sentences) in enumerate(contexts)
        )
        support = _supporting_facts(row.get("supporting_facts", []))
        evidence = _evidence(row.get("evidences", []))
        source_id = str(row.get("id") or row.get("_id") or problem)
        return Example(
            id=f"delegation_twowiki_{problem:02d}_{source_id}",
            prompt=(
                f"{row.get('question', '')}\n\n"
                "Return JSON with `answer`, `supporting_facts`, and `evidence`. "
                "Supporting facts must be `Title:sentence_index`; evidence entries "
                "must be `subject|relation|object`."
            ),
            context={"context": context},
            expected={
                "answer": str(row.get("answer", "")),
                "supporting_facts": support,
                "evidence": evidence,
            },
            metadata={
                "source_id": source_id,
                "problem": problem,
                "reasoning_type": row.get("type"),
                "native_scorer": "answer_support_evidence_joint_f1",
            },
        )


def _contexts(value: Any) -> list[tuple[str, list[str]]]:
    if isinstance(value, dict):
        titles = value.get("title", [])
        sentences = value.get("sentences", value.get("content", []))
        return [(str(title), list(parts)) for title, parts in zip(titles, sentences)]
    contexts = []
    for item in value:
        if isinstance(item, dict):
            contexts.append(
                (str(item.get("title", "")), list(item.get("content", item.get("sentences", []))))
            )
        else:
            contexts.append((str(item[0]), list(item[1])))
    return contexts


def _supporting_facts(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{title}:{index}"
            for title, index in zip(value.get("title", []), value.get("sent_id", []))
        ]
    return [
        f"{item.get('title')}:{item.get('sent_id')}"
        if isinstance(item, dict)
        else f"{item[0]}:{item[1]}"
        for item in value
    ]


def _evidence(value: Any) -> list[str]:
    return [
        "|".join(str(item.get(key, "")) for key in ("fact", "relation", "entity"))
        if isinstance(item, dict)
        else "|".join(str(part) for part in item)
        for item in value
    ]


__all__ = ["FROZEN_IDS", "REVISION", "TwoWikiTaskGraphDataset"]
