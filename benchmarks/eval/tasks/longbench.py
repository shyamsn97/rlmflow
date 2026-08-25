"""LongBench-family benchmark dataset adapters.

These are small ports of the corresponding tasks in avilum/minrlm's eval
harness, adapted to this repo's Dataset/Example/Score interface.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("official_longbench_v2", tags=["official", "long-context"])
class LongBenchV2Dataset(Dataset):
    """LongBench-v2 across all domains."""

    dataset_name = "zai-org/LongBench-v2"
    id_prefix = "official_longbench_v2"

    def __init__(
        self,
        data_dir: str = "evals/data",
        split: str = "train",
        max_context_tokens: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_context_tokens = max_context_tokens
        self.max_samples = max_samples
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        rows = _select_rows(self._load(self.split), limit=limit, seed=seed)
        return [self._example(row) for row in rows]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = str(example.expected or "").strip()
        answer = prediction.answer.strip()
        if expected in {"A", "B", "C", "D"}:
            match = re.search(r"\b([ABCD])\b", answer.upper())
            correct = bool(match and match.group(1) == expected)
        else:
            candidates = [item.strip() for item in expected.split("||") if item.strip()]
            answer_lower = answer.lower()
            correct = any(candidate.lower() in answer_lower for candidate in candidates)
        return Score(
            value=1.0 if correct else 0.0,
            correct=correct,
            details={"expected": expected},
        )

    def _load(self, split: str) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(_load_hf_rows(self.dataset_name, split, self.data_dir)):
            context = str(row.get("context", ""))
            if not self._include_row(row):
                continue
            if self.max_context_tokens is not None and len(context) // 4 > self.max_context_tokens:
                continue
            rows.append({**dict(row), "_source_index": index})
            if self.max_samples and len(rows) >= self.max_samples:
                break
        if not rows:
            raise ValueError("No LongBench-v2 examples fit the configured limits.")
        self._rows = rows
        return rows

    def _include_row(self, row: dict[str, Any]) -> bool:
        return True

    def _example(self, row: dict[str, Any]) -> Example:
        question = str(row.get("question", "")).strip()
        choices = {
            "A": str(row.get("choice_A", "")).strip(),
            "B": str(row.get("choice_B", "")).strip(),
            "C": str(row.get("choice_C", "")).strip(),
            "D": str(row.get("choice_D", "")).strip(),
        }
        if any(choices.values()):
            prompt = (
                f"{question}\n\nChoices:\n"
                f"A) {choices['A']}\n"
                f"B) {choices['B']}\n"
                f"C) {choices['C']}\n"
                f"D) {choices['D']}\n\n"
                "Return ONLY the letter (A, B, C, or D)."
            )
            expected = str(row.get("answer", "")).strip().upper()
        else:
            prompt = f"{question}\n\nReturn a concise answer."
            expected = "||".join(_normalize_answers(row.get("answers") or row.get("answer")))
        index = int(row.get("_source_index") or 0)
        return Example(
            id=f"{self.id_prefix}_{index:05d}",
            prompt=prompt,
            context={"context": str(row.get("context", ""))},
            expected=expected,
            metadata={
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "context_chars": len(str(row.get("context", ""))),
            },
        )


@dataset("official_codeqa", tags=["rlm-comparison", "long-context", "code"])
class CodeQADataset(LongBenchV2Dataset):
    """The paper-aligned code-repository subset of LongBench-v2."""

    id_prefix = "official_codeqa"

    def _include_row(self, row: dict[str, Any]) -> bool:
        return (
            row.get("domain") == "Code Repository Understanding"
            or row.get("sub_domain") == "Code repo QA"
        )


def _load_hf_rows(dataset_name: str, split: str, data_dir: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset, load_from_disk  # pyright: ignore[reportMissingImports]
        from datasets.utils.logging import (  # pyright: ignore[reportMissingImports]
            disable_progress_bar,
        )
    except ImportError as exc:
        raise RuntimeError(
            "These benchmarks require the eval extra: pip install -e '.[eval]'"
        ) from exc
    disable_progress_bar()
    local_names = [dataset_name.split("/", 1)[-1], dataset_name.lower().replace("/", "_")]
    for name in local_names:
        local = data_dir / name
        if local.exists():
            ds = load_from_disk(str(local))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
            return [dict(row) for row in ds]
    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def _select_rows(
    rows: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    return [rows[index] for index in indices]


def _normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict):
        for key in ("answer", "answers", "gold", "target", "label"):
            if key in raw:
                return _normalize_answers(raw[key])
        return []
    text = str(raw).strip()
    if "||" in text:
        return [item.strip() for item in text.split("||") if item.strip()]
    return [text] if text else []


__all__ = ["CodeQADataset", "LongBenchV2Dataset"]
