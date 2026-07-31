"""OOLONG benchmark dataset."""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("oolong", tags=["long-context"])
class OolongDataset(Dataset):
    """Load OOLONG examples from Hugging Face or a local `evals/data/oolong` copy."""

    dataset_name = "oolongbench/oolong-synth"

    def __init__(
        self,
        data_dir: str = "evals/data",
        max_context_chars: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        self._dataset: Any | None = None
        self._eligible: list[int] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        ds = self._load(split)
        eligible = self._eligible_indices(ds)
        if not eligible:
            raise ValueError("No OOLONG examples fit the configured limits.")
        count = limit or 1
        positions = list(range(len(eligible)))
        random.Random(seed).shuffle(positions)
        selected = positions[: min(count, len(positions))]
        # Only the sampled rows are pulled out of the memory-mapped dataset;
        # the full split (~20 GB of text for `test`) is never held in RAM.
        return [self._example(dict(ds[eligible[pos]]), index=pos) for pos in selected]

    def _load(self, split: str):
        if self._dataset is not None:
            return self._dataset
        try:
            from datasets import load_dataset, load_from_disk  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError("OOLONG requires the eval extra: pip install -e '.[eval]'") from exc

        local = self.data_dir / "oolong"
        if local.exists():
            ds = load_from_disk(str(local))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
        else:
            ds = load_dataset(self.dataset_name, split=split)

        self._dataset = ds
        return ds

    def _eligible_indices(self, ds) -> list[int]:
        """Indices that satisfy the configured context limits.

        Computed against the memory-mapped HF dataset so that, in the common
        no-limit case, we never decode a single context string here.
        """
        if self._eligible is not None:
            return self._eligible

        n = ds.num_rows
        if self.max_context_tokens is None and self.max_context_chars is None:
            self._eligible = list(range(n))
            return self._eligible

        # `context_len` is a cheap int column; load it once when token-limiting.
        context_lens = ds["context_len"] if self.max_context_tokens is not None else None
        eligible: list[int] = []
        for index in range(n):
            if self.max_context_tokens is not None:
                context_len = context_lens[index] if context_lens is not None else None
                if isinstance(context_len, (int, float)):
                    if context_len > self.max_context_tokens:
                        continue
                elif self._context_chars(ds, index) // 4 > self.max_context_tokens:
                    continue
            if (
                self.max_context_chars is not None
                and self._context_chars(ds, index) > self.max_context_chars
            ):
                continue
            eligible.append(index)
        self._eligible = eligible
        return eligible

    @staticmethod
    def _context_chars(ds, index: int) -> int:
        row = ds[index]
        return len(
            str(row.get("context_window_text_with_labels") or row.get("context_window_text") or "")
        )

    def _example(self, row: dict[str, Any], *, index: int) -> Example:
        context = str(
            row.get("context_window_text_with_labels") or row.get("context_window_text") or ""
        )
        answers = _normalize_answers(row.get("answer"))
        answer_type = row.get("answer_type")
        return Example(
            id=f"oolong_{index:05d}",
            prompt=str(row.get("question", "")).strip(),
            context={"context": context},
            expected=answers,
            metadata={
                "dataset": row.get("dataset"),
                "context_len": row.get("context_len"),
                "answer_type": answer_type,
                "context_window_id": row.get("context_window_id"),
            },
        )

    def score(self, example: Example, prediction: Prediction) -> Score:
        answer = prediction.answer.lower()
        best = 0.0
        matched = None
        for expected in example.expected or []:
            candidate = str(expected).strip()
            if not candidate:
                continue
            if _is_numeric(candidate):
                value = _extract_number(answer)
                if value is not None:
                    score = 1.0 if abs(value - float(candidate)) < 1e-6 else 0.0
                    if score > best:
                        best, matched = score, candidate
            elif re.search(r"\b" + re.escape(candidate.lower()) + r"\b", answer):
                best, matched = 1.0, candidate
                break
        return Score(
            value=best,
            correct=best >= 1.0,
            details={"matched": matched, "expected": example.expected},
        )


def _normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            out.extend(_normalize_answers(item))
        return out
    if isinstance(raw, dict):
        values = raw.get("answers") or raw.get("answer") or raw.values()
        return _normalize_answers(list(values) if not isinstance(values, str) else values)
    text = str(raw).strip()
    if not text:
        return []
    # HF rows often store answers as a Python-literal repr, e.g. "['incorrect']"
    # or "[48]". Parse those so we match the value, not the bracketed string.
    if text[0] in "[(" and text[-1] in ")]":
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _normalize_answers(list(parsed))
        if parsed is not None:
            return [str(parsed).strip()]
    return [part.strip() for part in text.split("||") if part.strip()]


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _extract_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


__all__ = ["OolongDataset"]
