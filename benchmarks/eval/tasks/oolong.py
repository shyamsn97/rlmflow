"""OOLONG benchmark dataset."""

from __future__ import annotations

import ast
import random
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


class OolongDataset(Dataset):
    """Load OOLONG examples from Hugging Face or a local `evals/data/oolong` copy."""

    dataset_name = "oolongbench/oolong-synth"
    dataset_revision = "f0d59eaf0febf130664cfceb710436c8e3216b2b"

    def __init__(
        self,
        data_dir: str = "evals/data",
        max_context_chars: int | None = None,
        max_context_tokens: int | None = None,
        max_samples: int | None = 50,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        self.max_samples = max_samples
        self._dataset: Any | None = None
        self._eligible: list[int] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        ds = self._load(split)
        eligible = self._eligible_indices(ds)
        if not eligible:
            raise ValueError("No OOLONG examples fit the configured limits.")
        positions = list(range(len(eligible)))
        random.Random(seed).shuffle(positions)
        selected = positions if limit is None else positions[: min(limit, len(positions))]
        # Only the sampled rows are pulled out of the memory-mapped dataset;
        # the full split (~20 GB of text for `test`) is never held in RAM.
        return [self._example(dict(ds[eligible[pos]]), index=pos) for pos in selected]

    def _load(self, split: str):
        if self._dataset is not None:
            return self._dataset
        try:
            from datasets import (  # pyright: ignore[reportMissingImports]
                load_dataset,
                load_from_disk,
            )
        except ImportError as exc:
            raise RuntimeError("OOLONG requires the eval extra: pip install -e '.[eval]'") from exc

        local = self.data_dir / "oolong"
        if local.exists():
            ds = load_from_disk(str(local))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
        else:
            ds = load_dataset(self.dataset_name, split=split, revision=self.dataset_revision)

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
            if self.max_samples is not None and len(eligible) >= self.max_samples:
                break
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
        question = str(row.get("question", "")).strip()
        return Example(
            id=f"oolong_{index:05d}",
            prompt=(
                f"{question}\n\n"
                "Parse the canonical observation JSON. Its `context` field contains "
                "the complete source text to use as evidence."
            ),
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
        best = max(
            (
                _synth_score(
                    expected,
                    prediction.answer,
                    answer_type=str(example.metadata.get("answer_type") or ""),
                )
                for expected in example.expected or []
            ),
            default=0.0,
        )
        return Score(
            value=best,
            correct=best >= 1.0,
            details={
                "expected": example.expected,
                "answer_type": example.metadata.get("answer_type"),
                "scorer": "alexzhang13/rlm:oolong-synth@d04208a",
            },
        )


@dataset("oolong-yahoo-128k", tags=["long-context", "paper-aligned"])
class OolongYahoo128KDataset(OolongDataset):
    """Deterministic Yahoo aggregation rows at the paper's 128K setting."""

    context_len = 131_072
    pilot_size = 2

    def __init__(self, data_dir: str = "evals/data") -> None:
        super().__init__(data_dir=data_dir, max_samples=None)

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del seed
        if split not in {"test", "pilot", "full"}:
            raise ValueError("OOLONG Yahoo exposes only pilot and full suite splits")
        ds = self._load("test")
        indices = self._eligible_indices(ds)
        if not indices:
            raise ValueError("No Yahoo OOLONG rows exist at context_len=131072")
        selected = indices[: self.pilot_size] if split in {"test", "pilot"} else indices
        if limit is not None:
            selected = selected[:limit]
        return [self._example(dict(ds[index]), index=index) for index in selected]

    def _eligible_indices(self, ds) -> list[int]:
        if self._eligible is not None:
            return self._eligible
        datasets = ds["dataset"]
        context_lengths = ds["context_len"]
        self._eligible = [
            index
            for index, (source, context_len) in enumerate(
                zip(datasets, context_lengths, strict=True)
            )
            if "yahoo" in str(source).casefold() and int(context_len or 0) == self.context_len
        ]
        return self._eligible


@dataset("oolong-trec-coarse-128k", tags=["long-context", "upstream-aligned"])
class OolongTrecCoarse128KDataset(OolongDataset):
    """The fixed 25-row upstream RLM training evaluation at 128K."""

    context_len = 131_072
    full_size = 25
    pilot_size = 2
    selection_seed = 42

    def __init__(self, data_dir: str = "evals/data") -> None:
        super().__init__(data_dir=data_dir, max_samples=None)
        self._selected_rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del seed
        if split not in {"test", "pilot", "full"}:
            raise ValueError("OOLONG TREC-coarse exposes only pilot and full suite splits")
        rows = self._rows()
        selected = rows[: self.pilot_size] if split in {"test", "pilot"} else rows
        if limit is not None:
            selected = selected[:limit]
        return [
            self._example(row, index=int(row.get("id", position)))
            for position, row in enumerate(selected)
        ]

    def _rows(self) -> list[dict[str, Any]]:
        if self._selected_rows is not None:
            return self._selected_rows
        local = self.data_dir / "oolong"
        if local.exists():
            validation = self._load("validation")
            shuffled = validation.shuffle(seed=self.selection_seed)
            rows = []
            for row in shuffled:
                if self._keep(row):
                    rows.append(dict(row))
                if len(rows) >= self.full_size:
                    break
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "OOLONG requires the eval extra: pip install -e '.[eval]'"
                ) from exc
            stream = load_dataset(
                self.dataset_name,
                split="validation",
                revision=self.dataset_revision,
                streaming=True,
            ).shuffle(seed=self.selection_seed, buffer_size=10_000)
            rows = []
            for row in stream:
                if self._keep(row):
                    rows.append(dict(row))
                if len(rows) >= self.full_size:
                    break
        if len(rows) < self.full_size:
            raise ValueError(f"Expected {self.full_size} TREC-coarse 128K rows, found {len(rows)}.")
        self._selected_rows = rows[: self.full_size]
        return self._selected_rows

    def _keep(self, row: dict[str, Any]) -> bool:
        return (
            row.get("dataset") == "trec_coarse"
            and int(row.get("context_len") or 0) == self.context_len
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


COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")


def _find_comparison_phrase(output: str) -> str | None:
    lowered = output.lower()
    hits = [(lowered.rfind(phrase), phrase) for phrase in COMPARISON_PHRASES if phrase in lowered]
    return max(hits)[1] if hits else None


def _attempt_answer_parse(answer: str) -> str:
    comparison = _find_comparison_phrase(answer)
    if comparison is not None:
        return comparison
    if ":" not in answer:
        return answer if len(answer) < 20 else answer.split()[-1]
    candidate = answer.split(":")[-1].strip()
    return candidate.replace("*", "").replace("[", "").replace("]", "")


def _synth_score(expected: Any, output: str, *, answer_type: str = "") -> float:
    gold = _parse_gold(expected)
    trimmed = _attempt_answer_parse(output)
    gold_text = str(gold)
    if trimmed == gold_text or trimmed.lower() == gold_text.lower():
        return 1.0

    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except (TypeError, ValueError):
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        parsed = _parse_date(trimmed)
        gold_date = _parse_date(gold)
        return 1.0 if parsed is not None and parsed == gold_date else 0.0
    if gold_text and gold_text.casefold() not in COMPARISON_PHRASES:
        return 1.0 if gold_text.casefold() in output.casefold() else 0.0
    return 0.0


def _parse_gold(value: Any) -> Any:
    text = str(value).strip()
    date_match = re.fullmatch(
        r"\[?datetime\.date\(\s*(\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\s*\)\]?",
        text,
    )
    if date_match:
        return date(*(int(part) for part in date_match.groups()))
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value
    if isinstance(parsed, (list, tuple)) and parsed:
        return parsed[0]
    return parsed


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        from dateutil import parser as date_parser

        return date_parser.parse(text).date()
    except (ImportError, OverflowError, TypeError, ValueError):
        pass
    for candidate in (text, text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate).date()
        except ValueError:
            pass
    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if match:
        try:
            return date(*(int(part) for part in match.groups()))
        except ValueError:
            return None
    return None


__all__ = [
    "OolongDataset",
    "OolongTrecCoarse128KDataset",
    "OolongYahoo128KDataset",
]
