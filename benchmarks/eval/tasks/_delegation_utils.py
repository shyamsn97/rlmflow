"""Shared loading and deterministic scoring helpers for task-graph adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def load_hf_rows(
    dataset_name: str,
    *,
    split: str,
    data_dir: str | Path,
    config: str | None = None,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset, load_from_disk  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("Install benchmark dependencies with: pip install -e '.[eval]'") from exc

    root = Path(data_dir)
    local = root / dataset_name.replace("/", "__")
    if config:
        local = local / config
    if revision:
        local = local / revision
    if local.exists():
        loaded = load_from_disk(str(local))
        rows = loaded[split] if hasattr(loaded, "keys") and split in loaded else loaded
    else:
        rows = load_dataset(dataset_name, config, split=split, revision=revision)
    return [dict(row) for row in rows]


def stable_rows(rows: Iterable[dict[str, Any]], *, key: str = "id") -> list[dict[str, Any]]:
    def row_key(row: dict[str, Any]) -> str:
        value = row.get(key) or row.get("_id") or row.get("task_id") or row
        return hashlib.sha256(str(value).encode()).hexdigest()

    return sorted((dict(row) for row in rows), key=row_key)


def normalized(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def answer_aliases(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def exact_or_alias(answer: str, expected: Any, aliases: Any = None) -> bool:
    candidates = [*answer_aliases(expected), *answer_aliases(aliases)]
    answer_number = _finite_decimal(answer)
    return any(
        normalized(answer) == normalized(candidate)
        or (
            answer_number is not None
            and (candidate_number := _finite_decimal(candidate)) is not None
            and matches_at_stated_precision(answer_number, candidate_number)
        )
        for candidate in candidates
    )


def matches_at_stated_precision(answer: Decimal, expected: Decimal) -> bool:
    """Compare two numbers at the precision the expected value states.

    An answer key of `9810.79` claims two decimals, so requiring more of an answer
    grades `str(float)` round-tripping rather than arithmetic: `gpt-5-mini` returned
    `9810.79000000000126` and was marked wrong. Comparison never goes coarser than
    two decimals, so an integer key still rejects `3.4`.
    """
    if answer == expected:
        return True
    quantum = Decimal(1).scaleb(min(expected.as_tuple().exponent, -2))
    try:
        return answer.quantize(quantum, rounding=ROUND_HALF_UP) == expected.quantize(
            quantum, rounding=ROUND_HALF_UP
        )
    except InvalidOperation:
        return False


def _finite_decimal(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def token_f1(answer: str, expected: str) -> float:
    actual = normalized(answer).split()
    gold = normalized(expected).split()
    if not actual or not gold:
        return float(actual == gold)
    overlap = sum((Counter(actual) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def set_f1(actual: Iterable[Any], expected: Iterable[Any]) -> float:
    actual_set = {str(item) for item in actual}
    expected_set = {str(item) for item in expected}
    if not actual_set or not expected_set:
        return float(actual_set == expected_set)
    overlap = len(actual_set & expected_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_set)
    recall = overlap / len(expected_set)
    return 2 * precision * recall / (precision + recall)


def parse_json_answer(answer: str) -> Any:
    text = answer.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((index for index in (text.find("["), text.find("{")) if index >= 0), default=-1)
        if start < 0:
            raise
        return json.loads(text[start:])


__all__ = [
    "answer_aliases",
    "exact_or_alias",
    "load_hf_rows",
    "normalized",
    "parse_json_answer",
    "set_f1",
    "stable_rows",
    "token_f1",
]
