"""Long-document QA: OOLONG ``trec_coarse`` at 128K tokens.

The first 15 rows of the fixed, seeded 25-row set the RLM authors evaluate on
(``OolongTrecCoarse128KDataset``), scored with their scorer. The document is a
file in the work directory, the same for every system. It's the text without
labels: with them, every question is a grep.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmarks.table.systems import Limits, Outcome

NAME = "oolong"
SIZE = 15
CACHE = Path.home() / ".cache" / "rlmflow-bench" / "oolong-trec-coarse-128k.json"
DATASET = "oolongbench/oolong-synth"
REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
CONTEXT_LEN = 131_072
SELECTION_SEED = 42

TASK = """\
{question}

The source text is in `context.txt` in this directory. Write only your final
answer to `answer.txt`."""


def examples() -> dict[str, dict]:
    if not CACHE.exists():
        rows = trec_coarse_rows()[:SIZE]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(
            json.dumps(
                {
                    f"oolong_{int(row['id']):05d}": {
                        "question": str(row["question"]).strip(),
                        "context": row["context_window_text"],
                        "expected": normalize_answers(row.get("answer")),
                        "answer_type": row.get("answer_type"),
                    }
                    for row in rows
                }
            )
        )
    return json.loads(CACHE.read_text())


def trec_coarse_rows(count: int = 25) -> list[dict[str, Any]]:
    """The RLM authors' fixed set: ``trec_coarse`` rows at 128K from a seeded shuffle of validation."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("OOLONG needs `datasets`: pip install datasets") from exc
    stream = load_dataset(DATASET, split="validation", revision=REVISION, streaming=True)
    rows = []
    for row in stream.shuffle(seed=SELECTION_SEED, buffer_size=10_000):
        if row.get("dataset") == "trec_coarse" and int(row.get("context_len") or 0) == CONTEXT_LEN:
            rows.append(dict(row))
            if len(rows) == count:
                return rows
    raise ValueError(f"expected {count} trec_coarse 128K rows, found {len(rows)}")


def ids() -> list[str]:
    return list(examples())


def prepare(task_id: str, workdir: Path, limits: Limits) -> str:
    row = examples()[task_id]
    (workdir / "context.txt").write_text(row["context"])
    return TASK.format(question=row["question"])


def score(task_id: str, workdir: Path, outcome: Outcome) -> dict:
    row = examples()[task_id]
    path = workdir / "answer.txt"
    answer = path.read_text().strip() if path.exists() else ""
    value = max(
        (
            synth_score(gold, answer, answer_type=row["answer_type"] or "")
            for gold in row["expected"]
        ),
        default=0.0,
    )
    return {"score": value, "answer": answer, "expected": row["expected"]}


# The RLM authors' OOLONG scorer.


def normalize_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            out.extend(normalize_answers(item))
        return out
    if isinstance(raw, dict):
        values = raw.get("answers") or raw.get("answer") or raw.values()
        return normalize_answers(list(values) if not isinstance(values, str) else values)
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
            return normalize_answers(list(parsed))
        if parsed is not None:
            return [str(parsed).strip()]
    return [part.strip() for part in text.split("||") if part.strip()]


COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")


def find_comparison_phrase(output: str) -> str | None:
    lowered = output.lower()
    hits = [(lowered.rfind(phrase), phrase) for phrase in COMPARISON_PHRASES if phrase in lowered]
    return max(hits)[1] if hits else None


def attempt_answer_parse(answer: str) -> str:
    comparison = find_comparison_phrase(answer)
    if comparison is not None:
        return comparison
    if ":" not in answer:
        return answer if len(answer) < 20 else answer.split()[-1]
    candidate = answer.split(":")[-1].strip()
    return candidate.replace("*", "").replace("[", "").replace("]", "")


def synth_score(expected: Any, output: str, *, answer_type: str = "") -> float:
    gold = parse_gold(expected)
    trimmed = attempt_answer_parse(output)
    gold_text = str(gold)
    if trimmed == gold_text or trimmed.lower() == gold_text.lower():
        return 1.0

    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except (TypeError, ValueError):
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        parsed = parse_date(trimmed)
        gold_date = parse_date(gold)
        return 1.0 if parsed is not None and parsed == gold_date else 0.0
    if gold_text and gold_text.casefold() not in COMPARISON_PHRASES:
        return 1.0 if gold_text.casefold() in output.casefold() else 0.0
    return 0.0


def parse_gold(value: Any) -> Any:
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


def parse_date(value: Any) -> date | None:
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
