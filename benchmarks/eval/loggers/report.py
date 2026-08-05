"""Markdown report logger."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row

MAX_ANSWER_CHARS = 300


@logger("report")
class ReportLogger(Logger):
    def __init__(self, root: Path | str = "benchmarks/runs/latest") -> None:
        self.root = Path(root)
        self._rows: list[Row] = []

    def row(self, row: Row) -> None:
        self._rows.append(row)
        self._write(self._rows)

    def summary(self, rows: list[Row]) -> None:
        self._write(rows)

    def _write(self, rows: list[Row]) -> None:
        summary = summarize(rows)
        lines = ["# Benchmark Report", ""]
        overall = summary.get("overall", {})
        if overall:
            lines.extend(
                [
                    "## Overall",
                    "",
                    f"- Count: {overall.get('count', 0)}",
                    f"- Correct: {overall.get('correct', 0)} / {overall.get('graded_count', 0)}",
                    f"- Accuracy: {_format_pct(overall.get('accuracy_pct'))}",
                    f"- Score: {_format_float(overall.get('score'))}",
                    f"- Errors: {overall.get('errors', 0)}",
                    "",
                ]
            )
        lines.extend(_summary_table("By Benchmark", summary.get("by_dataset", {})))
        lines.extend([""])
        lines.extend(_summary_table("By Runner", summary.get("by_runner", {})))
        lines.extend([""])
        lines.extend(
            _summary_table("By Runner And Benchmark", summary.get("by_runner_dataset", {}))
        )
        lines.extend(["", *_examples_section(rows)])
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table(title: str, values: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", ""]
    if not values:
        lines.append("No rows yet.")
        return lines
    lines.extend(
        [
            "| name | rows | correct | pct correct | score | errors | avg time | avg tokens | avg nodes | avg agents | max depth | subdelegated |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in values.items():
        tokens = int((item.get("input_tokens") or 0) + (item.get("output_tokens") or 0))
        lines.append(
            f"| `{name}` | {item.get('count', 0)} | "
            f"{item.get('correct', 0)}/{item.get('graded_count', 0)} | "
            f"{_format_pct(item.get('accuracy_pct'))} | "
            f"{_format_float(item.get('score'))} | "
            f"{item.get('errors', 0)} | "
            f"{_format_float(item.get('time_seconds'), suffix='s')} | "
            f"{tokens} | "
            f"{_format_float(item.get('graph_nodes'))} | "
            f"{_format_float(item.get('graph_agents'))} | "
            f"{_format_float(item.get('graph_max_depth'))} | "
            f"{item.get('subdelegated', 0)}/{item.get('graph_count', 0)} |"
        )
    return lines


def _examples_section(rows: list[Row]) -> list[str]:
    """Per-example answer-vs-expected comparison across every runner."""
    if not rows:
        return []
    grouped: OrderedDict[tuple, list[Row]] = OrderedDict()
    for row in rows:
        grouped.setdefault((row.dataset, row.example_id, row.seed), []).append(row)

    lines = ["## Examples", ""]
    for (dataset, example_id, seed), group in grouped.items():
        seed_label = "" if seed is None else f" (seed={seed})"
        lines.append(f"### `{dataset}` / `{example_id}`{seed_label}")
        lines.append("")
        lines.append(f"- Expected: {_format_cell(_expected(group))}")
        lines.append("")
        lines.append("| runner | correct | input tokens | output tokens | answer |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for row in group:
            answer = row.prediction.error or row.prediction.answer
            mark = "x" if row.prediction.error else _mark(row.score.correct, row.score.value)
            input_tokens = row.prediction.usage.get("input_tokens", 0)
            output_tokens = row.prediction.usage.get("output_tokens", 0)
            lines.append(
                f"| `{row.runner}` | {mark} | {input_tokens} | {output_tokens} | "
                f"{_format_cell(answer)} |"
            )
        lines.append("")
    return lines


def _expected(group: list[Row]):
    for row in group:
        expected = row.score.details.get("expected")
        if expected not in (None, [], ""):
            return expected
    return None


def _mark(correct: bool | None, value: float) -> str:
    if correct is True:
        return "PASS"
    if correct is False:
        return "FAIL"
    return f"{value:.2f}"


def _format_cell(value: object) -> str:
    if isinstance(value, (list, tuple)):
        text = " || ".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip() + "..."
    return text or "_(empty)_"


def _format_float(value: object, *, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3g}{suffix}"
    return ""


def _format_pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}%"
    return ""


__all__ = ["ReportLogger"]
