"""Generate machine-readable and Markdown comparisons between benchmark runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.loggers.jsonl import load_rows, write_json
from benchmarks.eval.types import Row


def compare_runs(runs: Mapping[str, list[Row]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")

    keyed = {
        label: {(row.dataset, row.example_id, row.model, row.seed): row for row in rows}
        for label, rows in runs.items()
    }
    keys = sorted(set().union(*(rows.keys() for rows in keyed.values())))
    labels = list(runs)
    tasks = []
    for key in keys:
        task_runs = {label: _row_metrics(rows[key]) for label, rows in keyed.items() if key in rows}
        task = {
            "dataset": key[0],
            "example_id": key[1],
            "model": key[2],
            "seed": key[3],
            "runs": task_runs,
        }
        if len(labels) == 2 and all(label in task_runs for label in labels):
            task["score_delta"] = task_runs[labels[1]]["score"] - task_runs[labels[0]]["score"]
        tasks.append(task)

    return {
        "labels": labels,
        "summary": {label: _run_summary(rows) for label, rows in runs.items()},
        "paired_tasks": sum(len(task["runs"]) == len(labels) for task in tasks),
        "tasks": tasks,
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    labels = list(comparison["labels"])
    lines = [
        "# Benchmark comparison",
        "",
        "## Summary",
        "",
        "| Run | Tasks | Accuracy | Mean score | Errors | Input tokens | Output tokens | Time (s) | Recursive tasks | Subcall tasks | Subcalls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        item = comparison["summary"][label]
        lines.append(
            f"| {_cell(label)} | {item['count']} | {item['accuracy']:.1%} | "
            f"{item['mean_score']:.4f} | {item['errors']} | "
            f"{item['input_tokens']} | {item['output_tokens']} | "
            f"{item['time_seconds']:.1f} | {item['delegated_tasks']} | "
            f"{item['subcall_tasks']} | {item['subcalls']} |"
        )

    lines.extend(
        [
            "",
            "## Per-problem results",
            "",
            _task_header(labels),
            _task_rule(labels),
        ]
    )
    for task in comparison["tasks"]:
        cells = [
            _cell(task["dataset"]),
            _cell(task["example_id"]),
        ]
        for label in labels:
            result = task["runs"].get(label)
            if result is None:
                cells.extend(["—", "—", "—", "—", "—", "—"])
                continue
            cells.extend(
                [
                    f"{result['score']:.4f}",
                    "yes" if result["correct"] else "no",
                    str(result["agents"]),
                    str(result["subcalls"]),
                    str(result["input_tokens"] + result["output_tokens"]),
                    f"{result['time_seconds']:.1f}",
                ]
            )
        if len(labels) == 2:
            delta = task.get("score_delta")
            cells.append("—" if delta is None else f"{delta:+.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_comparison(
    runs: Mapping[str, list[Row]],
    output_dir: Path,
) -> tuple[Path, Path]:
    comparison = compare_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    write_json(json_path, comparison)
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def _row_metrics(row: Row) -> dict[str, Any]:
    graph = row.prediction.metrics.get("graph", {})
    rlm = row.prediction.metrics.get("rlm", {})
    return {
        "runner": row.runner,
        "score": row.score.value,
        "correct": row.score.correct,
        "error": row.prediction.error,
        "input_tokens": row.prediction.usage.get("input_tokens", 0),
        "output_tokens": row.prediction.usage.get("output_tokens", 0),
        "time_seconds": row.prediction.metrics.get("time_seconds", 0.0),
        "agents": graph.get("agents", 0),
        "max_depth": graph.get("max_depth", 0),
        "subcalls": rlm.get("subcalls", 0),
    }


def _run_summary(rows: list[Row]) -> dict[str, Any]:
    graded = [row for row in rows if row.score.correct is not None]
    return {
        "count": len(rows),
        "accuracy": (
            sum(row.score.correct is True for row in graded) / len(graded) if graded else 0.0
        ),
        "mean_score": fmean(row.score.value for row in rows) if rows else 0.0,
        "errors": sum(bool(row.prediction.error) for row in rows),
        "input_tokens": sum(row.prediction.usage.get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.prediction.usage.get("output_tokens", 0) for row in rows),
        "time_seconds": sum(row.prediction.metrics.get("time_seconds", 0.0) for row in rows),
        "delegated_tasks": sum(
            row.prediction.metrics.get("graph", {}).get("agents", 0) > 1 for row in rows
        ),
        "subcall_tasks": sum(
            row.prediction.metrics.get("rlm", {}).get("subcalls", 0) > 0 for row in rows
        ),
        "subcalls": sum(row.prediction.metrics.get("rlm", {}).get("subcalls", 0) for row in rows),
    }


def _task_header(labels: list[str]) -> str:
    columns = ["Dataset", "Problem"]
    for label in labels:
        columns.extend(
            [
                f"{label} score",
                f"{label} correct",
                f"{label} agents",
                f"{label} subcalls",
                f"{label} tokens",
                f"{label} time (s)",
            ]
        )
    if len(labels) == 2:
        columns.append(f"Score delta ({labels[1]} − {labels[0]})")
    return "| " + " | ".join(_cell(column) for column in columns) + " |"


def _task_rule(labels: list[str]) -> str:
    count = 2 + 6 * len(labels) + int(len(labels) == 2)
    return "|" + "|".join(["---"] * count) + "|"


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RUN_DIR",
        help="Labeled run directory containing rows.jsonl; repeat at least twice.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    runs: dict[str, list[Row]] = {}
    for value in args.run:
        if "=" not in value:
            parser.error("--run must use LABEL=RUN_DIR")
        label, raw_path = value.split("=", 1)
        if not label or label in runs:
            parser.error(f"run label must be non-empty and unique: {label!r}")
        path = Path(raw_path).expanduser()
        rows_path = path if path.name == "rows.jsonl" else path / "rows.jsonl"
        runs[label] = load_rows(rows_path)
        if not runs[label]:
            parser.error(f"run has no rows: {rows_path}")

    json_path, markdown_path = write_comparison(runs, args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare_runs", "render_markdown", "write_comparison"]
