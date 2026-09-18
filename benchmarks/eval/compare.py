"""Generate machine-readable and Markdown comparisons between benchmark runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.loggers.jsonl import load_rows, write_json
from benchmarks.eval.telemetry import RowTelemetry
from benchmarks.eval.types import Row


def compare_runs(runs: Mapping[str, list[Row]]) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")

    pair_by_runner = any(
        len({(row.dataset, row.example_id, row.model, row.seed) for row in rows}) < len(rows)
        for rows in runs.values()
    )

    def row_key(row: Row) -> tuple:
        base = (row.dataset, row.example_id, row.model, row.seed)
        return (base[0], base[1], row.runner if pair_by_runner else None, base[2], base[3])

    keyed = {label: {row_key(row): row for row in rows} for label, rows in runs.items()}
    keys = sorted(set().union(*(rows.keys() for rows in keyed.values())))
    labels = list(runs)
    tasks = []
    for key in keys:
        task_runs = {label: _row_metrics(rows[key]) for label, rows in keyed.items() if key in rows}
        task = {
            "dataset": key[0],
            "example_id": key[1],
            "runner": key[2] or sorted({item["runner"] for item in task_runs.values()}),
            "model": key[3],
            "seed": key[4],
            "runs": task_runs,
        }
        if len(labels) == 2 and all(label in task_runs for label in labels):
            task["score_delta"] = task_runs[labels[1]]["score"] - task_runs[labels[0]]["score"]
        tasks.append(task)

    return {
        "labels": labels,
        "paired_by_runner_arm": pair_by_runner,
        "summary": {label: _run_summary(rows) for label, rows in runs.items()},
        "by_route": {label: _route_summary(rows) for label, rows in runs.items()},
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
    telemetry = RowTelemetry.from_row(row)
    graph = telemetry.graph
    delegation = telemetry.delegation
    environment = telemetry.environment
    candidates = environment.candidate_search
    return {
        "runner": telemetry.runner,
        "score": telemetry.score,
        "correct": telemetry.correct,
        "error": telemetry.error,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "time_seconds": telemetry.time_seconds,
        "agents": graph.get("agents", 0),
        "max_depth": graph.get("max_depth", 0),
        "subcalls": delegation.get("children", 0),
        "dependency_edges": delegation.get("dependency_edge_count", 0),
        "dependency_chain_depth": delegation.get("dependency_chain_depth", 0),
        "recursive_launches": delegation.get(
            "recursive_launch_count",
            0,
        ),
        "max_launch_batch": delegation.get("max_launch_batch", 0),
        "max_ready_launch_batch": delegation.get("max_ready_launch_batch", 0),
        "retrieval_ratio": delegation.get("retrieval_ratio", 0.0),
        "retrieved_results": delegation.get("retrieved_child_results", 0),
        "model_turns": graph.get("llm_turns", 0),
        "environment_actions": environment.accepted_actions,
        "primary_actions": environment.primary_actions,
        "probe_actions": environment.probe_actions,
        "environment_instances": environment.instances_forked,
        "candidate_count": candidates.get("evaluations_used", 0),
        "candidate_oracle_best_score": candidates.get("oracle_best_score"),
        "selected_candidate_score": candidates.get("selected_score"),
        "selection_regret": candidates.get("selection_regret"),
        "artifacts": telemetry.artifacts,
    }


def _run_summary(rows: list[Row]) -> dict[str, Any]:
    graded = [row for row in rows if row.score.correct is not None]
    telemetry = [RowTelemetry.from_row(row) for row in rows]
    return {
        "count": len(rows),
        "accuracy": (
            sum(row.score.correct is True for row in graded) / len(graded) if graded else 0.0
        ),
        "mean_score": fmean(row.score.value for row in rows) if rows else 0.0,
        "errors": sum(bool(item.error) for item in telemetry),
        "input_tokens": sum(item.input_tokens for item in telemetry),
        "output_tokens": sum(item.output_tokens for item in telemetry),
        "time_seconds": sum(item.time_seconds for item in telemetry),
        "delegated_tasks": sum(item.graph.get("agents", 0) > 1 for item in telemetry),
        "subcall_tasks": sum(item.delegation.get("children", 0) > 0 for item in telemetry),
        "subcalls": sum(item.delegation.get("children", 0) for item in telemetry),
    }


def _route_summary(rows: list[Row]) -> dict[str, Any]:
    routes = sorted(
        {
            str(row.metadata.get("route_hypothesis"))
            for row in rows
            if row.metadata.get("route_hypothesis")
        }
    )
    return {
        route: _run_summary([row for row in rows if row.metadata.get("route_hypothesis") == route])
        for route in routes
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
