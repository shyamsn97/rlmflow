"""Per-question comparison matrix across the runners and seeds of one run."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.loggers.jsonl import load_rows, write_json
from benchmarks.eval.types import Row

ERROR = "ERR"
MISSING = "—"
# An attempt that never produced a gradeable answer is missing data, not a zero.
# Anything else in `score.details["error"]` is the grader rejecting a real answer
# ("goal not reached", "malformed action"), which is a legitimate score of zero.
INFRA_ERRORS = (
    "APIConnectionError",
    "ConnectionError",
    "RateLimit",
    "Timeout",
    "Traceback",
    "budget exceeded",
)


def build_matrix(rows: Sequence[Row]) -> dict[str, Any]:
    """Group rows into a task × (runner, seed) grid with per-cell telemetry."""
    if not rows:
        raise ValueError("matrix requires at least one row")

    runners = sorted({row.runner for row in rows})
    seeds = sorted({row.seed for row in rows}, key=lambda seed: (seed is None, seed))
    cells = {(row.example_id, row.runner, row.seed): _cell_metrics(row) for row in rows}

    tasks = []
    for example_id in sorted({row.example_id for row in rows}):
        task_rows = [row for row in rows if row.example_id == example_id]
        per_runner = {
            runner: _runner_summary(
                [cells[key] for key in cells if key[0] == example_id and key[1] == runner]
            )
            for runner in runners
        }
        tasks.append(
            {
                "example_id": example_id,
                "dataset": task_rows[0].dataset,
                "role": task_rows[0].metadata.get("iteration_role", ""),
                "expected": _expected(task_rows[0]),
                "cells": {
                    runner: {str(seed): cells.get((example_id, runner, seed)) for seed in seeds}
                    for runner in runners
                },
                "runners": per_runner,
                "delta": _delta(per_runner, runners),
                "flaky": [runner for runner, item in per_runner.items() if item["spread"] > 0],
            }
        )

    return {
        "runners": runners,
        "seeds": [str(seed) for seed in seeds],
        "summary": {
            runner: _runner_summary([metrics for key, metrics in cells.items() if key[1] == runner])
            for runner in runners
        },
        "tasks": tasks,
    }


def render_markdown(matrix: Mapping[str, Any], *, title: str = "Benchmark matrix") -> str:
    runners = list(matrix["runners"])
    seeds = list(matrix["seeds"])
    lines = [f"# {title}", "", "## Summary", ""]
    lines.extend(_summary_table(matrix, runners))
    lines.extend(["", "## Per-question matrix", ""])
    lines.extend(_score_table(matrix, runners, seeds))
    lines.extend(["", "## Per-question rundown", ""])
    for task in matrix["tasks"]:
        lines.extend(_task_rundown(task, runners, seeds))
    return "\n".join(lines) + "\n"


def write_matrix(
    rows: Sequence[Row],
    output_dir: Path,
    *,
    title: str = "Benchmark matrix",
    stem: str = "matrix",
) -> tuple[Path, Path]:
    matrix = build_matrix(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    write_json(json_path, matrix)
    markdown_path.write_text(render_markdown(matrix, title=title), encoding="utf-8")
    return json_path, markdown_path


def _cell_metrics(row: Row) -> dict[str, Any]:
    metrics = row.prediction.metrics
    graph = metrics.get("graph", {})
    delegation = metrics.get("delegation", {})
    rlm = metrics.get("rlm", {})
    usage = row.prediction.usage
    error = row.prediction.error or row.score.details.get("error")
    return {
        "score": row.score.value,
        "correct": bool(row.score.correct),
        "error": error,
        "infra_error": _is_infra_error(error) or _is_infra_error(row.prediction.answer),
        "answer": row.prediction.answer,
        "llm_turns": graph.get("llm_turns", rlm.get("model_calls", 0)),
        "agents": graph.get("agents", 1),
        "children": delegation.get("children", rlm.get("subcalls", 0)),
        "max_depth": graph.get("max_depth", 0),
        "tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "time_seconds": metrics.get("time_seconds", 0.0),
    }


def _runner_summary(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(metrics)
    if not items:
        return {
            "count": 0,
            "unscored": 0,
            "mean_score": None,
            "spread": 0.0,
            "errors": 0,
            "llm_turns": 0.0,
            "children": 0,
            "delegating_runs": 0,
            "tokens": 0.0,
            "time_seconds": 0.0,
        }
    # Attempts that crashed carry no score, so averaging them as zero would grade the
    # infrastructure rather than the agent.
    scores = [item["score"] for item in items if not item["infra_error"]]
    return {
        "count": len(items),
        "unscored": sum(item["infra_error"] for item in items),
        "mean_score": fmean(scores) if scores else None,
        "spread": (max(scores) - min(scores)) if scores else 0.0,
        "errors": sum(bool(item["error"]) for item in items),
        "llm_turns": fmean(item["llm_turns"] for item in items),
        "children": sum(item["children"] for item in items),
        "delegating_runs": sum(item["children"] > 0 for item in items),
        "tokens": fmean(item["tokens"] for item in items),
        "time_seconds": fmean(item["time_seconds"] for item in items),
    }


def _delta(per_runner: Mapping[str, Mapping[str, Any]], runners: Sequence[str]) -> float | None:
    if len(runners) != 2:
        return None
    first, second = (per_runner[runner]["mean_score"] for runner in runners)
    if first is None or second is None:
        return None
    return first - second


def _expected(row: Row) -> str:
    for key in ("expected", "expected_answer"):
        if key in row.score.details:
            return _flat(row.score.details[key])
    return _flat(row.metadata.get("expected", ""))


def _summary_table(matrix: Mapping[str, Any], runners: Sequence[str]) -> list[str]:
    lines = [
        "| Runner | Runs | Scored | Mean score | Unscored (crashed) | Mean turns | "
        "Delegating runs | Children | Mean tokens | Mean time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for runner in runners:
        item = matrix["summary"][runner]
        lines.append(
            f"| `{runner}` | {item['count']} | {item['count'] - item['unscored']} | "
            f"{_mean(item['mean_score'], 3)} | {item['unscored']} | "
            f"{item['llm_turns']:.1f} | {item['delegating_runs']} | {item['children']} | "
            f"{item['tokens']:.0f} | {item['time_seconds']:.1f} |"
        )
    return lines


def _score_table(
    matrix: Mapping[str, Any],
    runners: Sequence[str],
    seeds: Sequence[str],
) -> list[str]:
    header = ["Question", "Role"]
    for runner in runners:
        header.extend(f"{runner} s{seed}" for seed in seeds)
        header.append(f"{runner} mean")
    if len(runners) == 2:
        header.append(f"Delta ({runners[0]} − {runners[1]})")
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for task in matrix["tasks"]:
        cells = [f"`{_short(task['example_id'])}`", task["role"] or MISSING]
        for runner in runners:
            for seed in seeds:
                cells.append(_score_cell(task["cells"][runner].get(seed)))
            cells.append(f"**{_mean(task['runners'][runner]['mean_score'], 2)}**")
        if task["delta"] is not None:
            cells.append(f"{task['delta']:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _score_cell(metrics: Mapping[str, Any] | None) -> str:
    if metrics is None:
        return MISSING
    if metrics["infra_error"]:
        return ERROR
    return f"{metrics['score']:.2f}".rstrip("0").rstrip(".") or "0"


def _is_infra_error(text: object) -> bool:
    value = str(text or "")
    return any(marker in value for marker in INFRA_ERRORS)


def _task_rundown(
    task: Mapping[str, Any],
    runners: Sequence[str],
    seeds: Sequence[str],
) -> list[str]:
    lines = [f"### `{_short(task['example_id'])}` — {task['role'] or 'unlabelled'}", ""]
    if task["expected"]:
        lines.extend([f"Expected: `{_truncate(task['expected'], 120)}`", ""])
    lines.extend(
        [
            "| Runner | Scores by seed | Mean | Turns | Children | Depth | Tokens | Time (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for runner in runners:
        item = task["runners"][runner]
        cells = task["cells"][runner]
        scores = " ".join(_score_cell(cells.get(seed)) for seed in seeds)
        depth = max((cells[seed]["max_depth"] for seed in seeds if cells.get(seed)), default=0)
        lines.append(
            f"| `{runner}` | {scores or MISSING} | {_mean(item['mean_score'], 2)} | "
            f"{item['llm_turns']:.1f} | {item['children']} | {depth} | {item['tokens']:.0f} | "
            f"{item['time_seconds']:.1f} |"
        )
    lines.append("")
    for note in _task_notes(task, runners, seeds):
        lines.append(f"- {note}")
    lines.append("")
    return lines


def _task_notes(
    task: Mapping[str, Any],
    runners: Sequence[str],
    seeds: Sequence[str],
) -> list[str]:
    notes = []
    for runner in runners:
        cells = task["cells"][runner]
        crashed = {
            seed: cells[seed]["error"] or cells[seed]["answer"]
            for seed in seeds
            if cells.get(seed) and cells[seed]["infra_error"]
        }
        if crashed:
            joined = "; ".join(f"s{seed}: {_truncate(text, 90)}" for seed, text in crashed.items())
            notes.append(f"`{runner}` produced no gradeable answer — {joined}")
        rejected = {
            seed: cells[seed]["error"]
            for seed in seeds
            if cells.get(seed) and cells[seed]["error"] and not cells[seed]["infra_error"]
        }
        if rejected:
            joined = "; ".join(f"s{seed}: {_truncate(text, 90)}" for seed, text in rejected.items())
            notes.append(f"`{runner}` was rejected by the grader — {joined}")
        wrong = [
            seed
            for seed in seeds
            if cells.get(seed) and not cells[seed]["correct"] and not cells[seed]["infra_error"]
        ]
        graded_wrong = [seed for seed in wrong if seed not in rejected]
        if graded_wrong:
            answers = "; ".join(
                f"s{seed}: `{_truncate(cells[seed]['answer'], 70)}`" for seed in graded_wrong
            )
            notes.append(f"`{runner}` answered incorrectly — {answers}")
        near = [
            seed for seed in wrong if _numeric_near_miss(cells[seed]["answer"], task["expected"])
        ]
        if near:
            joined = ", ".join(f"s{seed}" for seed in near)
            notes.append(
                f"`{runner}` matched the expected number to 1e-6 on {joined} but was still "
                "graded wrong: a scorer formatting artifact, not a reasoning failure"
            )
    if task["flaky"]:
        joined = ", ".join(f"`{runner}`" for runner in task["flaky"])
        notes.append(f"Seed-unstable for {joined}: the same task scored differently across seeds")
    if not notes:
        notes.append("Every runner solved this on every seed")
    return notes


def _numeric_near_miss(answer: object, expected: str, *, tolerance: float = 1e-6) -> bool:
    """Report answers that match the expected number but were graded wrong anyway."""
    try:
        given = float(_flat(answer))
        target = float(expected)
    except (TypeError, ValueError):
        return False
    scale = max(abs(target), 1.0)
    return abs(given - target) <= tolerance * scale


def _mean(value: float | None, places: int) -> str:
    return MISSING if value is None else f"{value:.{places}f}"


def _short(example_id: str) -> str:
    return example_id.removeprefix("delegation_")


def _flat(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _truncate(value: object, limit: int) -> str:
    text = _flat(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or rows.jsonl path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Benchmark matrix")
    parser.add_argument("--stem", default="matrix")
    args = parser.parse_args(argv)

    path = args.run.expanduser()
    rows_path = path if path.suffix == ".jsonl" else path / "rows.jsonl"
    rows = load_rows(rows_path)
    if not rows:
        parser.error(f"run has no rows: {rows_path}")

    json_path, markdown_path = write_matrix(rows, args.output_dir, title=args.title, stem=args.stem)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_matrix", "render_markdown", "write_matrix"]
