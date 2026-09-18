"""Per-question comparison matrix across the runners and seeds of one run."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.acceptance import DEFAULT_ACCEPTANCE_POLICY
from benchmarks.eval.loggers.jsonl import load_rows, write_json
from benchmarks.eval.telemetry import (
    INFRA_ERROR_MARKERS,
    RowTelemetry,
    expected_from_row,
)
from benchmarks.eval.types import Row

DEPTH_BASELINE_RUNNER = DEFAULT_ACCEPTANCE_POLICY.baseline_runner
DEPTH_TREATMENT_RUNNER = DEFAULT_ACCEPTANCE_POLICY.treatment_runner

ERROR = "ERR"
MISSING = "—"
INFRA_ERRORS = INFRA_ERROR_MARKERS


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
                "role": task_rows[0].metadata.get(
                    "route_hypothesis",
                    task_rows[0].metadata.get("iteration_role", ""),
                ),
                "domain": task_rows[0].metadata.get("domain", ""),
                "expected": expected_from_row(task_rows[0]),
                "cells": {
                    runner: {str(seed): cells.get((example_id, runner, seed)) for seed in seeds}
                    for runner in runners
                },
                "runners": per_runner,
                "delta": _delta(per_runner, runners),
                "deltas_from_depth_0": _baseline_deltas(per_runner),
                "flaky": [runner for runner, item in per_runner.items() if item["spread"] > 0],
            }
        )

    paired = _paired_deltas_by_domain(tasks, runners)
    return {
        "runners": runners,
        "seeds": [str(seed) for seed in seeds],
        "summary": {
            runner: _runner_summary([metrics for key, metrics in cells.items() if key[1] == runner])
            for runner in runners
        },
        "by_route": _route_summaries(rows, cells, runners),
        "by_domain": _domain_summaries(rows, cells, runners),
        "paired_deltas_by_domain": paired,
        "acceptance_gates": _acceptance_gates(tasks, runners),
        "tasks": tasks,
    }


def render_markdown(matrix: Mapping[str, Any], *, title: str = "Benchmark matrix") -> str:
    runners = list(matrix["runners"])
    seeds = list(matrix["seeds"])
    lines = [f"# {title}", "", "## Summary", ""]
    lines.extend(_summary_table(matrix, runners))
    if matrix.get("by_domain"):
        lines.extend(
            [
                "",
                "Native scores are comparable only within a domain; the mixed summary "
                "above is operational telemetry, not a headline score.",
                "",
                "## Domain-native comparisons",
                "",
            ]
        )
        for domain, per_runner in matrix["by_domain"].items():
            lines.append(f"### {domain}")
            lines.append("")
            for runner in runners:
                item = per_runner[runner]
                detail = (
                    f"`{runner}`: score {_mean(item['mean_score'], 3)}, "
                    f"seed spread {item['spread']:.3f}, "
                    f"{item['unscored']} infra-unscored, "
                    f"{item['environment_actions']} environment actions, "
                    f"{item['tokens']:.0f} mean tokens"
                )
                paired = matrix.get("paired_deltas_by_domain", {}).get(domain, {}).get(runner)
                if paired:
                    detail += (
                        f"; paired Δ {paired['mean_delta']:.3f}, 95% CI "
                        f"[{paired['ci95'][0]:.3f}, {paired['ci95'][1]:.3f}]"
                    )
                oracle_paired = (
                    matrix.get("paired_deltas_vs_oracle_by_domain", {}).get(domain, {}).get(runner)
                )
                if oracle_paired:
                    detail += (
                        f"; vs equal-budget oracle Δ {oracle_paired['mean_delta']:.3f}, "
                        f"95% CI [{oracle_paired['ci95'][0]:.3f}, "
                        f"{oracle_paired['ci95'][1]:.3f}]"
                    )
                lines.append(f"- {detail}")
            lines.append("")
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
    telemetry = RowTelemetry.from_row(row)
    metrics = row.prediction.metrics
    graph = telemetry.graph
    delegation = telemetry.delegation
    gold_alignment = delegation.get("gold_graph_alignment", {})
    candidate_search = telemetry.environment.candidate_search
    error = telemetry.error
    return {
        "score": telemetry.score,
        "correct": bool(telemetry.correct),
        "error": error,
        "infra_error": telemetry.infra_error,
        "answer": row.prediction.answer,
        "llm_turns": graph.get("llm_turns", 0),
        "agents": graph.get("agents", 1),
        "children": delegation.get("children", 0),
        "max_depth": graph.get("max_depth", 0),
        "delegation_topology": telemetry.topology,
        "dependency_edges": delegation.get("dependency_edge_count", 0),
        "dependency_chain_depth": delegation.get("dependency_chain_depth", 0),
        "recursive_launches": delegation.get(
            "recursive_launch_count",
            0,
        ),
        "max_launch_batch": delegation.get("max_launch_batch", 0),
        "max_ready_launch_batch": delegation.get("max_ready_launch_batch", 0),
        "concurrent_launch_batches": delegation.get("concurrent_launch_batches", 0),
        "retrieved_results": delegation.get(
            "retrieved_child_results",
            0,
        ),
        "retrieval_ratio": delegation.get(
            "retrieval_ratio",
            0.0,
        ),
        "gold_edge_precision": gold_alignment.get("edge_precision"),
        "gold_edge_recall": gold_alignment.get("edge_recall"),
        "query_calls": delegation.get("query_calls", 0),
        "environment_actions": telemetry.environment.accepted_actions,
        "primary_actions": telemetry.environment.primary_actions,
        "probe_actions": telemetry.environment.probe_actions,
        "environment_instances": telemetry.environment.instances_forked,
        "candidate_count": candidate_search.get("evaluations_used", 0),
        "selection_compliance": metrics.get("environment", {}).get("selection_compliance"),
        "candidate_oracle_best_score": candidate_search.get("oracle_best_score"),
        "selected_candidate_score": candidate_search.get("selected_score"),
        "selection_regret": candidate_search.get("selection_regret"),
        "candidate_diversity": candidate_search.get("candidate_diversity"),
        "candidate_outcome_diversity": candidate_search.get("outcome_diversity"),
        "selected_candidate_depth": next(
            (
                candidate.get("owner_depth")
                for candidate in candidate_search.get("candidates", [])
                if candidate.get("id") == candidate_search.get("selected_candidate_id")
            ),
            None,
        ),
        "candidate_owner_count": len(
            {
                candidate.get("owner_id")
                for candidate in candidate_search.get("candidates", [])
                if candidate.get("owner_id")
            }
        ),
        "candidate_branch_utilization": (
            len(
                {
                    candidate.get("owner_id")
                    for candidate in candidate_search.get("candidates", [])
                    if candidate.get("owner_id")
                }
            )
            / max(
                1,
                telemetry.environment.instances_forked + 1,
            )
            if candidate_search.get("candidates")
            else None
        ),
        "tokens": telemetry.input_tokens + telemetry.output_tokens,
        "time_seconds": telemetry.time_seconds,
        "compute_seconds": row.metadata.get(
            "aggregate_compute_seconds",
            metrics.get("time_seconds", 0.0),
        ),
        "artifacts": dict(row.prediction.artifacts),
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
            "dependency_edges": 0,
            "recursive_launches": 0,
            "max_launch_batch": 0.0,
            "max_ready_launch_batch": 0.0,
            "concurrent_launch_batches": 0,
            "retrieved_results": 0,
            "gold_edge_precision": None,
            "gold_edge_recall": None,
            "query_calls": 0,
            "environment_actions": 0,
            "candidate_count": 0.0,
            "selection_compliance": None,
            "selection_regret": None,
            "candidate_diversity": None,
            "candidate_outcome_diversity": None,
            "candidate_oracle_best_score": None,
            "selected_candidate_depth": None,
            "candidate_owner_count": 0.0,
            "candidate_branch_utilization": None,
            "score_per_1k_tokens": None,
            "score_per_environment_action": None,
            "score_per_candidate_evaluation": None,
            "score_per_second": None,
            "delegating_runs": 0,
            "delegation_topologies": {},
            "tokens": 0.0,
            "time_seconds": 0.0,
            "compute_seconds": 0.0,
        }

    def optional_mean(key: str) -> float | None:
        values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
        return fmean(values) if values else None

    # Attempts that crashed carry no score, so averaging them as zero would grade the
    # infrastructure rather than the agent.
    scores = [item["score"] for item in items if not item["infra_error"]]
    token_total = sum(item["tokens"] for item in items)
    environment_total = sum(item["environment_actions"] for item in items)
    candidate_total = sum(item["candidate_count"] for item in items)
    time_total = sum(item["time_seconds"] for item in items)
    score_total = sum(scores)
    return {
        "count": len(items),
        "unscored": sum(item["infra_error"] for item in items),
        "mean_score": fmean(scores) if scores else None,
        "spread": (max(scores) - min(scores)) if scores else 0.0,
        "errors": sum(bool(item["error"]) for item in items),
        "llm_turns": fmean(item["llm_turns"] for item in items),
        "children": sum(item["children"] for item in items),
        "dependency_edges": sum(item["dependency_edges"] for item in items),
        "recursive_launches": sum(item["recursive_launches"] for item in items),
        "max_launch_batch": fmean(item["max_launch_batch"] for item in items),
        "max_ready_launch_batch": fmean(item["max_ready_launch_batch"] for item in items),
        "concurrent_launch_batches": sum(item["concurrent_launch_batches"] for item in items),
        "retrieved_results": sum(item["retrieved_results"] for item in items),
        "gold_edge_precision": optional_mean("gold_edge_precision"),
        "gold_edge_recall": optional_mean("gold_edge_recall"),
        "query_calls": sum(item["query_calls"] for item in items),
        "environment_actions": sum(item["environment_actions"] for item in items),
        "candidate_count": fmean(item["candidate_count"] for item in items),
        "selection_compliance": optional_mean("selection_compliance"),
        "selection_regret": optional_mean("selection_regret"),
        "candidate_diversity": optional_mean("candidate_diversity"),
        "candidate_outcome_diversity": optional_mean("candidate_outcome_diversity"),
        "candidate_oracle_best_score": optional_mean("candidate_oracle_best_score"),
        "selected_candidate_depth": optional_mean("selected_candidate_depth"),
        "candidate_owner_count": fmean(item["candidate_owner_count"] for item in items),
        "candidate_branch_utilization": optional_mean("candidate_branch_utilization"),
        "score_per_1k_tokens": (score_total * 1_000 / token_total if token_total else None),
        "score_per_environment_action": (
            score_total / environment_total if environment_total else None
        ),
        "score_per_candidate_evaluation": (
            score_total / candidate_total if candidate_total else None
        ),
        "score_per_second": score_total / time_total if time_total else None,
        "delegating_runs": sum(item["children"] > 0 for item in items),
        "delegation_topologies": dict(
            sorted(Counter(item["delegation_topology"] for item in items).items())
        ),
        "tokens": fmean(item["tokens"] for item in items),
        "time_seconds": fmean(item["time_seconds"] for item in items),
        "compute_seconds": fmean(item["compute_seconds"] for item in items),
    }


def _delta(per_runner: Mapping[str, Mapping[str, Any]], runners: Sequence[str]) -> float | None:
    if len(runners) != 2:
        return None
    first, second = (per_runner[runner]["mean_score"] for runner in runners)
    if first is None or second is None:
        return None
    return first - second


def _baseline_deltas(per_runner: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    baseline = per_runner.get(DEPTH_BASELINE_RUNNER, {}).get("mean_score")
    return {
        runner: (
            None
            if baseline is None or metrics.get("mean_score") is None
            else metrics["mean_score"] - baseline
        )
        for runner, metrics in per_runner.items()
        if runner != DEPTH_BASELINE_RUNNER
    }


def _route_summaries(
    rows: Sequence[Row],
    cells: Mapping[tuple, Mapping[str, Any]],
    runners: Sequence[str],
) -> dict[str, dict[str, Any]]:
    routes = sorted(
        {
            str(row.metadata.get("route_hypothesis"))
            for row in rows
            if row.metadata.get("route_hypothesis")
        }
    )
    return {
        route: {
            runner: _runner_summary(
                [
                    cells[(row.example_id, row.runner, row.seed)]
                    for row in rows
                    if row.runner == runner and row.metadata.get("route_hypothesis") == route
                ]
            )
            for runner in runners
        }
        for route in routes
    }


def _domain_summaries(
    rows: Sequence[Row],
    cells: Mapping[tuple, Mapping[str, Any]],
    runners: Sequence[str],
) -> dict[str, dict[str, Any]]:
    domains = sorted(
        {str(row.metadata.get("domain")) for row in rows if row.metadata.get("domain")}
    )
    return {
        domain: {
            runner: _runner_summary(
                [
                    cells[(row.example_id, row.runner, row.seed)]
                    for row in rows
                    if row.runner == runner and row.metadata.get("domain") == domain
                ]
            )
            for runner in runners
        }
        for domain in domains
    }


def _paired_deltas_by_domain(
    tasks: Sequence[Mapping[str, Any]],
    runners: Sequence[str],
    *,
    baseline: str = DEPTH_BASELINE_RUNNER,
) -> dict[str, dict[str, Any]]:
    if baseline not in runners:
        return {}
    domains = sorted({str(task.get("domain", "")) for task in tasks if task.get("domain")})
    result: dict[str, dict[str, Any]] = {}
    for domain in domains:
        domain_tasks = [task for task in tasks if task.get("domain") == domain]
        comparisons: dict[str, Any] = {}
        for runner in runners:
            if runner == baseline:
                continue
            deltas: list[float] = []
            example_deltas: list[float] = []
            for task in domain_tasks:
                baseline_cells = task["cells"].get(baseline, {})
                runner_cells = task["cells"].get(runner, {})
                task_deltas: list[float] = []
                for seed in set(baseline_cells) & set(runner_cells):
                    base = baseline_cells[seed]
                    candidate = runner_cells[seed]
                    if (
                        base is None
                        or candidate is None
                        or base["infra_error"]
                        or candidate["infra_error"]
                    ):
                        continue
                    delta = float(candidate["score"]) - float(base["score"])
                    deltas.append(delta)
                    task_deltas.append(delta)
                if task_deltas:
                    example_deltas.append(fmean(task_deltas))
            if deltas:
                lower, upper = _bootstrap_mean_ci(example_deltas)
                comparisons[runner] = {
                    "paired_rows": len(deltas),
                    "paired_examples": len(example_deltas),
                    "mean_delta": fmean(example_deltas),
                    "ci95": [lower, upper],
                    "wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "losses": sum(delta < 0 for delta in deltas),
                }
        if comparisons:
            result[domain] = comparisons
    return result


def _acceptance_gates(
    tasks: Sequence[Mapping[str, Any]],
    runners: Sequence[str],
) -> dict[str, dict[str, Any]]:
    domains = sorted({str(task.get("domain", "")) for task in tasks if task.get("domain")})
    result: dict[str, dict[str, Any]] = {}
    for domain in domains:
        domain_tasks = [task for task in tasks if task.get("domain") == domain]

        def scores(runner: str) -> list[float]:
            return [
                float(cell["score"])
                for task in domain_tasks
                for cell in task["cells"].get(runner, {}).values()
                if cell is not None and not cell["infra_error"]
            ]

        baseline = scores(DEFAULT_ACCEPTANCE_POLICY.baseline_runner)
        treatment = scores(DEFAULT_ACCEPTANCE_POLICY.treatment_runner)
        candidate_variation = any(
            cell is not None
            and cell.get("candidate_count", 0) > 1
            and (
                cell.get("selection_regret") not in (None, 0)
                or cell.get("candidate_owner_count", 0) > 1
            )
            for task in domain_tasks
            for runner_cells in task["cells"].values()
            for cell in runner_cells.values()
        )
        result[domain] = DEFAULT_ACCEPTANCE_POLICY.assess_scores(
            baseline,
            treatment,
            candidate_variation=candidate_variation,
        )
    return result


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int = 2_000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    estimates = sorted(fmean(rng.choice(values) for _ in values) for _ in range(samples))
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


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
