"""Compare normalized environment reports from benchmark runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.reporting import EnvironmentReport, load_environment_reports


def compare_runs(
    runs: Mapping[str, list[EnvironmentReport]],
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")

    def key(report: EnvironmentReport) -> tuple:
        return (
            report.benchmark,
            report.environment_id,
            report.runner,
            report.model,
            report.seed,
        )

    indexed = {label: {key(report): report for report in reports} for label, reports in runs.items()}
    keys = sorted(set().union(*(reports.keys() for reports in indexed.values())))
    labels = list(runs)
    environments = []
    for report_key in keys:
        values = {
            label: _metrics(reports[report_key])
            for label, reports in indexed.items()
            if report_key in reports
        }
        item: dict[str, Any] = {
            "benchmark": report_key[0],
            "environment_id": report_key[1],
            "runner": report_key[2],
            "model": report_key[3],
            "seed": report_key[4],
            "runs": values,
        }
        if len(labels) == 2 and all(label in values for label in labels):
            item["score_delta"] = values[labels[1]]["score"] - values[labels[0]]["score"]
        environments.append(item)
    return {
        "labels": labels,
        "summary": {
            label: {
                runner: _summary([report for report in reports if report.runner == runner])
                for runner in sorted({report.runner for report in reports})
            }
            for label, reports in runs.items()
        },
        "paired_environments": sum(
            len(item["runs"]) == len(labels) for item in environments
        ),
        "environments": environments,
    }


def render_markdown(comparison: Mapping[str, Any]) -> str:
    labels = list(comparison["labels"])
    columns = ["benchmark", "environment", "runner", *labels]
    if len(labels) == 2:
        columns.append("delta")
    summary_columns = [
        "run",
        "runner",
        "mean score",
        "correct",
        "errors",
        "input tokens",
        "output tokens",
        "reasoning tokens",
        "seconds",
    ]
    lines = [
        "# Benchmark comparison",
        "",
        "| " + " | ".join(summary_columns) + " |",
        "|" + "|".join("---" for _ in summary_columns) + "|",
    ]
    for label in labels:
        for runner, summary in comparison["summary"][label].items():
            values = [
                label,
                runner,
                f"{summary['mean_score']:.3f}",
                f"{summary['correct']}/{summary['count']}",
                summary["errors"],
                f"{summary['input_tokens']:,}",
                f"{summary['output_tokens']:,}",
                f"{summary['reasoning_tokens']:,}",
                f"{summary['time_seconds']:.0f}",
            ]
            lines.append("| " + " | ".join(str(value) for value in values) + " |")
    lines += [
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for item in comparison["environments"]:
        values = [item["benchmark"], item["environment_id"], item["runner"]]
        values.extend(
            (
                f"{item['runs'][label]['score']:.4f}"
                if label in item["runs"]
                else "—"
            )
            for label in labels
        )
        if len(labels) == 2:
            delta = item.get("score_delta")
            values.append("—" if delta is None else f"{delta:+.4f}")
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def write_comparison(
    runs: Mapping[str, list[EnvironmentReport]],
    output_dir: Path,
) -> tuple[Path, Path]:
    comparison = compare_runs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    return json_path, markdown_path


def _metrics(report: EnvironmentReport) -> dict[str, Any]:
    return {
        "score": report.score.value,
        "correct": report.score.correct,
        "error": report.error,
        "input_tokens": report.runtime.input_tokens,
        "output_tokens": report.runtime.output_tokens,
        "reasoning_tokens": report.runtime.reasoning_tokens,
        "time_seconds": report.runtime.wall_time_seconds,
        "agents": report.runtime.agents,
        "nodes": report.runtime.nodes,
    }


def _summary(reports: list[EnvironmentReport]) -> dict[str, Any]:
    return {
        "count": len(reports),
        "mean_score": fmean(report.score.value for report in reports) if reports else 0.0,
        "correct": sum(report.score.correct is True for report in reports),
        "errors": sum(report.error is not None for report in reports),
        "input_tokens": sum(report.runtime.input_tokens for report in reports),
        "output_tokens": sum(report.runtime.output_tokens for report in reports),
        "reasoning_tokens": sum(report.runtime.reasoning_tokens for report in reports),
        "time_seconds": sum(report.runtime.wall_time_seconds for report in reports),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    runs: dict[str, list[EnvironmentReport]] = {}
    for value in args.run:
        if "=" not in value:
            parser.error("--run must use LABEL=RUN_DIR")
        label, directory = value.split("=", 1)
        runs[label] = load_environment_reports(Path(directory) / "rows.jsonl")
    write_comparison(runs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compare_runs", "render_markdown", "write_comparison"]
