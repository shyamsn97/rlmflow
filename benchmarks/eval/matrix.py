"""Per-environment comparison matrix over normalized reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.eval.reporting import EnvironmentReport, load_environment_reports


def build_matrix(reports: Sequence[EnvironmentReport]) -> dict[str, Any]:
    if not reports:
        raise ValueError("matrix requires at least one report")
    runners = sorted({report.runner for report in reports})
    seeds = sorted({report.seed for report in reports}, key=lambda value: (value is None, value))
    environments = sorted({report.environment_id for report in reports})
    cells = {
        (report.environment_id, report.runner, report.seed): _cell(report)
        for report in reports
    }
    return {
        "runners": runners,
        "seeds": [str(seed) for seed in seeds],
        "summary": {
            runner: _summary([report for report in reports if report.runner == runner])
            for runner in runners
        },
        "environments": [
            {
                "environment_id": environment_id,
                "cells": {
                    runner: {
                        str(seed): cells.get((environment_id, runner, seed))
                        for seed in seeds
                    }
                    for runner in runners
                },
            }
            for environment_id in environments
        ],
    }


def render_markdown(
    matrix: Mapping[str, Any],
    *,
    title: str = "Benchmark matrix",
) -> str:
    runners = list(matrix["runners"])
    seeds = list(matrix["seeds"])
    columns = ["environment", *[f"{runner} / seed {seed}" for runner in runners for seed in seeds]]
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for environment in matrix["environments"]:
        values = [environment["environment_id"]]
        for runner in runners:
            for seed in seeds:
                cell = environment["cells"][runner][seed]
                values.append("—" if cell is None else f"{cell['score']:.4f}")
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def write_matrix(
    reports: Sequence[EnvironmentReport],
    output_dir: Path,
    *,
    title: str = "Benchmark matrix",
    stem: str = "matrix",
) -> tuple[Path, Path]:
    matrix = build_matrix(reports)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(matrix, title=title), encoding="utf-8")
    return json_path, markdown_path


def _cell(report: EnvironmentReport) -> dict[str, Any]:
    environment = _mapping(report.environment)
    delegation = _mapping(environment.get("delegation"))
    return {
        "score": report.score.value,
        "correct": report.score.correct,
        "error": report.error,
        "answer": report.answer,
        "input_tokens": report.runtime.input_tokens,
        "output_tokens": report.runtime.output_tokens,
        "reasoning_tokens": report.runtime.reasoning_tokens,
        "time_seconds": report.runtime.wall_time_seconds,
        "llm_turns": report.runtime.llm_turns,
        "agents": report.runtime.agents,
        "nodes": report.runtime.nodes,
        "children": delegation.get("children", 0),
        "query_calls": delegation.get("query_calls", 0),
        "environment": environment,
        "artifacts": report.artifacts,
    }


def _summary(reports: list[EnvironmentReport]) -> dict[str, Any]:
    return {
        "count": len(reports),
        "mean_score": fmean(report.score.value for report in reports),
        "correct": sum(report.score.correct is True for report in reports),
        "errors": sum(report.error is not None for report in reports),
        "input_tokens": sum(report.runtime.input_tokens for report in reports),
        "output_tokens": sum(report.runtime.output_tokens for report in reports),
        "reasoning_tokens": sum(report.runtime.reasoning_tokens for report in reports),
        "time_seconds": sum(report.runtime.wall_time_seconds for report in reports),
    }


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    return dict(to_dict()) if to_dict is not None else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_dir
    write_matrix(load_environment_reports(args.run_dir / "rows.jsonl"), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_matrix", "render_markdown", "write_matrix"]
