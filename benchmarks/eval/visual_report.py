"""Compatibility entry point for the normalized HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.eval.reporting import (
    EnvironmentReport,
    load_environment_reports,
    write_reports,
)
from benchmarks.eval.types import Row


def write_visual_report(
    reports: list[EnvironmentReport] | list[Row],
    output: Path,
    *,
    artifact_root: Path | None = None,
    title: str = "benchmark report",
) -> Path:
    del artifact_root
    normalized: list[EnvironmentReport] = []
    for item in reports:
        if isinstance(item, EnvironmentReport):
            normalized.append(item)
        else:
            contract = item.prediction.metrics.get("benchmark_task", {})
            normalized.append(
                EnvironmentReport.from_row(
                    benchmark=item.dataset,
                    row=item,
                    description_sha256=str(
                        contract.get(
                            "description_sha256",
                            contract.get("task_prompt_sha256", ""),
                        )
                    ),
                    observation_sha256=str(
                        contract.get(
                            "observation_sha256",
                            contract.get("inputs_sha256", ""),
                        )
                    ),
                )
            )
    write_reports(
        root=output.parent,
        name=title,
        description="",
        reports=normalized,
        persist_rows=False,
        persist_summary=False,
    )
    return output


def build_visual_report(
    reports: list[EnvironmentReport],
    *,
    title: str = "benchmark report",
    **_: Any,
) -> str:
    """Return the generated HTML for callers that need an in-memory value."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "report.html"
        write_visual_report(reports, path, title=title)
        return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    reports = load_environment_reports(args.run_dir / "rows.jsonl")
    write_visual_report(reports, args.run_dir / "report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_visual_report", "write_visual_report"]
