"""Compatibility logger backed by normalized EnvironmentReport rows."""

from __future__ import annotations

from pathlib import Path

from benchmarks.eval import logger
from benchmarks.eval.reporting import EnvironmentReport, write_reports
from benchmarks.eval.types import Logger, Row


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
        reports = [_from_legacy_row(row) for row in rows]
        write_reports(
            root=self.root,
            name="legacy",
            description="Legacy benchmark artifact converted to EnvironmentReport.",
            reports=reports,
            persist_rows=False,
            persist_summary=False,
        )


def _from_legacy_row(row: Row) -> EnvironmentReport:
    contract = row.prediction.metrics.get("benchmark_task", {})
    return EnvironmentReport.from_row(
        benchmark=row.dataset,
        row=row,
        description_sha256=str(
            contract.get("description_sha256", contract.get("task_prompt_sha256", ""))
        ),
        observation_sha256=str(
            contract.get("observation_sha256", contract.get("inputs_sha256", ""))
        ),
    )


__all__ = ["ReportLogger"]
