"""JSONL run logger."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row


@logger("jsonl")
class JsonlLogger(Logger):
    def __init__(self, root: Path | str = "benchmarks/runs/latest") -> None:
        self.root = Path(root)
        self.rows_path = self.root / "rows.jsonl"
        self._rows: list[Row] = []

    def start(self, config: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "config.json", config)
        if not config.get("resume"):
            self.rows_path.write_text("", encoding="utf-8")

    def row(self, row: Row) -> None:
        self._rows.append(row)
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
        write_json(row_artifact_path(self.root, row), row.to_dict())
        write_json(self.root / "summary.json", summarize(self._rows))

    def summary(self, rows: list[Row]) -> None:
        write_json(self.root / "summary.json", summarize(rows))


def row_artifact_path(root: Path, row: Row) -> Path:
    return root / "artifacts" / row.dataset / row.example_id / row.runner / "prediction.json"


def load_rows(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [
        Row.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_rows(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.to_dict(), sort_keys=True) for row in rows]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["JsonlLogger", "load_rows", "row_artifact_path", "write_json", "write_rows"]
