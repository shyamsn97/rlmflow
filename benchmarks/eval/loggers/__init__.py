"""Built-in benchmark loggers."""

from __future__ import annotations

from benchmarks.eval import LOGGERS, logger
from benchmarks.eval.loggers import console, jsonl, report, wandb  # noqa: F401
from benchmarks.eval.types import Example, Logger, Row


class MultiLogger(Logger):
    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = loggers

    def start(self, config: dict) -> None:
        for item in self.loggers:
            item.start(config)

    def example_start(self, example: Example, *, runner: str, model: str) -> None:
        for item in self.loggers:
            item.example_start(example, runner=runner, model=model)

    def row(self, row: Row) -> None:
        for item in self.loggers:
            item.row(row)

    def summary(self, rows: list[Row]) -> None:
        for item in self.loggers:
            item.summary(rows)

    def finish(self) -> None:
        for item in reversed(self.loggers):
            item.finish()


__all__ = ["LOGGERS", "MultiLogger", "logger"]
