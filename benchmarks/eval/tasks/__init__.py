"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import (
    browsecomp,  # noqa: F401
    livecodebench,  # noqa: F401
    longbench,  # noqa: F401
    oolong,  # noqa: F401
    sudoku,  # noqa: F401
    synthetic_needle,  # noqa: F401
)

DATASETS.alias("smoke", ["synthetic_needle"])
DATASETS.alias("needle", ["synthetic_needle"])
DATASETS.alias("all", DATASETS.names())

__all__ = ["DATASETS", "dataset"]
