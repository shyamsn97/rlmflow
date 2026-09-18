"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset
from benchmarks.eval.suite import register_suites

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import (
    arc_agi_3,  # noqa: F401
    mazebench,  # noqa: F401
    minimaze,  # noqa: F401
    oolong,  # noqa: F401
    oolong_pairs,  # noqa: F401
    sudoku_search,  # noqa: F401
    synthetic_needle,  # noqa: F401
)

register_suites(DATASETS)

__all__ = ["DATASETS", "dataset"]
