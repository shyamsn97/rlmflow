"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import (
    aime,  # noqa: F401
    browsecomp,  # noqa: F401
    livecodebench,  # noqa: F401
    longbench,  # noqa: F401
    oolong,  # noqa: F401
    sniah,  # noqa: F401
    sudoku,  # noqa: F401
    synthetic_needle,  # noqa: F401
)

DATASETS.alias("smoke", ["synthetic_needle"])
DATASETS.alias("needle", ["synthetic_needle"])
DATASETS.alias(
    "rlm-core",
    [
        "official_sniah",
        "official_aime_2025",
        "official_sudoku_extreme",
        "oolong",
        "official_codeqa",
    ],
)
DATASETS.alias("all", DATASETS.names())

__all__ = ["DATASETS", "dataset"]
