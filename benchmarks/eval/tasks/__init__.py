"""Built-in benchmark datasets."""

from __future__ import annotations

from benchmarks.eval import DATASETS, dataset

# Explicit built-ins so decorator registration remains grep-able.
from benchmarks.eval.tasks import (
    aime,  # noqa: F401
    arc_agi,  # noqa: F401
    browsecomp,  # noqa: F401
    dabstep,  # noqa: F401
    delegation_codeqa,  # noqa: F401
    delegation_iteration,  # noqa: F401
    delegation_sudoku,  # noqa: F401
    entailmentbank,  # noqa: F401
    grsqa,  # noqa: F401
    livecodebench,  # noqa: F401
    longbench,  # noqa: F401
    musique,  # noqa: F401
    natural_plan,  # noqa: F401
    oolong,  # noqa: F401
    parallelqa,  # noqa: F401
    planbench,  # noqa: F401
    sniah,  # noqa: F401
    sudoku,  # noqa: F401
    synthetic_needle,  # noqa: F401
    twowiki,  # noqa: F401
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
DATASETS.alias(
    "delegation-suite",
    [
        "delegation_parallelqa",
        "delegation_musique",
        "delegation_twowiki",
        "delegation_grsqa",
        "delegation_entailmentbank",
        "delegation_dabstep",
        "delegation_natural_plan",
        "delegation_planbench",
        "delegation_codeqa",
        "delegation_sudoku",
        "delegation_arc_agi",
    ],
)
DATASETS.alias(
    "delegation-suite-phase1",
    [
        "delegation_parallelqa",
        "delegation_musique",
        "delegation_twowiki",
        "delegation_dabstep",
        "delegation_natural_plan",
        "delegation_codeqa",
        "delegation_sudoku",
    ],
)
DATASETS.alias("all", DATASETS.names())

__all__ = ["DATASETS", "dataset"]
