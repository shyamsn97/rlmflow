"""Hidden deterministic fault schedules for later recovery runs."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum

from benchmarks.eval.delegation.manifest import FAULT_SEEDS


class FaultKind(StrEnum):
    CHILD_TIMEOUT = "child_timeout"
    MALFORMED_RESULT = "malformed_result"
    WRONG_TOOL_VALUE = "wrong_tool_value"
    PARTIAL_OUTPUT = "partial_output"
    CONFLICTING_RESULTS = "conflicting_results"
    LATE_COMPLETION = "late_completion"


@dataclass(frozen=True)
class FaultSpec:
    problem: int
    seed: int
    kind: FaultKind
    target_launch: int


FAULT_KINDS = tuple(FaultKind)


def fault_spec(problem: int, seed: int) -> FaultSpec:
    """Map a hidden suite seed to one reproducible fault and child launch."""

    if seed not in FAULT_SEEDS:
        raise ValueError("fault seed must be one of the sealed suite seeds")
    material = hashlib.sha256(f"{problem}:{seed}".encode()).digest()
    rng = random.Random(material)
    return FaultSpec(
        problem=problem,
        seed=seed,
        kind=FAULT_KINDS[FAULT_SEEDS.index(seed)],
        target_launch=rng.randrange(1, 4),
    )


__all__ = ["FAULT_KINDS", "FaultKind", "FaultSpec", "fault_spec"]
