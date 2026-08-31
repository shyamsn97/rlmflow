"""Task-graph delegation benchmark helpers."""

from benchmarks.eval.delegation.align import ChildAlignment, score_alignment
from benchmarks.eval.delegation.annotations import (
    ANNOTATIONS,
    Obligation,
    TaskAnnotation,
    annotation_for,
)
from benchmarks.eval.delegation.conditions import (
    DelegationCondition,
    apply_condition,
    system_prompt_for,
)
from benchmarks.eval.delegation.faults import FaultKind, FaultSpec, fault_spec
from benchmarks.eval.delegation.manifest import PROBLEM_IDS, SLOTS, TaskGraphSlot
from benchmarks.eval.delegation.metrics import delegation_metrics
from benchmarks.eval.delegation.report import delegation_report

__all__ = [
    "ANNOTATIONS",
    "ChildAlignment",
    "FaultKind",
    "FaultSpec",
    "Obligation",
    "SLOTS",
    "PROBLEM_IDS",
    "DelegationCondition",
    "TaskAnnotation",
    "TaskGraphSlot",
    "annotation_for",
    "apply_condition",
    "delegation_report",
    "delegation_metrics",
    "fault_spec",
    "score_alignment",
    "system_prompt_for",
]
