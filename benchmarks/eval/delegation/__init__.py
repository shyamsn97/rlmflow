"""Shared graph telemetry and comparison contracts."""

from benchmarks.eval.delegation.metrics import delegation_metrics
from benchmarks.eval.delegation.parity import comparison_contract

__all__ = ["comparison_contract", "delegation_metrics"]
