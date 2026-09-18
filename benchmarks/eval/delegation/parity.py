"""Versioned comparison contract recorded by official and rlmflow arms."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from benchmarks.eval.environments.registry import environment_contract

COMPARISON_CONTRACT_VERSION = "official-rlm-comparison-v1"


def comparison_contract(
    *,
    runtime: str,
    prompt: str,
    inputs: Mapping[str, str],
    root_model: str,
    worker_model: str | None,
    recursive_levels: int,
    max_iterations: int,
    max_tokens: int | None,
    attempt_timeout_seconds: int,
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_budget = int(environment.get("candidate_budget", 0)) if environment is not None else 0
    action_budget = int(environment.get("max_actions", 0)) if environment is not None else 0
    kind = str(environment.get("kind", "")) if environment is not None else ""
    action_schema = environment_contract(kind) if kind else None
    effective_worker = worker_model or root_model
    return {
        "version": COMPARISON_CONTRACT_VERSION,
        "runtime": runtime,
        "task_prompt_sha256": _hash(prompt),
        "inputs_sha256": _hash(
            json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "prompt_strategy": (
            "capability-neutral-candidate-search-v1"
            if candidate_budget
            else "runtime-native-recursion-v1"
        ),
        "root_model": root_model,
        "worker_model": effective_worker,
        "equal_model": effective_worker == root_model,
        "recursive_levels": recursive_levels,
        "limits": {
            "iterations_per_agent": max_iterations,
            "aggregate_tokens": max_tokens,
            "attempt_seconds": attempt_timeout_seconds,
            "environment_actions": action_budget,
            "candidate_evaluations": candidate_budget,
        },
        "action_schema": action_schema,
        "native_scorer": "selected-instance" if candidate_budget else "root-instance",
        "fresh_state": "fork-immediate-parent-accepted-prefix" if kind else None,
        "candidate_ledger": "shared-row-explicit-root-selection" if candidate_budget else None,
        "handle_dag": {
            "supported": runtime == "rlmflow",
            "official_composition_credit": False,
        },
    }


def comparable_contract_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must be byte-for-byte equal across runtime arms."""
    return {key: value for key, value in contract.items() if key not in {"runtime", "handle_dag"}}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = [
    "COMPARISON_CONTRACT_VERSION",
    "comparable_contract_payload",
    "comparison_contract",
]
