"""Validate the offline conformance manifests without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import benchmarks.eval.tasks  # noqa: F401
from benchmarks.eval import DATASETS
from benchmarks.eval.acceptance import DEFAULT_ACCEPTANCE_POLICY
from benchmarks.eval.loggers.jsonl import write_json

CONFORMANCE_DATASETS = ("minimaze", "sudoku_search")


def calibrate() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    accepted = True
    for name in CONFORMANCE_DATASETS:
        dataset = DATASETS.make(name)
        pilot = dataset.examples(split="pilot", limit=None, seed=0)
        full = dataset.examples(split="full", limit=None, seed=0)
        all_examples = (*pilot, *full)
        environments = [example.metadata.get("environment") for example in all_examples]
        bounded = all(
            isinstance(environment, dict)
            and isinstance(environment.get("max_actions"), int)
            and environment["max_actions"] > 0
            and isinstance(environment.get("candidate_budget"), int)
            and environment["candidate_budget"] > 0
            for environment in environments
        )
        explicit_selection = all(
            "submit_candidate" in example.prompt and "select_candidate" in example.prompt
            for example in all_examples
        )
        disjoint = set(example.id for example in pilot).isdisjoint(example.id for example in full)
        row_accepted = bool(pilot and full and bounded and explicit_selection and disjoint)
        accepted &= row_accepted
        checks.append(
            {
                "dataset": name,
                "pilot": [example.id for example in pilot],
                "full": [example.id for example in full],
                "accepted": row_accepted,
                "gates": {
                    "nonempty_splits": bool(pilot and full),
                    "bounded_shared_budgets": bounded,
                    "explicit_selection": explicit_selection,
                    "pilot_full_ids_disjoint": disjoint,
                },
            }
        )
    policy = DEFAULT_ACCEPTANCE_POLICY
    return {
        "schema_version": 2,
        "model_calls": 0,
        "accepted": accepted,
        "checks": checks,
        "acceptance_policy": {
            "required_runners": sorted(policy.required_runners),
            "required_domains": sorted(policy.required_domains),
            "required_seeds": sorted(policy.required_seeds),
            "minimum_selection_compliance": policy.minimum_selection_compliance,
            "depth0_range": [policy.depth0_minimum, policy.depth0_maximum],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/eval/calibration/conformance.json"),
    )
    args = parser.parse_args()
    report = calibrate()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
