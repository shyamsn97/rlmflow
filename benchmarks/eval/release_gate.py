"""Evaluate and enforce empirical visual-benchmark release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks.eval.acceptance import DEFAULT_ACCEPTANCE_POLICY
from benchmarks.eval.loggers.jsonl import load_rows, write_json
from benchmarks.eval.matrix import build_matrix
from benchmarks.eval.telemetry import EnvironmentMetricsView, RowTelemetry
from benchmarks.eval.types import Row

GATE_SCHEMA_VERSION = 1
DEFAULT_ACCEPTANCE_PATH = Path("benchmarks/eval/calibration/conformance_acceptance.json")
REQUIRED_RUNNERS = DEFAULT_ACCEPTANCE_POLICY.required_runners
REQUIRED_DOMAINS = DEFAULT_ACCEPTANCE_POLICY.required_domains
REQUIRED_SEEDS = DEFAULT_ACCEPTANCE_POLICY.required_seeds
REQUIRED_ROWS_PER_ARM_DOMAIN = DEFAULT_ACCEPTANCE_POLICY.rows_per_runner_domain


def evaluate_pilot(rows: list[Row], *, max_row_seconds: float = 300.0) -> dict[str, Any]:
    matrix = build_matrix(rows)
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.metadata.get("domain", "")), row.runner)].append(row)

    protocol: dict[str, Any] = {}
    protocol_ok = True
    for domain in sorted(REQUIRED_DOMAINS):
        for runner in sorted(REQUIRED_RUNNERS):
            items = grouped.get((domain, runner), [])
            selected = sum(_selection_made(row) for row in items)
            compliance = selected / len(items) if items else 0.0
            infra_errors = sum(_infra_error(row) for row in items)
            seeds = {str(row.seed) for row in items}
            max_seconds = max(
                (float(row.prediction.metrics.get("time_seconds", 0.0)) for row in items),
                default=0.0,
            )
            accepted = (
                len(items) >= REQUIRED_ROWS_PER_ARM_DOMAIN
                and seeds >= REQUIRED_SEEDS
                and compliance >= DEFAULT_ACCEPTANCE_POLICY.minimum_selection_compliance
                and infra_errors == 0
                and max_seconds
                < min(
                    max_row_seconds,
                    DEFAULT_ACCEPTANCE_POLICY.maximum_row_seconds,
                )
            )
            protocol_ok &= accepted
            protocol[f"{domain}/{runner}"] = {
                "rows": len(items),
                "seeds": sorted(seeds),
                "selection_compliance": compliance,
                "infra_errors": infra_errors,
                "max_row_seconds": max_seconds,
                "accepted": accepted,
            }

    difficulty: dict[str, Any] = {}
    difficulty_ok = True
    for domain in sorted(REQUIRED_DOMAINS):
        baseline = grouped.get((domain, DEFAULT_ACCEPTANCE_POLICY.baseline_runner), [])
        scores = [_native_score(row) for row in baseline if not _infra_error(row)]
        scores = [score for score in scores if score is not None]
        different_outcomes = any(
            _outcome_diversity(row) > 1 / max(1, _candidate_count(row))
            for row in rows
            if row.metadata.get("domain") == domain
        )
        assessment = DEFAULT_ACCEPTANCE_POLICY.assess_scores(
            scores,
            [],
            candidate_variation=different_outcomes,
        )
        baseline_median = assessment["depth0_median"]
        floor_share = assessment["floor_share"] if scores else 1.0
        ceiling_share = assessment["ceiling_share"] if scores else 1.0
        accepted = (
            baseline_median is not None
            and assessment["depth0_target_30_70"]
            and floor_share <= DEFAULT_ACCEPTANCE_POLICY.maximum_floor_share
            and ceiling_share <= DEFAULT_ACCEPTANCE_POLICY.maximum_ceiling_share
            and different_outcomes
        )
        difficulty_ok &= accepted
        difficulty[domain] = {
            "depth0_median": baseline_median,
            "floor_share": floor_share,
            "ceiling_share": ceiling_share,
            "different_candidate_outcomes": different_outcomes,
            "accepted": accepted,
        }

    claim_gates = matrix.get("acceptance_gates", {})
    claim_delegation_benefit = bool(claim_gates) and all(
        bool(claim_gates.get(domain, {}).get("claim_delegation_benefit"))
        for domain in REQUIRED_DOMAINS
    )
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "approved": protocol_ok and difficulty_ok,
        "protocol_health": protocol,
        "difficulty": difficulty,
        "delegation_claim_approved": claim_delegation_benefit,
        "claim_policy": (
            "Benchmark release does not require positive lift. A delegation-benefit "
            "claim requires depth 2 to beat depth 0 under the shared contract."
        ),
        "row_count": len(rows),
        "run_ids": sorted({row.run_id for row in rows}),
    }


def sign_report(report: dict[str, Any], *, signed_by: str) -> dict[str, Any]:
    if not signed_by.strip():
        raise ValueError("signed_by must be non-empty")
    unsigned = {**report, "signed_by": signed_by.strip()}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **unsigned,
        "signature": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
    }


def verify_report(report: dict[str, Any]) -> None:
    signature = report.get("signature")
    if not isinstance(signature, str) or not signature.startswith("sha256:"):
        raise RuntimeError("pilot acceptance artifact is unsigned")
    unsigned = {key: value for key, value in report.items() if key != "signature"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    if signature != expected:
        raise RuntimeError("pilot acceptance artifact signature does not match")
    if report.get("schema_version") != GATE_SCHEMA_VERSION:
        raise RuntimeError("unsupported pilot acceptance schema")
    if report.get("approved") is not True:
        raise RuntimeError("pilot empirical gates have not passed")


def _selection_made(row: Row) -> bool:
    return EnvironmentMetricsView.from_prediction(row.prediction).selection_made


def _candidate_count(row: Row) -> int:
    search = EnvironmentMetricsView.from_prediction(row.prediction).candidate_search
    return int(search.get("evaluations_used", 0))


def _outcome_diversity(row: Row) -> float:
    search = EnvironmentMetricsView.from_prediction(row.prediction).candidate_search
    value = search.get("outcome_diversity", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _native_score(row: Row) -> float | None:
    primary = EnvironmentMetricsView.from_prediction(row.prediction).primary
    value = primary.get("score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    return None


def _infra_error(row: Row) -> bool:
    telemetry = RowTelemetry.from_row(row)
    return telemetry.infra_error or bool(telemetry.error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--require-pilot", action="store_true")
    group.add_argument("--evaluate-run", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACCEPTANCE_PATH)
    parser.add_argument("--signed-by")
    args = parser.parse_args()

    if args.require_pilot:
        if not args.output.is_file():
            raise SystemExit(
                f"held-out suite is gated: missing accepted pilot artifact {args.output}"
            )
        report = json.loads(args.output.read_text(encoding="utf-8"))
        verify_report(report)
        print(f"accepted pilot gate: {args.output}")
        return 0

    rows_path = args.evaluate_run
    if rows_path.is_dir():
        rows_path = rows_path / "rows.jsonl"
    rows = load_rows(rows_path)
    report = evaluate_pilot(rows)
    if args.signed_by:
        report = sign_report(report, signed_by=args.signed_by)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
