"""Pure summary helpers for benchmark rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from benchmarks.eval.telemetry import RowTelemetry
from benchmarks.eval.types import Row
from rlmflow import Graph
from rlmflow.graph.nodes import AGENT_START, LLM_OUTPUT


def graph_metrics(graph: Graph | None) -> dict[str, Any]:
    if graph is None:
        return {}
    nodes = graph.nodes()
    agents = graph.nodes(AGENT_START)
    return {
        "agents": len(agents),
        "nodes": len(nodes),
        "llm_turns": len(graph.nodes(LLM_OUTPUT)),
        "max_depth": max((graph.depth(agent) for agent in agents), default=0),
        "max_branching": max((len(graph.children(agent)) for agent in agents), default=0),
    }


def summarize(rows: list[Row]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "overall": {}, "by_runner": {}, "by_dataset": {}}

    def group(items: list[Row]) -> dict[str, Any]:
        graded = [row for row in items if row.score.correct is not None]
        telemetry = [RowTelemetry.from_row(row) for row in items]
        correct = sum(1 for row in graded if row.score.correct is True)
        accuracy = correct / len(graded) if graded else None
        graphs = []
        for row in items:
            graph = row.prediction.metrics.get("graph")
            if isinstance(graph, dict) and graph:
                graphs.append(graph)

        def graph_mean(key: str) -> float | None:
            values = [graph.get(key) for graph in graphs]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def graph_max(key: str) -> float | None:
            values = [graph.get(key) for graph in graphs]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return max(numeric) if numeric else None

        def metric_mean(section: str, key: str) -> float | None:
            values = [row.prediction.metrics.get(section, {}).get(key) for row in items]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def delegation_mean(key: str, *, official_key: str | None = None) -> float | None:
            del official_key
            values = []
            for item in telemetry:
                values.append(item.delegation.get(key))
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def environment_mean(key: str) -> float | None:
            values = [row.prediction.metrics.get("environment", {}).get(key) for row in items]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def candidate_mean(key: str) -> float | None:
            values = [
                row.prediction.metrics.get("environment", {}).get("candidate_search", {}).get(key)
                for row in items
            ]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def branch_utilization(row: Row) -> float | None:
            environment = row.prediction.metrics.get("environment", {})
            search = environment.get("candidate_search", {})
            candidates = search.get("candidates", [])
            if not isinstance(candidates, list) or not candidates:
                return None
            owners = {
                candidate.get("owner_id")
                for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("owner_id")
            }
            instances = int(environment.get("instances_forked", 0)) + 1
            return len(owners) / max(1, instances)

        def selected_depth(row: Row) -> int | None:
            search = row.prediction.metrics.get("environment", {}).get("candidate_search", {})
            selected_id = search.get("selected_candidate_id")
            return next(
                (
                    int(candidate["owner_depth"])
                    for candidate in search.get("candidates", [])
                    if isinstance(candidate, dict)
                    and candidate.get("id") == selected_id
                    and isinstance(candidate.get("owner_depth"), int)
                ),
                None,
            )

        def delegation_topology(row: Row) -> str:
            metrics = row.prediction.metrics
            delegation = metrics.get("delegation", {})
            rlm = metrics.get("rlm", {})
            if delegation.get("dependency_edge_count", 0):
                return "dependency-dag"
            if delegation.get("children", rlm.get("subcalls", 0)):
                return "delegation-tree"
            return "single-agent"

        token_total = sum(item.input_tokens + item.output_tokens for item in telemetry)
        wall_total = sum(item.time_seconds for item in telemetry)
        environment_counts = [
            _environment_action_count(row.prediction.metrics.get("environment", {}))
            for row in items
        ]
        environment_total = sum(environment_counts)
        candidate_total = sum(
            row.prediction.metrics.get("environment", {})
            .get("candidate_search", {})
            .get("evaluations_used", 0)
            for row in items
        )
        score_total = sum(row.score.value for row in items)

        return {
            "count": len(items),
            "graded_count": len(graded),
            "correct": correct,
            "incorrect": len(graded) - correct,
            "accuracy": accuracy,
            "accuracy_pct": accuracy * 100.0 if accuracy is not None else None,
            "score": mean(row.score.value for row in items),
            "errors": sum(1 for item in telemetry if item.error),
            "input_tokens": mean(item.input_tokens for item in telemetry),
            "output_tokens": mean(item.output_tokens for item in telemetry),
            "time_seconds": mean(item.time_seconds for item in telemetry),
            "compute_seconds": mean(
                row.metadata.get(
                    "aggregate_compute_seconds",
                    row.prediction.metrics.get("time_seconds", 0.0),
                )
                for row in items
            ),
            "score_per_1k_tokens": (score_total * 1_000 / token_total if token_total else None),
            "score_per_environment_action": (
                score_total / environment_total if environment_total else None
            ),
            "score_per_candidate_evaluation": (
                score_total / candidate_total if candidate_total else None
            ),
            "score_per_second": score_total / wall_total if wall_total else None,
            "graph_count": len(graphs),
            "graph_nodes": graph_mean("nodes"),
            "graph_agents": graph_mean("agents"),
            "graph_llm_turns": graph_mean("llm_turns"),
            "environment_actions_per_llm_turn": (
                environment_total / sum(graph.get("llm_turns", 0) for graph in graphs)
                if graphs and sum(graph.get("llm_turns", 0) for graph in graphs)
                else None
            ),
            "graph_max_depth": graph_max("max_depth"),
            "graph_max_branching": graph_max("max_branching"),
            "subdelegated": sum(1 for graph in graphs if graph.get("agents", 0) > 1),
            "delegation_topologies": dict(
                sorted(Counter(delegation_topology(row) for row in items).items())
            ),
            "delegation_children": delegation_mean(
                "children",
                official_key="recursive_subcalls",
            ),
            "delegation_dependency_edges": metric_mean(
                "delegation",
                "dependency_edge_count",
            ),
            "delegation_retrieved_results": delegation_mean(
                "retrieved_child_results",
                official_key="recursive_subcalls",
            ),
            "delegation_retrieval_ratio": delegation_mean(
                "retrieval_ratio",
                official_key="recursive_subcalls",
            ),
            "delegation_recursive_launches": delegation_mean(
                "recursive_launch_count",
                official_key="recursive_subcalls",
            ),
            "delegation_dependency_chain_depth": graph_max_from_rows(
                items,
                section="delegation",
                key="dependency_chain_depth",
            ),
            "delegation_max_launch_batch": metric_mean(
                "delegation",
                "max_launch_batch",
            ),
            "delegation_max_ready_launch_batch": metric_mean(
                "delegation",
                "max_ready_launch_batch",
            ),
            "delegation_concurrent_launch_batches": metric_mean(
                "delegation",
                "concurrent_launch_batches",
            ),
            "environment_actions": mean(environment_counts),
            "environment_primary_actions": environment_mean("primary_actions"),
            "environment_probe_actions": environment_mean("probe_actions"),
            "environment_instances": environment_mean("instances_forked"),
            "selection_compliance": environment_mean("selection_compliance"),
            "candidate_count": candidate_mean("evaluations_used"),
            "candidate_oracle_best_score": candidate_mean("oracle_best_score"),
            "candidate_selection_regret": candidate_mean("selection_regret"),
            "candidate_diversity": candidate_mean("candidate_diversity"),
            "candidate_outcome_diversity": candidate_mean("outcome_diversity"),
            "candidate_branch_utilization": (
                mean(values)
                if (
                    values := [
                        value for row in items if (value := branch_utilization(row)) is not None
                    ]
                )
                else None
            ),
            "candidate_selected_depth": (
                mean(values)
                if (
                    values := [value for row in items if (value := selected_depth(row)) is not None]
                )
                else None
            ),
        }

    by_runner: dict[str, list[Row]] = defaultdict(list)
    by_dataset: dict[str, list[Row]] = defaultdict(list)
    by_pair: dict[str, list[Row]] = defaultdict(list)
    by_route: dict[str, list[Row]] = defaultdict(list)
    by_domain: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_runner[row.runner].append(row)
        by_dataset[row.dataset].append(row)
        by_pair[f"{row.runner}/{row.dataset}"].append(row)
        if route := row.metadata.get("route_hypothesis"):
            by_route[str(route)].append(row)
        if domain := row.metadata.get("domain"):
            by_domain[str(domain)].append(row)
    return {
        "count": len(rows),
        "overall": group(rows),
        "by_runner": {key: group(items) for key, items in sorted(by_runner.items())},
        "by_dataset": {key: group(items) for key, items in sorted(by_dataset.items())},
        "by_runner_dataset": {key: group(items) for key, items in sorted(by_pair.items())},
        "by_route": {key: group(items) for key, items in sorted(by_route.items())},
        "by_domain": {key: group(items) for key, items in sorted(by_domain.items())},
    }


def _environment_action_count(environment: dict) -> int:
    value = environment.get("total_actions", environment.get("accepted_actions"))
    if isinstance(value, (int, float)):
        return int(value)
    return len(environment.get("actions", []))


def graph_max_from_rows(
    rows: list[Row],
    *,
    section: str,
    key: str,
) -> float | None:
    values = [row.prediction.metrics.get(section, {}).get(key) for row in rows]
    numeric = [value for value in values if isinstance(value, (int, float))]
    return max(numeric) if numeric else None


__all__ = ["graph_metrics", "summarize"]
