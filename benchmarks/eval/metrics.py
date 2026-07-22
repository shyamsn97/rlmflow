"""Pure summary helpers for benchmark rows."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlmflow import Graph

from benchmarks.eval.types import Row


def graph_metrics(graph: "Graph | None") -> dict[str, Any]:
    if graph is None:
        return {}
    agents = list(graph.walk())
    node_counts = [len(agent.nodes) for agent in agents]
    child_counts = [len(agent.children) for agent in agents]
    llm_turns = sum(
        1 for agent in agents for node in agent.nodes if node.type == "llm_output"
    )
    return {
        "agents": len(agents),
        "nodes": sum(node_counts),
        "llm_turns": llm_turns,
        "max_depth": max((agent.depth for agent in agents), default=0),
        "max_branching": max(child_counts, default=0),
    }


def summarize(rows: list[Row]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "overall": {}, "by_runner": {}, "by_dataset": {}}

    def group(items: list[Row]) -> dict[str, Any]:
        graded = [row for row in items if row.score.correct is not None]
        correct = sum(1 for row in graded if row.score.correct is True)
        accuracy = correct / len(graded) if graded else None
        graphs = [
            graph
            for row in items
            if isinstance((graph := row.prediction.metrics.get("graph")), dict) and graph
        ]

        def graph_mean(key: str) -> float | None:
            values = [graph.get(key) for graph in graphs]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return mean(numeric) if numeric else None

        def graph_max(key: str) -> float | None:
            values = [graph.get(key) for graph in graphs]
            numeric = [value for value in values if isinstance(value, (int, float))]
            return max(numeric) if numeric else None

        return {
            "count": len(items),
            "graded_count": len(graded),
            "correct": correct,
            "incorrect": len(graded) - correct,
            "accuracy": accuracy,
            "accuracy_pct": accuracy * 100.0 if accuracy is not None else None,
            "score": mean(row.score.value for row in items),
            "errors": sum(1 for row in items if row.prediction.error),
            "input_tokens": mean(row.prediction.usage.get("input_tokens", 0) for row in items),
            "output_tokens": mean(row.prediction.usage.get("output_tokens", 0) for row in items),
            "time_seconds": mean(row.prediction.metrics.get("time_seconds", 0.0) for row in items),
            "graph_count": len(graphs),
            "graph_nodes": graph_mean("nodes"),
            "graph_agents": graph_mean("agents"),
            "graph_llm_turns": graph_mean("llm_turns"),
            "graph_max_depth": graph_max("max_depth"),
            "graph_max_branching": graph_max("max_branching"),
            "subdelegated": sum(1 for graph in graphs if graph.get("agents", 0) > 1),
        }

    by_runner: dict[str, list[Row]] = defaultdict(list)
    by_dataset: dict[str, list[Row]] = defaultdict(list)
    by_pair: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_runner[row.runner].append(row)
        by_dataset[row.dataset].append(row)
        by_pair[f"{row.runner}/{row.dataset}"].append(row)
    return {
        "count": len(rows),
        "overall": group(rows),
        "by_runner": {key: group(items) for key, items in sorted(by_runner.items())},
        "by_dataset": {key: group(items) for key, items in sorted(by_dataset.items())},
        "by_runner_dataset": {
            key: group(items) for key, items in sorted(by_pair.items())
        },
    }


__all__ = ["graph_metrics", "summarize"]
