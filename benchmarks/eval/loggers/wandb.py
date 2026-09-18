"""Optional Weights & Biases logger."""

from __future__ import annotations

from pathlib import Path

from benchmarks.eval import logger
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import Logger, Row


@logger("wandb")
class WandbLogger(Logger):
    def __init__(
        self,
        project: str = "rlmflow-eval",
        entity: str | None = None,
        root: Path | str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.root = Path(root) if root is not None else None
        self._wandb = None

    def start(self, config: dict) -> None:
        import wandb

        self._wandb = wandb
        wandb.init(project=self.project, entity=self.entity, config=config)

    def row(self, row: Row) -> None:
        if self._wandb is None:
            return
        graph = row.prediction.metrics.get("graph") or {}
        delegation = row.prediction.metrics.get("delegation") or {}
        environment = row.prediction.metrics.get("environment") or {}
        candidates = environment.get("candidate_search") or {}
        self._wandb.log(
            {
                "dataset": row.dataset,
                "runner": row.runner,
                "score": row.score.value,
                "correct": row.score.correct,
                "error": 1 if row.prediction.error else 0,
                "input_tokens": row.prediction.usage.get("input_tokens", 0),
                "output_tokens": row.prediction.usage.get("output_tokens", 0),
                "time_seconds": row.prediction.metrics.get("time_seconds", 0.0),
                "graph/nodes": graph.get("nodes"),
                "graph/agents": graph.get("agents"),
                "graph/llm_turns": graph.get("llm_turns"),
                "graph/max_depth": graph.get("max_depth"),
                "graph/max_branching": graph.get("max_branching"),
                "delegation/children": delegation.get("children"),
                "delegation/dependency_edges": delegation.get("dependency_edge_count"),
                "delegation/dependency_chain_depth": delegation.get("dependency_chain_depth"),
                "delegation/recursive_launches": delegation.get("recursive_launch_count"),
                "delegation/max_launch_batch": delegation.get("max_launch_batch"),
                "delegation/max_ready_launch_batch": delegation.get("max_ready_launch_batch"),
                "delegation/retrieval_ratio": delegation.get("retrieval_ratio"),
                "environment/primary_actions": environment.get("primary_actions"),
                "environment/probe_actions": environment.get("probe_actions"),
                "environment/instances": environment.get("instances_forked"),
                "candidates/evaluated": candidates.get("evaluations_used"),
                "candidates/selected_score": candidates.get("selected_score"),
                "candidates/oracle_best_score": candidates.get("oracle_best_score"),
                "candidates/selection_regret": candidates.get("selection_regret"),
                "candidates/diversity": candidates.get("candidate_diversity"),
                "candidates/outcome_diversity": candidates.get("outcome_diversity"),
            }
        )

    def summary(self, rows: list[Row]) -> None:
        if self._wandb is None:
            return
        summary = summarize(rows)
        overall = summary.get("overall", {})
        payload = _flatten_summary("overall", overall)
        for name, values in summary.get("by_dataset", {}).items():
            payload.update(_flatten_summary(f"benchmark/{name}", values))
        for name, values in summary.get("by_runner", {}).items():
            payload.update(_flatten_summary(f"runner/{name}", values))
        for name, values in summary.get("by_runner_dataset", {}).items():
            payload.update(_flatten_summary(f"runner_benchmark/{name}", values))
        self._wandb.summary.update(payload)
        self._wandb.log(
            {
                "summary/by_benchmark": _summary_table(self._wandb, summary.get("by_dataset", {})),
                "summary/by_runner": _summary_table(self._wandb, summary.get("by_runner", {})),
                "summary/by_runner_benchmark": _summary_table(
                    self._wandb, summary.get("by_runner_dataset", {})
                ),
            }
        )
        if self.root is not None:
            report_path = self.root / "report.md"
            if report_path.exists():
                self._wandb.save(str(report_path), policy="now")

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _flatten_summary(prefix: str, values: dict) -> dict[str, object]:
    return {
        f"{prefix}/count": values.get("count"),
        f"{prefix}/graded_count": values.get("graded_count"),
        f"{prefix}/correct": values.get("correct"),
        f"{prefix}/incorrect": values.get("incorrect"),
        f"{prefix}/accuracy": values.get("accuracy"),
        f"{prefix}/accuracy_pct": values.get("accuracy_pct"),
        f"{prefix}/score": values.get("score"),
        f"{prefix}/errors": values.get("errors"),
        f"{prefix}/input_tokens": values.get("input_tokens"),
        f"{prefix}/output_tokens": values.get("output_tokens"),
        f"{prefix}/time_seconds": values.get("time_seconds"),
        f"{prefix}/graph_count": values.get("graph_count"),
        f"{prefix}/graph_nodes": values.get("graph_nodes"),
        f"{prefix}/graph_agents": values.get("graph_agents"),
        f"{prefix}/graph_llm_turns": values.get("graph_llm_turns"),
        f"{prefix}/graph_max_depth": values.get("graph_max_depth"),
        f"{prefix}/graph_max_branching": values.get("graph_max_branching"),
        f"{prefix}/subdelegated": values.get("subdelegated"),
        f"{prefix}/delegation_recursive_launches": values.get("delegation_recursive_launches"),
        f"{prefix}/delegation_dependency_chain_depth": values.get(
            "delegation_dependency_chain_depth"
        ),
        f"{prefix}/delegation_max_ready_launch_batch": values.get(
            "delegation_max_ready_launch_batch"
        ),
        f"{prefix}/environment_primary_actions": values.get("environment_primary_actions"),
        f"{prefix}/environment_probe_actions": values.get("environment_probe_actions"),
        f"{prefix}/candidate_count": values.get("candidate_count"),
        f"{prefix}/candidate_oracle_best_score": values.get("candidate_oracle_best_score"),
        f"{prefix}/candidate_selection_regret": values.get("candidate_selection_regret"),
        f"{prefix}/candidate_branch_utilization": values.get("candidate_branch_utilization"),
        f"{prefix}/candidate_selected_depth": values.get("candidate_selected_depth"),
        f"{prefix}/candidate_diversity": values.get("candidate_diversity"),
        f"{prefix}/candidate_outcome_diversity": values.get("candidate_outcome_diversity"),
        f"{prefix}/score_per_candidate_evaluation": values.get("score_per_candidate_evaluation"),
    }


def _summary_table(wandb, values: dict[str, dict]):
    table = wandb.Table(
        columns=[
            "name",
            "rows",
            "graded",
            "correct",
            "pct_correct",
            "score",
            "errors",
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_time_seconds",
            "graph_rows",
            "avg_graph_nodes",
            "avg_graph_agents",
            "avg_graph_llm_turns",
            "max_graph_depth",
            "max_graph_branching",
            "subdelegated",
        ]
    )
    for name, item in values.items():
        table.add_data(
            name,
            item.get("count"),
            item.get("graded_count"),
            item.get("correct"),
            item.get("accuracy_pct"),
            item.get("score"),
            item.get("errors"),
            item.get("input_tokens"),
            item.get("output_tokens"),
            item.get("time_seconds"),
            item.get("graph_count"),
            item.get("graph_nodes"),
            item.get("graph_agents"),
            item.get("graph_llm_turns"),
            item.get("graph_max_depth"),
            item.get("graph_max_branching"),
            item.get("subdelegated"),
        )
    return table


__all__ = ["WandbLogger"]
