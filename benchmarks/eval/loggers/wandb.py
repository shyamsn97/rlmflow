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
