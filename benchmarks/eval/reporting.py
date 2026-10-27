"""The single persisted result model for benchmark environments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Generic, TypeVar

from benchmarks.eval.telemetry import RowTelemetry
from benchmarks.eval.types import Prediction, Row, Score

MetricsT = TypeVar("MetricsT")


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    wall_time_seconds: float = 0.0
    llm_turns: int = 0
    agents: int = 0
    nodes: int = 0
    environment_actions_per_llm_turn: float = 0.0


@dataclass(frozen=True)
class EnvironmentReport(Generic[MetricsT]):
    """One runner's complete result for one environment."""

    benchmark: str
    environment_id: str
    runner: str
    model: str
    seed: int | None
    answer: str
    score: Score
    description_sha256: str
    observation_sha256: str
    runtime: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    environment: MetricsT | dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentReport[dict[str, Any]]:
        return cls(
            **{
                **data,
                "score": Score(**data["score"]),
                "runtime": RuntimeMetrics(**data.get("runtime", {})),
            }
        )

    @classmethod
    def from_row(
        cls,
        *,
        benchmark: str,
        row: Row,
        description_sha256: str,
        observation_sha256: str,
    ) -> EnvironmentReport[dict[str, Any]]:
        telemetry = RowTelemetry.from_row(row)
        metrics = row.prediction.metrics
        runtime = metrics.get("runtime", {})
        graph = telemetry.graph
        environment = dict(metrics.get("environment", {}))
        llm_turns = int(
            runtime.get(
                "llm_turns",
                metrics.get("llm_turns", graph.get("llm_turns", 0)),
            )
        )
        environment_actions = environment.get(
            "total_actions",
            environment.get("accepted_actions", len(environment.get("actions", []))),
        )
        reasoning = row.prediction.usage.get("reasoning_tokens", 0)
        return cls(
            benchmark=benchmark,
            environment_id=row.example_id,
            runner=row.runner,
            model=row.model,
            seed=row.seed,
            answer=row.prediction.answer,
            score=row.score,
            description_sha256=description_sha256,
            observation_sha256=observation_sha256,
            runtime=RuntimeMetrics(
                input_tokens=telemetry.input_tokens,
                output_tokens=telemetry.output_tokens,
                reasoning_tokens=int(reasoning),
                wall_time_seconds=telemetry.time_seconds,
                llm_turns=llm_turns,
                agents=int(
                    runtime.get(
                        "agents",
                        metrics.get("agents", graph.get("agents", 0)),
                    )
                ),
                nodes=int(
                    runtime.get(
                        "nodes",
                        metrics.get("nodes", graph.get("nodes", 0)),
                    )
                ),
                environment_actions_per_llm_turn=(
                    float(environment_actions) / llm_turns if llm_turns else 0.0
                ),
            ),
            environment={
                **environment,
                **({"delegation": telemetry.delegation} if telemetry.delegation else {}),
            },
            artifacts=telemetry.artifacts,
            error=telemetry.error,
            timestamp=row.timestamp,
        )

    def as_prediction(self) -> Prediction:
        """Compatibility helper for isolated readers of legacy row artifacts."""
        return Prediction(
            answer=self.answer,
            usage={
                "input_tokens": self.runtime.input_tokens,
                "output_tokens": self.runtime.output_tokens,
                "reasoning_tokens": self.runtime.reasoning_tokens,
            },
            metrics={
                "time_seconds": self.runtime.wall_time_seconds,
                "environment": self.environment,
                "graph": {
                    "llm_turns": self.runtime.llm_turns,
                    "agents": self.runtime.agents,
                    "nodes": self.runtime.nodes,
                },
            },
            artifacts=self.artifacts,
            error=self.error,
        )


def write_reports(
    *,
    root: Path,
    name: str,
    description: str,
    reports: list[EnvironmentReport],
    persist_rows: bool = True,
    persist_summary: bool = True,
) -> None:
    """Persist all report formats from the same normalized rows."""
    root.mkdir(parents=True, exist_ok=True)
    if persist_rows:
        with (root / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for report in reports:
                handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
    if persist_summary:
        summary = summarize_reports(name=name, description=description, reports=reports)
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    columns, rows = report_table(reports)
    markdown = [
        f"# {name} benchmark",
        "",
        description,
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *[
            "| "
            + " | ".join(_markdown_cell(columns[index], value) for index, value in enumerate(row))
            + " |"
            for row in rows
        ],
    ]
    (root / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_html_cell(columns[index], value)}</td>" for index, value in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    (root / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{escape(name)} benchmark</title>"
        "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:.35rem;text-align:left}th{position:sticky;"
        "top:0;background:#fff}</style>"
        f"<h1>{escape(name)} benchmark</h1><p>{escape(description)}</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>",
        encoding="utf-8",
    )


def load_environment_reports(path: Path) -> list[EnvironmentReport]:
    """Read current reports and isolate conversion of legacy Row artifacts."""
    if not path.exists():
        return []
    reports: list[EnvironmentReport] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if "benchmark" in data and "environment_id" in data:
            reports.append(EnvironmentReport.from_dict(data))
            continue
        row = Row.from_dict(data)
        contract = row.prediction.metrics.get("benchmark_task", {})
        reports.append(
            EnvironmentReport.from_row(
                benchmark=row.dataset,
                row=row,
                description_sha256=str(
                    contract.get(
                        "description_sha256",
                        contract.get("task_prompt_sha256", ""),
                    )
                ),
                observation_sha256=str(
                    contract.get(
                        "observation_sha256",
                        contract.get("inputs_sha256", ""),
                    )
                ),
            )
        )
    return reports


def summarize_reports(
    *,
    name: str,
    description: str,
    reports: list[EnvironmentReport],
) -> dict[str, Any]:
    grouped: dict[str, list[EnvironmentReport]] = {}
    for report in reports:
        grouped.setdefault(report.runner, []).append(report)
    return {
        "benchmark": name,
        "description": description,
        "count": len(reports),
        "by_runner": {
            runner: {
                "count": len(items),
                "score": sum(item.score.value for item in items) / len(items),
                "correct": sum(item.score.correct is True for item in items),
                "errors": sum(item.error is not None for item in items),
                "input_tokens": sum(item.runtime.input_tokens for item in items),
                "output_tokens": sum(item.runtime.output_tokens for item in items),
                "reasoning_tokens": sum(item.runtime.reasoning_tokens for item in items),
                "wall_time_seconds": sum(item.runtime.wall_time_seconds for item in items),
                "environment_actions_per_llm_turn": (
                    sum(item.runtime.environment_actions_per_llm_turn for item in items)
                    / len(items)
                ),
            }
            for runner, items in sorted(grouped.items())
        },
        "updated_at": datetime.now(UTC).isoformat(),
    }


def report_table(
    reports: list[EnvironmentReport],
) -> tuple[list[str], list[list[Any]]]:
    columns = [
        "environment",
        "runner",
        "description hash",
        "observation hash",
        "score",
        "correct",
        "error",
        "input tokens",
        "output tokens",
        "reasoning tokens",
        "seconds",
        "LLM turns",
        "actions / LLM turn",
        "agents",
        "nodes",
        "children",
        "query calls",
        "launch batches",
        "retrieved",
        "duplicates",
        "shard answers",
        "actions",
        "forks",
        "selected",
        "native score",
        "completed",
        "artifacts",
        "answer",
    ]
    rows: list[list[Any]] = []
    for report in reports:
        environment = _environment_dict(report.environment)
        delegation = _mapping(environment.get("delegation"))
        primary = _mapping(environment.get("primary"))
        search = _mapping(environment.get("candidate_search"))
        rows.append(
            [
                report.environment_id,
                report.runner,
                report.description_sha256,
                report.observation_sha256,
                round(report.score.value, 4),
                report.score.correct,
                report.error or "",
                report.runtime.input_tokens,
                report.runtime.output_tokens,
                report.runtime.reasoning_tokens,
                round(report.runtime.wall_time_seconds, 3),
                report.runtime.llm_turns,
                round(report.runtime.environment_actions_per_llm_turn, 3),
                report.runtime.agents,
                report.runtime.nodes,
                delegation.get("children", environment.get("children", 0)),
                delegation.get("query_calls", environment.get("query_calls", 0)),
                delegation.get(
                    "launch_batches",
                    delegation.get("concurrent_launch_batches", 0),
                ),
                delegation.get("retrieved_child_results", 0),
                delegation.get("duplicate_launches", 0),
                json.dumps(
                    environment.get("per_shard_answers", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                environment.get("total_actions", environment.get("accepted_actions", 0)),
                environment.get("instances_forked", 0),
                search.get("selected_candidate_id", ""),
                primary.get("score", primary.get("gems_collected", "")),
                primary.get("completed", primary.get("exact", primary.get("game_won", ""))),
                ", ".join(f"{name}={path}" for name, path in sorted(report.artifacts.items())),
                " ".join(report.answer.split())[:300],
            ]
        )
    return columns, rows


def _environment_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    return dict(to_dict()) if to_dict is not None else asdict(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _markdown_cell(column: str, value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    if column != "artifacts" or not text:
        return text
    links = []
    for item in text.split(", "):
        name, separator, path = item.partition("=")
        links.append(f"[{name}]({path})" if separator else item)
    return ", ".join(links)


def _html_cell(column: str, value: Any) -> str:
    text = str(value)
    if column != "artifacts" or not text:
        return escape(text)
    links = []
    for item in text.split(", "):
        name, separator, path = item.partition("=")
        if not separator:
            links.append(escape(item))
            continue
        links.append(f"<a href='{escape(path, quote=True)}'>{escape(name)}</a>")
    return ", ".join(links)


__all__ = [
    "EnvironmentReport",
    "RuntimeMetrics",
    "load_environment_reports",
    "report_table",
    "summarize_reports",
    "write_reports",
]
