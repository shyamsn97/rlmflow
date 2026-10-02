from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.reporting import EnvironmentReport, report_table
from benchmarks.eval.types import Prediction, Row, Score
from rlmflow import (
    LLMOutput,
    LLMUsage,
    TransitionDecision,
    UserQuery,
    start,
)


def test_graph_metrics_count_turns_decisions_and_decision_costs():
    root = start("task")
    root.append(LLMOutput(usage=LLMUsage(input_tokens=10)))
    first_decision = TransitionDecision(
        usage=LLMUsage(input_tokens=3, output_tokens=1, reasoning_tokens=2),
        started_at=10.0,
        finished_at=10.25,
    )
    root.frontier.append(first_decision)
    first_decision.append(UserQuery())
    root.frontier.append(LLMOutput(usage=LLMUsage(input_tokens=11)))
    second_decision = TransitionDecision(
        selected_rule_id="retry",
        usage=LLMUsage(input_tokens=4, output_tokens=1),
        started_at=11.0,
        finished_at=11.5,
    )
    root.frontier.append(second_decision)
    second_decision.append(UserQuery())
    root.frontier.append(LLMOutput(usage=LLMUsage(input_tokens=12)))

    metrics = graph_metrics(root)

    assert metrics["llm_turns"] == 3
    assert metrics["decision_calls"] == 2
    assert metrics["decision_input_tokens"] == 7
    assert metrics["decision_output_tokens"] == 2
    assert metrics["decision_reasoning_tokens"] == 2
    assert metrics["decision_latency_seconds"] == 0.75
    assert root.usage.input_tokens == 40


def test_environment_report_persists_decision_and_action_efficiency_metrics():
    graph = {
        "llm_turns": 4,
        "agents": 1,
        "nodes": 12,
        "decision_calls": 2,
        "decision_input_tokens": 7,
        "decision_output_tokens": 2,
        "decision_reasoning_tokens": 2,
        "decision_latency_seconds": 0.75,
    }
    row = Row(
        run_id="run",
        dataset="interactive",
        example_id="maze",
        runner="rlmflow",
        model="fake",
        seed=0,
        prediction=Prediction(
            answer="done",
            usage={"input_tokens": 53, "output_tokens": 8, "reasoning_tokens": 2},
            metrics={
                "time_seconds": 2.0,
                "graph": graph,
                "environment": {"total_actions": 6},
            },
        ),
        score=Score(value=1.0, correct=True),
    )

    report = EnvironmentReport.from_row(
        benchmark="interactive",
        row=row,
        description_sha256="description",
        observation_sha256="observation",
    )
    restored = EnvironmentReport.from_dict(report.to_dict())
    columns, rows = report_table([restored])

    assert restored.runtime.decision_calls == 2
    assert restored.runtime.decision_input_tokens == 7
    assert restored.runtime.environment_actions_per_llm_turn == 1.5
    assert rows[0][columns.index("decision calls")] == 2
    assert rows[0][columns.index("actions / LLM turn")] == 1.5
