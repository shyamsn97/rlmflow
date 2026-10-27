from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.reporting import EnvironmentReport, report_table
from benchmarks.eval.types import Prediction, Row, Score
from rlmflow import CODE, LLM_OUTPUT, Graph, LLMUsage, Node


def test_graph_metrics_count_agents_and_turns():
    g = Graph()
    root = g.spawn("task")
    for node in (
        Node(LLM_OUTPUT, text="a", usage=LLMUsage(input_tokens=10).as_dict()),
        Node(CODE, code="x = 1", output=""),
        Node(LLM_OUTPUT, text="b", usage=LLMUsage(input_tokens=11).as_dict()),
    ):
        g.append(root, node)

    metrics = graph_metrics(g)

    assert metrics["agents"] == 1
    assert metrics["llm_turns"] == 2
    assert g.usage([root]).input_tokens == 21


def test_environment_report_persists_action_efficiency_metrics():
    graph = {
        "llm_turns": 4,
        "agents": 1,
        "nodes": 12,
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

    assert restored.runtime.environment_actions_per_llm_turn == 1.5
    assert rows[0][columns.index("actions / LLM turn")] == 1.5
