from __future__ import annotations

import json

from benchmarks.eval.compare import compare_runs, render_markdown, write_comparison
from benchmarks.eval.types import Prediction, Row, Score


def _row(
    *,
    runner: str,
    score: float,
    correct: bool,
    agents: int,
    tokens: int,
) -> Row:
    return Row(
        run_id=runner,
        dataset="delegation_task",
        example_id="problem-1",
        runner=runner,
        model="gpt-5-mini",
        seed=0,
        prediction=Prediction(
            answer="answer",
            usage={"input_tokens": tokens, "output_tokens": 10},
            metrics={
                "time_seconds": 2.5,
                "graph": {"agents": agents, "max_depth": agents - 1},
            },
        ),
        score=Score(value=score, correct=correct),
    )


def test_compare_runs_builds_paired_summary_and_delta():
    comparison = compare_runs(
        {
            "rlmflow": [
                _row(
                    runner="rlmflow-local",
                    score=0.25,
                    correct=False,
                    agents=1,
                    tokens=100,
                )
            ],
            "official": [
                _row(
                    runner="official-rlm",
                    score=1.0,
                    correct=True,
                    agents=2,
                    tokens=200,
                )
            ],
        }
    )

    assert comparison["paired_tasks"] == 1
    assert comparison["tasks"][0]["score_delta"] == 0.75
    assert comparison["summary"]["official"]["accuracy"] == 1.0
    assert comparison["summary"]["official"]["delegated_tasks"] == 1
    assert comparison["summary"]["official"]["subcalls"] == 0


def test_comparison_writes_json_and_markdown(tmp_path):
    runs = {
        "rlmflow": [
            _row(
                runner="rlmflow-local",
                score=0.0,
                correct=False,
                agents=1,
                tokens=100,
            )
        ],
        "official": [
            _row(
                runner="official-rlm",
                score=1.0,
                correct=True,
                agents=2,
                tokens=200,
            )
        ],
    }

    json_path, markdown_path = write_comparison(runs, tmp_path)

    assert json.loads(json_path.read_text())["paired_tasks"] == 1
    markdown = markdown_path.read_text()
    assert "# Benchmark comparison" in markdown
    assert "official score" in markdown
    assert "Score delta (official − rlmflow)" in markdown
    assert markdown == render_markdown(compare_runs(runs))
