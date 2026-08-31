from __future__ import annotations

import json

from benchmarks.eval.matrix import build_matrix, render_markdown, write_matrix
from benchmarks.eval.types import Prediction, Row, Score


def _row(
    *,
    runner: str,
    seed: int,
    score: float,
    example_id: str = "delegation_parallelqa_63",
    answer: str = "8408.63",
    expected: str = "8408.63",
    children: int = 0,
    error: str | None = None,
) -> Row:
    return Row(
        run_id="run",
        dataset="delegation-iteration-ten",
        example_id=example_id,
        runner=runner,
        model="gpt-5-mini",
        seed=seed,
        prediction=Prediction(
            answer=answer,
            usage={"input_tokens": 100, "output_tokens": 20},
            metrics={
                "time_seconds": 3.0,
                "graph": {"agents": 1 + children, "llm_turns": 2, "max_depth": int(bool(children))},
                "delegation": {"children": children},
            },
            error=None,
        ),
        score=Score(
            value=score,
            correct=score >= 1.0,
            details={"expected": expected} | ({"error": error} if error else {}),
        ),
        metadata={"iteration_role": "local_arithmetic_control"},
    )


def test_matrix_grids_every_question_by_runner_and_seed():
    rows = [
        _row(runner="rlmflow-local", seed=0, score=1.0),
        _row(runner="rlmflow-local", seed=1, score=0.0, answer="568.375"),
        _row(runner="official-rlm", seed=0, score=1.0),
        _row(runner="official-rlm", seed=1, score=1.0),
    ]

    matrix = build_matrix(rows)

    assert matrix["runners"] == ["official-rlm", "rlmflow-local"]
    assert matrix["seeds"] == ["0", "1"]
    task = matrix["tasks"][0]
    assert task["runners"]["rlmflow-local"]["mean_score"] == 0.5
    assert task["runners"]["official-rlm"]["mean_score"] == 1.0
    assert task["delta"] == 0.5
    # A task that scores differently across seeds of one runner cannot grade a prompt
    # change, so the matrix names it rather than averaging the instability away.
    assert task["flaky"] == ["rlmflow-local"]
    assert matrix["summary"]["rlmflow-local"]["spread"] == 1.0


def test_matrix_reports_missing_runs_instead_of_scoring_them_zero():
    rows = [
        _row(runner="rlmflow-local", seed=0, score=1.0),
        _row(runner="official-rlm", seed=0, score=1.0),
        _row(runner="official-rlm", seed=1, score=1.0),
    ]

    matrix = build_matrix(rows)
    markdown = render_markdown(matrix)

    task = matrix["tasks"][0]
    assert task["cells"]["rlmflow-local"]["1"] is None
    assert task["delta"] == 0.0
    assert "—" in markdown


def test_matrix_flags_numeric_answers_graded_wrong_on_formatting_alone():
    # `official-rlm` returned 8408.629999999999475 for an expected 8408.63 and was
    # graded incorrect; reading that as a reasoning failure would misdirect prompt work.
    rows = [
        _row(runner="official-rlm", seed=0, score=0.0, answer="8408.629999999999475"),
        _row(runner="rlmflow-local", seed=0, score=0.0, answer="568.375"),
    ]

    markdown = render_markdown(build_matrix(rows))

    assert "a scorer formatting artifact, not a reasoning failure" in markdown
    assert markdown.count("scorer formatting artifact") == 1


def test_matrix_marks_errored_attempts_distinctly_from_wrong_answers():
    rows = [
        _row(
            runner="rlmflow-local", seed=0, score=0.0, error="APIConnectionError: Connection error."
        ),
        _row(runner="official-rlm", seed=0, score=1.0),
    ]

    matrix = build_matrix(rows)
    markdown = render_markdown(matrix)

    assert matrix["summary"]["rlmflow-local"]["errors"] == 1
    assert "| ERR |" in markdown
    assert "APIConnectionError" in markdown


def test_crashed_attempts_are_unscored_rather_than_averaged_as_zero():
    # Three tracebacks are missing data. Averaging them as 0.00 grades the harness.
    rows = [
        _row(runner="official-rlm", seed=seed, score=0.0, error="Traceback (most recent call last)")
        for seed in (0, 1, 2)
    ] + [_row(runner="rlmflow-local", seed=seed, score=1.0) for seed in (0, 1, 2)]

    matrix = build_matrix(rows)

    official = matrix["summary"]["official-rlm"]
    assert official["unscored"] == 3
    assert official["mean_score"] is None
    assert matrix["tasks"][0]["delta"] is None
    assert "produced no gradeable answer" in render_markdown(matrix)


def test_grader_rejection_counts_as_zero_not_as_a_crash():
    # "goal not reached" means the agent returned a real but invalid plan.
    rows = [
        _row(runner="rlmflow-local", seed=0, score=0.0, error="goal not reached"),
        _row(runner="rlmflow-local", seed=1, score=1.0),
        _row(runner="official-rlm", seed=0, score=1.0),
        _row(runner="official-rlm", seed=1, score=1.0),
    ]

    matrix = build_matrix(rows)
    markdown = render_markdown(matrix)

    assert matrix["summary"]["rlmflow-local"]["unscored"] == 0
    assert matrix["tasks"][0]["runners"]["rlmflow-local"]["mean_score"] == 0.5
    assert "was rejected by the grader — s0: goal not reached" in markdown
    assert "ERR" not in markdown


def test_matrix_writes_json_and_markdown(tmp_path):
    rows = [
        _row(runner="rlmflow-local", seed=0, score=1.0),
        _row(
            runner="official-rlm",
            seed=0,
            score=0.0,
            example_id="delegation_codeqa_17",
            answer="wrong",
            expected="right",
            children=2,
        ),
    ]

    json_path, markdown_path = write_matrix(rows, tmp_path, title="Ten-task matrix")

    assert json.loads(json_path.read_text())["runners"] == ["official-rlm", "rlmflow-local"]
    markdown = markdown_path.read_text()
    assert "# Ten-task matrix" in markdown
    assert "## Per-question matrix" in markdown
    assert "## Per-question rundown" in markdown
    assert "`codeqa_17`" in markdown
    assert markdown == render_markdown(build_matrix(rows), title="Ten-task matrix")
