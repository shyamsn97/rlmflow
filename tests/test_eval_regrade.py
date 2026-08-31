from __future__ import annotations

from benchmarks.eval.loggers.jsonl import load_rows, write_rows
from benchmarks.eval.regrade import regrade_rows
from benchmarks.eval.types import Dataset, Example, Prediction, Row, Score


class _StrictThenLenientDataset(Dataset):
    """Scores exactly on the first pass and at two decimals on later ones."""

    name = "toy"

    def __init__(self) -> None:
        self.lenient = False

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        return [Example(id="problem-1", prompt="add", expected="9810.79")]

    def score(self, example: Example, prediction: Prediction) -> Score:
        if self.lenient:
            correct = round(float(prediction.answer), 2) == round(float(example.expected), 2)
        else:
            correct = prediction.answer == example.expected
        return Score(value=float(correct), correct=correct, details={"expected": example.expected})


class _RaisingDataset(_StrictThenLenientDataset):
    def score(self, example: Example, prediction: Prediction) -> Score:
        raise ValueError("expected 36 grid cells, got 0")


def _row(answer: str, *, score: float, error: str | None = None) -> Row:
    return Row(
        run_id="run",
        dataset="toy",
        example_id="problem-1",
        runner="official-rlm",
        model="gpt-5-mini",
        seed=0,
        prediction=Prediction(answer=answer, error=error),
        score=Score(value=score, correct=score >= 1.0),
    )


def test_regrade_rescores_saved_answers_without_a_model():
    dataset = _StrictThenLenientDataset()
    dataset.lenient = True
    rows = [_row("9810.79000000000126", score=0.0)]

    regraded, changes = regrade_rows(rows, datasets={"toy": dataset})

    assert regraded[0].score.value == 1.0
    assert regraded[0].prediction.answer == "9810.79000000000126"
    assert changes == [
        {
            "dataset": "toy",
            "example_id": "problem-1",
            "runner": "official-rlm",
            "seed": 0,
            "answer": "9810.79000000000126",
            "was": 0.0,
            "now": 1.0,
        }
    ]


def test_regrade_leaves_unchanged_scores_and_failed_attempts_alone():
    dataset = _StrictThenLenientDataset()
    dataset.lenient = True
    rows = [
        _row("9810.79", score=1.0),
        _row("", score=0.0, error="APIConnectionError: Connection error."),
    ]

    regraded, changes = regrade_rows(rows, datasets={"toy": dataset})

    assert changes == []
    # An attempt that never produced an answer has nothing to regrade.
    assert regraded[1].score.value == 0.0
    assert regraded[1].prediction.error == "APIConnectionError: Connection error."


def test_regrade_treats_a_raising_scorer_as_zero_like_the_harness_does():
    rows = [_row("not a grid", score=0.0)]

    regraded, changes = regrade_rows(rows, datasets={"toy": _RaisingDataset()})

    assert changes == []
    assert regraded[0].score.value == 0.0
    assert "expected 36 grid cells" in regraded[0].score.details["error"]


def test_regraded_rows_round_trip_through_jsonl(tmp_path):
    dataset = _StrictThenLenientDataset()
    dataset.lenient = True
    regraded, _ = regrade_rows([_row("9810.7912", score=0.0)], datasets={"toy": dataset})

    path = tmp_path / "rows-regraded.jsonl"
    write_rows(path, regraded)

    assert load_rows(path) == regraded
