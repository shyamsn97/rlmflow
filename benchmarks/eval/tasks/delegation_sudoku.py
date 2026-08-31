"""One 6x6 Sudoku-Bench coupled-control problem."""

from __future__ import annotations

import re

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import parse_json_answer
from benchmarks.eval.types import Dataset, Example, Prediction, Score

SOURCE_ID = "cross-product"
INITIAL_BOARD = "." * 36
SOLUTION = "143562625134452316316245261453534621"
RULES = """Normal 6x6 Sudoku rules apply.
Every indicated diagonal has the same product."""
VISUAL_ELEMENTS = """- diagonal arrow at r0c4, pointing lower right
- diagonal arrow at r2c0, pointing lower right
- diagonal arrow at r4c0, pointing upper right
- diagonal arrow at r5c1, pointing upper right"""


@dataset("delegation_sudoku", tags=["delegation", "task-graph", "control"])
class DelegationSudokuDataset(Dataset):
    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        return [] if limit == 0 else [self._example()]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = _grid(str(example.expected), size=6)
        actual = _answer_grid(prediction.answer, size=6)
        placements = sum(
            value == expected[row][column]
            for row, values in enumerate(actual)
            for column, value in enumerate(values)
        )
        fraction = placements / 36
        exact = actual == expected
        return Score(
            value=fraction,
            correct=exact,
            details={"exact": exact, "correct_placement_fraction": fraction},
        )

    def _example(self) -> Example:
        board = _grid(INITIAL_BOARD, size=6, blanks=True)
        rendered = "\n".join(
            " ".join("." if value == 0 else str(value) for value in line) for line in board
        )
        return Example(
            id=f"delegation_sudoku_18_{SOURCE_ID}",
            prompt=(
                f"{RULES}\n\nVisual elements:\n{VISUAL_ELEMENTS}\n\n"
                "Solve the 6x6 grid below. Return only a JSON array of six rows.\n\n"
                f"{rendered}"
            ),
            expected=SOLUTION,
            metadata={
                "source_id": SOURCE_ID,
                "problem": 18,
                "native_scorer": "sudoku_exact_and_placement",
                "adaptation": "text-only frozen challenge_100 prompt",
            },
        )


def _grid(value: str, *, size: int, blanks: bool = False) -> list[list[int]]:
    allowed = r"[^0-9.]" if blanks else r"[^0-9]"
    cells = re.sub(allowed, "", value)
    if len(cells) != size * size:
        raise ValueError(f"expected {size * size} grid cells, got {len(cells)}")
    values = [0 if cell == "." else int(cell) for cell in cells]
    return [values[index : index + size] for index in range(0, len(values), size)]


def _answer_grid(answer: str, *, size: int) -> list[list[int]]:
    try:
        parsed = parse_json_answer(answer)
        if (
            isinstance(parsed, list)
            and len(parsed) == size
            and all(isinstance(row, list) and len(row) == size for row in parsed)
        ):
            return [[int(value) for value in row] for row in parsed]
    except (TypeError, ValueError):
        pass
    return _grid(answer, size=size)


__all__ = ["DelegationSudokuDataset"]
