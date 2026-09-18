"""Frozen visual Sudoku tasks for branch-and-select evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from benchmarks.eval import dataset
from benchmarks.eval.environments.base import EnvironmentEpisode
from benchmarks.eval.environments.sudoku_search import (
    SudokuSearchAdapter,
    SudokuSearchConfig,
    SudokuSearchPuzzle,
    render_grid,
)
from benchmarks.eval.tasks.environment import grade_environment
from benchmarks.eval.types import Dataset, Example, Prediction, Score

DEFAULT_ACTION_BUDGET = 192
DEFAULT_CANDIDATE_BUDGET = 8
MANIFEST_REVISION = "visual-sudoku-search-v2"

DEVELOPMENT_6X6_PUZZLES: tuple[SudokuSearchPuzzle, ...] = (
    SudokuSearchPuzzle(
        id="visual_6x6_branch_01",
        clues=("...5.2", "......", "4.231.", ".16...", ".6..53", "5.4..1"),
        solution=("143562", "625134", "452316", "316245", "261453", "534621"),
        box_rows=2,
        box_columns=3,
    ),
    SudokuSearchPuzzle(
        id="visual_6x6_branch_02",
        clues=("1...56", ".....3", "234.6.", "5...3.", "...612", ".....5"),
        solution=("123456", "456123", "234561", "561234", "345612", "612345"),
        box_rows=2,
        box_columns=3,
    ),
)

FROZEN_9X9_AND_CONTROL_PUZZLES: tuple[SudokuSearchPuzzle, ...] = (
    SudokuSearchPuzzle(
        id="visual_9x9_branch_01",
        clues=(
            ".34....1.",
            ".......48",
            ".9.3.....",
            "8.976...3",
            "42..5.7..",
            "..392....",
            "..1....8.",
            "...4.9.35",
            ".4..86...",
        ),
        solution=(
            "534678912",
            "672195348",
            "198342567",
            "859761423",
            "426853791",
            "713924856",
            "961537284",
            "287419635",
            "345286179",
        ),
        box_rows=3,
        box_columns=3,
    ),
    SudokuSearchPuzzle(
        id="visual_9x9_branch_02",
        clues=(
            "....536..",
            "9....21..",
            ".75.91..3",
            "....3.896",
            ".6...5...",
            "2......34",
            "5.19.4..8",
            "4........",
            "..6.....2",
        ),
        solution=(
            "812753649",
            "943682175",
            "675491283",
            "154237896",
            "369845721",
            "287169534",
            "521974368",
            "438526917",
            "796318452",
        ),
        box_rows=3,
        box_columns=3,
    ),
    SudokuSearchPuzzle(
        id="visual_6x6_branch_03",
        clues=("..3...", "4..1..", ".3..6.", "..1..4", "3..6..", ".1..4."),
        solution=("123456", "456123", "234561", "561234", "345612", "612345"),
        box_rows=2,
        box_columns=3,
    ),
)
REJECTED_DEVELOPMENT_PUZZLES = (DEVELOPMENT_6X6_PUZZLES[0],)
PILOT_PUZZLES = (
    FROZEN_9X9_AND_CONTROL_PUZZLES[0],
    DEVELOPMENT_6X6_PUZZLES[1],
)
HELDOUT_PUZZLES = FROZEN_9X9_AND_CONTROL_PUZZLES[1:]
FULL_PUZZLES = HELDOUT_PUZZLES
ALL_PUZZLES = (*REJECTED_DEVELOPMENT_PUZZLES, *PILOT_PUZZLES, *HELDOUT_PUZZLES)
RESTRAINT_PUZZLE_IDS = frozenset({"visual_6x6_branch_02", "visual_6x6_branch_03"})

TASK_PROMPT = """\
Solve this frozen visual Sudoku through the host-owned environment. Coordinates
are zero-based. Use observe(), place(row, column, value), erase(row, column),
place_sequence(placements), and check(). A check reports only visible constraint
violations and fill progress; it never compares your cells with the hidden
solution.

Explore genuinely different solution approaches or independent checks when
useful. Register each solution state you want considered with submit_candidate(),
compare its redacted summary with list_candidates(), then explicitly call
select_candidate(candidate_id). Only that explicit selection is graded; the
evaluator does not silently choose the best alternative, and all mutations share
one action budget. Execute both calls before finish(); writing a candidate ID in
the final answer does not count. If deterministic deductions make alternatives
unnecessary, submit the current grid and select it early enough to complete the
protocol.
"""


@dataset(
    "sudoku_search",
    tags=["conformance", "environment", "sudoku"],
)
class SudokuSearchDataset(Dataset):
    """Deterministic visual Sudoku manifests with private candidate branches."""

    def __init__(
        self,
        manifest: str | Sequence[str] | None = None,
        *,
        action_budget: int = DEFAULT_ACTION_BUDGET,
        candidate_budget: int = DEFAULT_CANDIDATE_BUDGET,
    ) -> None:
        self.puzzles = _select_puzzles(manifest) if manifest is not None else None
        self.manifest = manifest if isinstance(manifest, str) else "split"
        self.action_budget = _positive_integer(action_budget, "action_budget")
        self.candidate_budget = _positive_integer(candidate_budget, "candidate_budget")

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del seed
        if split not in {"test", "public", "pilot", "full"}:
            raise ValueError("Sudoku search exposes only pilot and full suite splits")
        if limit is not None and limit < 1:
            return []
        puzzles = (
            self.puzzles
            if self.puzzles is not None
            else FULL_PUZZLES
            if split == "full"
            else PILOT_PUZZLES
        )
        selected = puzzles if limit is None else puzzles[:limit]
        return [self._example(puzzle) for puzzle in selected]

    def score(self, example: Example, prediction: Prediction) -> Score:
        grade = grade_environment(
            prediction,
            native_keys=("score", "placement_score"),
        )
        native_exact = bool(grade.view.primary.get("exact", False))
        exact = native_exact and grade.selected
        return Score(
            value=grade.value,
            correct=exact,
            details=grade.details(
                error=prediction.error,
                puzzle_id=example.metadata.get("source_id"),
                native_scorer="sudoku_exact_and_placement",
                placement_score=grade.view.native_float("placement_score", bounded=True),
                native_exact=native_exact,
                exact=exact,
            ),
        )

    def make_episode(
        self,
        example: Example,
        *,
        artifact_dir: str | Path | None = None,
        role: str = "primary",
    ) -> EnvironmentEpisode:
        """Open the runner-neutral environment represented by ``example``."""

        del role
        environment = example.metadata.get("environment", {})
        if not isinstance(environment, dict) or environment.get("kind") != "sudoku_search":
            raise ValueError("example is not a Sudoku search environment task")
        puzzle = _puzzle_by_id(str(environment.get("puzzle_id", "")))
        action_budget = _positive_integer(environment.get("action_budget"), "action_budget")
        candidate_budget = _positive_integer(
            environment.get("candidate_budget"),
            "candidate_budget",
        )
        return EnvironmentEpisode(
            SudokuSearchAdapter(
                SudokuSearchConfig(
                    puzzle=puzzle,
                    artifact_dir=Path(artifact_dir) if artifact_dir is not None else None,
                )
            ),
            max_actions=action_budget,
            episode_id=example.id,
            candidate_budget=candidate_budget,
        )

    create_episode = make_episode
    episode = make_episode

    def _example(self, puzzle: SudokuSearchPuzzle) -> Example:
        visual = render_grid(
            puzzle.initial_grid,
            box_rows=puzzle.box_rows,
            box_columns=puzzle.box_columns,
        )
        prompt = (
            f"{TASK_PROMPT}\nPuzzle: {puzzle.id}\n"
            f"Rules: fill 1..{puzzle.size}; each row, column, and "
            f"{puzzle.box_rows}x{puzzle.box_columns} box contains every value once.\n\n"
            f"{visual}"
        )
        environment = {
            "kind": "sudoku_search",
            "adapter": (
                "benchmarks.eval.environments.sudoku_search:SudokuSearchEnvironmentAdapter"
            ),
            "puzzle_id": puzzle.id,
            "size": puzzle.size,
            "box_rows": puzzle.box_rows,
            "box_columns": puzzle.box_columns,
            "operations": ["observe", "place", "erase", "place_sequence", "check"],
            "action_budget": self.action_budget,
            "max_actions": self.action_budget,
            "candidate_budget": self.candidate_budget,
            "shared_budget": "root_and_probes",
            "fork": "fresh_puzzle_replay_immediate_parent_prefix",
            "records": ["action", "grid_evolution", "candidate_selection"],
        }
        return Example(
            id=f"sudoku_search_{puzzle.id}",
            prompt=prompt,
            expected={"puzzle_id": puzzle.id},
            context={
                "puzzle_id": puzzle.id,
                "coordinate_system": "zero_based",
            },
            metadata={
                "source": "rlmkit/frozen-visual-sudoku-search",
                "source_id": puzzle.id,
                "revision": MANIFEST_REVISION,
                "license": "MIT",
                "domain": "sudoku_search",
                "difficulty": puzzle.difficulty,
                "route_hypothesis": "branch_and_select",
                "manifest_split": ("development" if puzzle in PILOT_PUZZLES else "heldout"),
                "restraint_control": puzzle.id in RESTRAINT_PUZZLE_IDS,
                "native_scorer": "sudoku_exact_and_placement",
                "action_budget": self.action_budget,
                "candidate_budget": self.candidate_budget,
                "environment": environment,
            },
        )


def _select_puzzles(manifest: str | Sequence[str]) -> tuple[SudokuSearchPuzzle, ...]:
    if isinstance(manifest, str):
        label = manifest.strip().lower()
        if label == "pilot":
            return PILOT_PUZZLES
        if label in {"full", "all"}:
            return FULL_PUZZLES
        ids = tuple(part.strip() for part in manifest.split(",") if part.strip())
    else:
        ids = tuple(str(value).strip() for value in manifest if str(value).strip())
    if not ids:
        raise ValueError("select at least one frozen Sudoku puzzle")
    if len(set(ids)) != len(ids):
        raise ValueError("Sudoku puzzle selection contains duplicates")
    known = {puzzle.id: puzzle for puzzle in ALL_PUZZLES}
    unknown = [puzzle_id for puzzle_id in ids if puzzle_id not in known]
    if unknown:
        raise ValueError(f"unknown frozen Sudoku puzzle(s): {', '.join(unknown)}")
    return tuple(known[puzzle_id] for puzzle_id in ids)


def _puzzle_by_id(puzzle_id: str) -> SudokuSearchPuzzle:
    try:
        return next(puzzle for puzzle in ALL_PUZZLES if puzzle.id == puzzle_id)
    except StopIteration as exc:
        raise ValueError(f"unknown frozen Sudoku puzzle {puzzle_id!r}") from exc


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


FULL_TASKS = FULL_PUZZLES
PILOT_TASKS = PILOT_PUZZLES
SudokuSearchTaskGraphDataset = SudokuSearchDataset

__all__ = [
    "DEFAULT_ACTION_BUDGET",
    "DEFAULT_CANDIDATE_BUDGET",
    "ALL_PUZZLES",
    "FULL_PUZZLES",
    "HELDOUT_PUZZLES",
    "FULL_TASKS",
    "MANIFEST_REVISION",
    "PILOT_PUZZLES",
    "RESTRAINT_PUZZLE_IDS",
    "PILOT_TASKS",
    "TASK_PROMPT",
    "SudokuSearchDataset",
    "SudokuSearchTaskGraphDataset",
]
