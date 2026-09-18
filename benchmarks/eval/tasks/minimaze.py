"""Frozen MiniMaze manifests for visual branch-and-select evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from benchmarks.eval import dataset
from benchmarks.eval.environments.base import EnvironmentEpisode
from benchmarks.eval.environments.minimaze import (
    SCORE_WEIGHTS,
    MiniMazeAdapter,
    MiniMazeConfig,
    MiniMazeSpec,
)
from benchmarks.eval.tasks.environment import grade_environment
from benchmarks.eval.types import Dataset, Example, Prediction, Score

TASK_PROMPT = """\
Navigate the deterministic fog-of-war ASCII MiniMaze and collect every gem (*).
Use observe(), action("up"|"down"|"left"|"right"), and action_sequence([...]).
Lowercase letters are reusable keys for matching uppercase doors. Build a map
from observations instead of assuming unknown (?) cells are walls.

Explore and compare genuinely different route hypotheses before committing when
useful. Register each trajectory you want considered with submit_candidate(),
compare its public gem completion, coverage, and path-efficiency metrics using
list_candidates(), then explicitly call select_candidate() on the best evaluated
candidate. Only the explicitly selected candidate is scored, and all exploration
shares one action budget. Execute submit_candidate and select_candidate as tool
calls before finish(); writing a candidate ID in the final answer is not
selection. If no alternative is useful, submit the current trajectory and select
it early enough to complete the protocol.
"""


FROZEN_MINIMAZE_MANIFEST = (
    MiniMazeSpec(
        id="crossroads",
        rows=(
            "###########",
            "#S..#...*.#",
            "#.#.#.###.#",
            "#.#a..A...#",
            "#.###.#.#.#",
            "#*....#...#",
            "###########",
        ),
        action_budget=44,
        candidate_budget=6,
    ),
    MiniMazeSpec(
        id="two_keys",
        rows=(
            "#############",
            "#S....#....*#",
            "#.###.#.###.#",
            "#...a.A.....#",
            "###..#.####.#",
            "#*..#...B.b.#",
            "#############",
        ),
        action_budget=54,
        candidate_budget=7,
    ),
    MiniMazeSpec(
        id="upper_loop",
        rows=(
            "###########",
            "#S..aA..*.#",
            "#.#####.#.#",
            "#.....#.#.#",
            "###.#.#.#.#",
            "#*..#.....#",
            "###########",
        ),
        action_budget=48,
        candidate_budget=6,
    ),
    MiniMazeSpec(
        id="lower_loop",
        rows=(
            "###########",
            "#S........#",
            "#.#######.#",
            "#.#*...A#.#",
            "#.#.###.#.#",
            "#*..a.....#",
            "###########",
        ),
        action_budget=52,
        candidate_budget=7,
    ),
    MiniMazeSpec(
        id="double_gate",
        rows=(
            "#############",
            "#S..a.A....*#",
            "#.###.#####.#",
            "#...#.....#.#",
            "###.#.###.#.#",
            "#*..b.B.....#",
            "#############",
        ),
        action_budget=58,
        candidate_budget=8,
    ),
    MiniMazeSpec(
        id="key_detour",
        rows=(
            "#############",
            "#S....#....*#",
            "#####.#.###.#",
            "#a..A...#...#",
            "#.#######.#.#",
            "#*..........#",
            "#############",
        ),
        action_budget=60,
        candidate_budget=8,
    ),
    MiniMazeSpec(
        id="three_gems",
        rows=(
            "#############",
            "#S..*.......#",
            "#.#####.###.#",
            "#...a.A...#*#",
            "###.#.###.#.#",
            "#*..#.......#",
            "#############",
        ),
        action_budget=62,
        candidate_budget=8,
    ),
    MiniMazeSpec(
        id="split_vault",
        rows=(
            "#############",
            "#S....#....*#",
            "#.##a.#.###.#",
            "#...#.A.....#",
            "###.#.#####.#",
            "#*..#.......#",
            "#############",
        ),
        action_budget=58,
        candidate_budget=8,
    ),
)

PILOT_TASK_IDS = ("crossroads", "two_keys")
FULL_TASK_IDS = tuple(spec.id for spec in FROZEN_MINIMAZE_MANIFEST if spec.id not in PILOT_TASK_IDS)
RESTRAINT_TASK_IDS = frozenset({"two_keys", "key_detour"})
PILOT_IDS = PILOT_TASK_IDS
FULL_MANIFEST = FROZEN_MINIMAZE_MANIFEST
_SPECS = {spec.id: spec for spec in FROZEN_MINIMAZE_MANIFEST}


@dataset(
    "minimaze",
    tags=["conformance", "environment", "visual"],
)
class MiniMazeDataset(Dataset):
    """Offline MiniMaze corpus selected by the suite split."""

    def __init__(
        self,
        tasks: str | Sequence[str] | None = None,
        *,
        action_budget: int | None = None,
        candidate_budget: int | None = None,
    ) -> None:
        self.task_ids = _select_tasks(tasks) if tasks is not None else None
        self.action_budget = _optional_positive_int("action_budget", action_budget)
        self.candidate_budget = _optional_positive_int("candidate_budget", candidate_budget)

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del seed
        if split not in {"test", "pilot", "full"}:
            raise ValueError("MiniMaze exposes only frozen test, pilot, and full splits")
        task_ids = (
            self.task_ids
            if self.task_ids is not None
            else FULL_TASK_IDS
            if split == "full"
            else PILOT_TASK_IDS
        )
        selected = task_ids if limit is None else task_ids[: max(0, limit)]
        return [self._example(_SPECS[task_id]) for task_id in selected]

    def score(self, example: Example, prediction: Prediction) -> Score:
        del example
        grade = grade_environment(prediction, native_keys=("score",))
        primary = grade.view.primary
        completed = bool(primary.get("completed")) and grade.selected
        return Score(
            value=grade.value,
            correct=completed,
            details=grade.details(
                error=prediction.error,
                completed=completed,
                gems_collected=_nonnegative_int(primary.get("gems_collected")),
                total_gems=_nonnegative_int(primary.get("total_gems")),
                gem_completion=grade.view.native_float("gem_completion", bounded=True),
                coverage=grade.view.native_float("coverage", bounded=True),
                path_efficiency=grade.view.native_float("path_efficiency", bounded=True),
            ),
        )

    def make_episode(
        self,
        example: Example,
        *,
        artifact_dir: str | Path | None = None,
        role: str = "primary",
    ) -> EnvironmentEpisode:
        """Create the shared candidate-aware episode represented by ``example``."""

        del role
        environment = example.metadata.get("environment", {})
        if not isinstance(environment, dict) or environment.get("kind") != "minimaze":
            raise ValueError("example is not a MiniMaze environment task")
        task_id = str(environment.get("task_id", ""))
        try:
            base_spec = _SPECS[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown frozen MiniMaze task {task_id!r}") from exc
        spec = MiniMazeSpec(
            id=base_spec.id,
            rows=base_spec.rows,
            fog_radius=int(environment["fog_radius"]),
            action_budget=int(environment["action_budget"]),
            candidate_budget=int(environment["candidate_budget"]),
        )
        return EnvironmentEpisode(
            MiniMazeAdapter(MiniMazeConfig(spec=spec, artifact_dir=artifact_dir)),
            max_actions=spec.action_budget,
            episode_id=example.id,
            candidate_budget=spec.candidate_budget,
        )

    create_episode = make_episode
    episode = make_episode

    def _example(self, base_spec: MiniMazeSpec) -> Example:
        action_budget = self.action_budget or base_spec.action_budget
        candidate_budget = self.candidate_budget or base_spec.candidate_budget
        return Example(
            id=f"minimaze_{base_spec.id}",
            prompt=TASK_PROMPT,
            expected={"completed": True, "total_gems": "".join(base_spec.rows).count("*")},
            context={
                "task_id": base_spec.id,
                "protocol": "observe/action/action_sequence",
                "score_weights": SCORE_WEIGHTS,
            },
            metadata={
                "source": "rlmflow:minimaze",
                "source_id": base_spec.id,
                "revision": 1,
                "license": "MIT",
                "domain": "minimaze",
                "split": "frozen",
                "route_hypothesis": "branch-and-select",
                "manifest_split": ("development" if base_spec.id in PILOT_TASK_IDS else "heldout"),
                "restraint_control": base_spec.id in RESTRAINT_TASK_IDS,
                "native_scorer": "minimaze_partial_progress",
                "action_budget": action_budget,
                "candidate_budget": candidate_budget,
                "environment": {
                    "kind": "minimaze",
                    "adapter": ("benchmarks.eval.environments.minimaze:MiniMazeEnvironmentAdapter"),
                    "task_id": base_spec.id,
                    "fog_radius": base_spec.fog_radius,
                    "operations": ["observe", "action", "action_sequence"],
                    "action_budget": action_budget,
                    "max_actions": action_budget,
                    "candidate_budget": candidate_budget,
                    "shared_budget": "root_and_probes",
                    "fork": "fresh_replay_immediate_parent_prefix",
                    "observation_mode": "fogged_ascii",
                    "score_weights": dict(SCORE_WEIGHTS),
                    "network": "disabled",
                },
            },
        )


def _select_tasks(tasks: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(tasks, str):
        label = tasks.strip().lower()
        if label == "pilot":
            return PILOT_TASK_IDS
        if label in {"all", "full"}:
            return FULL_TASK_IDS
        selected = tuple(part.strip() for part in tasks.split(",") if part.strip())
    else:
        selected = tuple(tasks)
    if not selected:
        raise ValueError("select at least one MiniMaze task")
    unknown = [task_id for task_id in selected if task_id not in _SPECS]
    if unknown:
        raise ValueError(f"unknown frozen MiniMaze task(s): {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("MiniMaze task selection contains duplicates")
    return selected


def _optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "FROZEN_MINIMAZE_MANIFEST",
    "FULL_MANIFEST",
    "FULL_TASK_IDS",
    "PILOT_IDS",
    "PILOT_TASK_IDS",
    "RESTRAINT_TASK_IDS",
    "TASK_PROMPT",
    "MiniMazeDataset",
]
