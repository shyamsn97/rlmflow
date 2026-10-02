"""Frozen interactive environment catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.environment import Environment
from benchmarks.eval.environments.mazebench import (
    MazeBenchConfig,
    MazeBenchEnvironment,
)
from benchmarks.eval.environments.minimaze import (
    MiniMazeConfig,
    MiniMazeEnvironment,
    MiniMazeSpec,
)
from benchmarks.eval.environments.sudoku_search import (
    SudokuEnvironment,
    SudokuSearchConfig,
)
from benchmarks.eval.tasks.mazebench import MazeBenchDataset
from benchmarks.eval.tasks.minimaze import FROZEN_MINIMAZE_MANIFEST, MiniMazeDataset
from benchmarks.eval.tasks.sudoku_search import ALL_PUZZLES, SudokuSearchDataset

ENVIRONMENT_TYPES = {
    "minimaze": MiniMazeEnvironment,
    "sudoku_search": SudokuEnvironment,
    "mazebench": MazeBenchEnvironment,
}


def environment_class(kind: str) -> type[Environment]:
    try:
        return ENVIRONMENT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark environment kind {kind!r}") from exc


def interactive_environments(
    *,
    split: str,
    seed: int,
    artifact_root: str | Path = "benchmarks/runs/environment-work",
) -> list[Environment]:
    del seed
    root = Path(artifact_root)
    environments: list[Environment] = []
    minimaze_examples = MiniMazeDataset().examples(split=split, limit=None, seed=0)
    for example in minimaze_examples:
        task_id = str(example.metadata["source_id"])
        environments.append(
            environment_from_id(
                kind="minimaze",
                environment_id=task_id,
                artifact_dir=root / example.id,
                description=example.prompt,
                configuration=example.metadata["environment"],
            )
        )
    sudoku_examples = SudokuSearchDataset().examples(split=split, limit=None, seed=0)
    for example in sudoku_examples:
        puzzle_id = str(example.metadata["source_id"])
        environments.append(
            environment_from_id(
                kind="sudoku_search",
                environment_id=puzzle_id,
                artifact_dir=root / example.id,
                description=example.prompt,
                configuration=example.metadata["environment"],
            )
        )
    maze_example = MazeBenchDataset(preflight=False).examples(
        split=split,
        limit=None,
        seed=0,
    )[0]
    environments.append(
        environment_from_id(
            kind="mazebench",
            environment_id="mazebench_hxi",
            artifact_dir=root / maze_example.id,
            description=maze_example.prompt,
            configuration=maze_example.metadata["environment"],
        )
    )
    return environments


def environment_from_id(
    *,
    kind: str,
    environment_id: str,
    artifact_dir: Path,
    configuration: Mapping[str, Any] | None = None,
    description: str | None = None,
) -> Environment:
    values = dict(configuration or {})
    if kind == "minimaze":
        source_id = environment_id.removeprefix("minimaze_")
        base = next(
            (spec for spec in FROZEN_MINIMAZE_MANIFEST if spec.id == source_id),
            None,
        )
        if base is None:
            raise ValueError(f"unknown frozen MiniMaze environment {environment_id!r}")
        spec = MiniMazeSpec(
            id=base.id,
            rows=base.rows,
            fog_radius=int(values.get("fog_radius", base.fog_radius)),
            action_budget=int(values.get("max_actions", base.action_budget)),
            candidate_budget=int(values.get("candidate_budget", base.candidate_budget)),
        )
        return MiniMazeEnvironment(
            MiniMazeConfig(spec=spec, artifact_dir=artifact_dir),
            description=description,
            environment_id=f"minimaze_{source_id}",
        )
    if kind == "sudoku_search":
        source_id = environment_id.removeprefix("sudoku_search_")
        puzzle = next((item for item in ALL_PUZZLES if item.id == source_id), None)
        if puzzle is None:
            raise ValueError(f"unknown frozen Sudoku environment {environment_id!r}")
        return SudokuEnvironment(
            SudokuSearchConfig(puzzle=puzzle, artifact_dir=artifact_dir),
            description=description,
            environment_id=f"sudoku_search_{source_id}",
            action_budget=int(values.get("max_actions", values.get("action_budget", 192))),
            candidate_budget=int(values.get("candidate_budget", 8)),
        )
    if kind == "mazebench":
        return MazeBenchEnvironment(
            MazeBenchConfig(
                artifact_dir=artifact_dir,
                game_id=str(values.get("game_id", "maze")),
                level=str(values.get("level", "HxI")),
                view=str(values.get("view", "top-diagonal")),
                yaw=int(values.get("yaw", 0)),
            ),
            description=description,
            environment_id="mazebench_hxi",
            action_budget=int(values.get("max_actions", 256)),
        )
    raise ValueError(f"unsupported benchmark environment kind {kind!r}")


__all__ = [
    "environment_class",
    "environment_from_id",
    "interactive_environments",
]
