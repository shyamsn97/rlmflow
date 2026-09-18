"""Deterministic Sudoku environment for visual branch-and-select benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.base import EnvironmentAdapter, EnvironmentResult

MAX_SEQUENCE_LENGTH = 256


@dataclass(frozen=True, slots=True)
class SudokuSearchPuzzle:
    """One frozen Sudoku puzzle and its evaluator-private solution."""

    id: str
    clues: tuple[str, ...]
    solution: tuple[str, ...]
    box_rows: int
    box_columns: int
    difficulty: str = "medium"

    def __post_init__(self) -> None:
        size = len(self.clues)
        if not self.id.strip():
            raise ValueError("puzzle id must be non-empty")
        if size < 2 or len(self.solution) != size:
            raise ValueError("clues and solution must be equally sized square grids")
        if self.box_rows * self.box_columns != size:
            raise ValueError("box dimensions must multiply to the grid size")
        if any(len(row) != size for row in (*self.clues, *self.solution)):
            raise ValueError("every Sudoku row must match the grid size")
        if not any(cell != "." for row in self.clues for cell in row):
            raise ValueError("a frozen Sudoku task must contain at least one clue")
        if not any(cell == "." for row in self.clues for cell in row):
            raise ValueError("a frozen Sudoku task must contain at least one editable cell")

        expected = {str(value) for value in range(1, size + 1)}
        for row in self.clues:
            if any(cell != "." and cell not in expected for cell in row):
                raise ValueError("clues contain a value outside the Sudoku alphabet")
        if any(set(row) != expected for row in self.solution):
            raise ValueError("each solution row must contain every value exactly once")
        solution_grid = self.solution_grid
        if _constraint_violations(solution_grid, self.box_rows, self.box_columns):
            raise ValueError("the supplied solution violates Sudoku constraints")
        for row in range(size):
            for column in range(size):
                clue = self.clues[row][column]
                if clue != "." and int(clue) != solution_grid[row][column]:
                    raise ValueError("a clue does not match the supplied solution")

    @property
    def size(self) -> int:
        return len(self.clues)

    @property
    def initial_grid(self) -> list[list[int]]:
        return [[0 if cell == "." else int(cell) for cell in row] for row in self.clues]

    @property
    def solution_grid(self) -> list[list[int]]:
        return [[int(cell) for cell in row] for row in self.solution]


@dataclass(frozen=True, slots=True)
class SudokuSearchConfig:
    puzzle: SudokuSearchPuzzle
    artifact_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.artifact_dir is not None:
            object.__setattr__(
                self,
                "artifact_dir",
                Path(self.artifact_dir).expanduser().resolve(),
            )


@dataclass(slots=True)
class _SudokuInstance:
    grid: list[list[int]]
    commands: list[str] = field(default_factory=list)
    evolution: list[dict[str, Any]] = field(default_factory=list)
    replay_actions: int = 0


class SudokuSearchAdapter(EnvironmentAdapter):
    """In-memory private-branch Sudoku implementation."""

    kind = "sudoku_search"

    def __init__(
        self,
        config: SudokuSearchConfig | SudokuSearchPuzzle,
        *,
        artifact_dir: str | Path | None = None,
    ) -> None:
        if isinstance(config, SudokuSearchPuzzle):
            config = SudokuSearchConfig(
                config,
                Path(artifact_dir) if artifact_dir is not None else None,
            )
        elif artifact_dir is not None:
            raise ValueError("artifact_dir belongs in SudokuSearchConfig when config is supplied")
        self.config = config
        self.puzzle = config.puzzle
        self._instances: dict[str, _SudokuInstance] = {}

    def start(self, instance_id: str) -> EnvironmentResult:
        if instance_id in self._instances:
            raise ValueError(f"Sudoku instance {instance_id!r} already exists")
        instance = _SudokuInstance(grid=self.puzzle.initial_grid)
        instance.evolution.append(
            {"index": 0, "operation": "start", "replay": False, "grid": _copy_grid(instance.grid)}
        )
        self._instances[instance_id] = instance
        observation = self._observation(instance)
        return EnvironmentResult(observation, observation_summary=self._summary(instance))

    def fork(
        self,
        instance_id: str,
        *,
        parent_instance_id: str,
        command_prefix: tuple[str, ...],
    ) -> EnvironmentResult:
        self._instance(parent_instance_id)
        self.start(instance_id)
        child = self._instance(instance_id)
        for encoded in command_prefix:
            operation, payload = _decode_command(encoded)
            self._apply(child, operation, payload, replay=True)
        observation = self._observation(child)
        return EnvironmentResult(observation, observation_summary=self._summary(child))

    def execute(
        self,
        instance_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        max_actions: int,
    ) -> EnvironmentResult:
        instance = self._instance(instance_id)
        if operation == "observe":
            observation = self._observation(instance)
            return EnvironmentResult(observation, observation_summary=self._summary(instance))
        if operation == "check":
            report = self._check(instance)
            return EnvironmentResult(report, observation_summary=self._summary(instance))
        if operation in {"place", "erase"}:
            if max_actions < 1:
                raise RuntimeError("the shared environment action budget is exhausted")
            normalized = self._normalize_action(operation, payload)
            command = self._apply(instance, operation, normalized, replay=False)
            return EnvironmentResult(
                {
                    "accepted": True,
                    "action": normalized,
                    "observation": self._observation(instance),
                    "actions_remaining": max_actions - 1,
                },
                accepted_actions=(command,),
                observation_summary=self._summary(instance),
            )
        if operation == "place_sequence":
            requested = payload.get("placements")
            if not isinstance(requested, list) or not requested:
                raise ValueError("placements must contain at least one item")
            if len(requested) > MAX_SEQUENCE_LENGTH:
                raise ValueError(f"placements must contain at most {MAX_SEQUENCE_LENGTH} items")
            normalized = [
                self._normalize_action("place", item, index=index)
                for index, item in enumerate(requested)
            ]
            selected = normalized[:max_actions]
            if not selected:
                raise RuntimeError("the shared environment action budget is exhausted")
            commands = tuple(
                self._apply(instance, "place", item, replay=False) for item in selected
            )
            return EnvironmentResult(
                {
                    "requested_count": len(normalized),
                    "completed_count": len(commands),
                    "stopped_early": len(commands) < len(normalized),
                    "stop_reason": (
                        "shared_budget_exhausted"
                        if len(commands) < len(normalized)
                        else "completed"
                    ),
                    "observation": self._observation(instance),
                    "actions_remaining": max(0, max_actions - len(commands)),
                },
                accepted_actions=commands,
                observation_summary=self._summary(instance),
            )
        raise ValueError(f"unknown Sudoku operation {operation!r}")

    def primary_metrics(self, instance_id: str) -> dict[str, Any]:
        instance = self._instance(instance_id)
        solution = self.puzzle.solution_grid
        editable = [
            (row, column)
            for row in range(self.puzzle.size)
            for column in range(self.puzzle.size)
            if self.puzzle.clues[row][column] == "."
        ]
        placements = sum(
            instance.grid[row][column] == solution[row][column] for row, column in editable
        )
        placement_score = placements / len(editable)
        violations = _constraint_violations(
            instance.grid,
            self.puzzle.box_rows,
            self.puzzle.box_columns,
        )
        exact = instance.grid == solution
        score = 0.6 * placement_score + 0.4 * float(exact)
        return {
            "puzzle_id": self.puzzle.id,
            "score": score,
            "placement_score": placement_score,
            "exact": exact,
            "filled_count": _filled_count(instance.grid),
            "total_cells": self.puzzle.size**2,
            "violation_count": len(violations),
            "accepted_actions": len(instance.commands) - instance.replay_actions,
            "replay_actions": instance.replay_actions,
        }

    def candidate_metrics(self, instance_id: str) -> tuple[float, dict[str, Any]]:
        instance = self._instance(instance_id)
        hidden_score = float(self.primary_metrics(instance_id)["score"])
        # Deliberately omit exactness, placement score, and correct-cell counts.
        return hidden_score, {
            "puzzle_id": self.puzzle.id,
            **self._summary(instance),
        }

    def artifacts(self, instance_id: str) -> dict[str, str]:
        if self.config.artifact_dir is None:
            return {}
        instance = self._instance(instance_id)
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.artifact_dir / "sudoku_replay.json"
        replay = {
            "schema_version": 1,
            "kind": self.kind,
            "puzzle_id": self.puzzle.id,
            "size": self.puzzle.size,
            "box_rows": self.puzzle.box_rows,
            "box_columns": self.puzzle.box_columns,
            "clues": list(self.puzzle.clues),
            "initial_grid": self.puzzle.initial_grid,
            "grid_evolution": instance.evolution,
            "final_grid": _copy_grid(instance.grid),
            "selected_instance_id": instance_id,
            "instances": {
                current_id: {
                    "selected": current_id == instance_id,
                    "grid_evolution": current.evolution,
                    "final_grid": _copy_grid(current.grid),
                    "metrics": self.primary_metrics(current_id),
                }
                for current_id, current in self._instances.items()
            },
        }
        path.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"sudoku_replay": str(path)}

    def _instance(self, instance_id: str) -> _SudokuInstance:
        try:
            return self._instances[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown Sudoku instance {instance_id!r}") from exc

    def _normalize_action(
        self,
        operation: str,
        payload: object,
        *,
        index: int | None = None,
    ) -> dict[str, int]:
        label = "action" if index is None else f"placements[{index}]"
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be an object")
        row = _integer(payload.get("row"), f"{label}.row")
        column = _integer(payload.get("column"), f"{label}.column")
        if not 0 <= row < self.puzzle.size or not 0 <= column < self.puzzle.size:
            raise ValueError(f"{label} coordinates are outside the Sudoku grid")
        if self.puzzle.clues[row][column] != ".":
            raise ValueError(f"{label} cannot change clue cell ({row}, {column})")
        normalized = {"row": row, "column": column}
        if operation == "place":
            value = _integer(payload.get("value"), f"{label}.value")
            if not 1 <= value <= self.puzzle.size:
                raise ValueError(f"{label}.value must be in 1..{self.puzzle.size}")
            normalized["value"] = value
        return normalized

    def _apply(
        self,
        instance: _SudokuInstance,
        operation: str,
        payload: dict[str, int],
        *,
        replay: bool,
    ) -> str:
        normalized = self._normalize_action(operation, payload)
        row, column = normalized["row"], normalized["column"]
        instance.grid[row][column] = normalized["value"] if operation == "place" else 0
        command = _encode_command(operation, normalized)
        instance.commands.append(command)
        instance.replay_actions += int(replay)
        instance.evolution.append(
            {
                "index": len(instance.evolution),
                "operation": operation,
                "action": normalized,
                "replay": replay,
                "grid": _copy_grid(instance.grid),
            }
        )
        return command

    def _observation(self, instance: _SudokuInstance) -> dict[str, Any]:
        return {
            "puzzle_id": self.puzzle.id,
            "size": self.puzzle.size,
            "box_rows": self.puzzle.box_rows,
            "box_columns": self.puzzle.box_columns,
            "coordinate_system": "zero_based",
            "symbols": list(range(1, self.puzzle.size + 1)),
            "grid": _copy_grid(instance.grid),
            "clue_mask": [[cell != "." for cell in row] for row in self.puzzle.clues],
            "visual_grid": render_grid(
                instance.grid,
                box_rows=self.puzzle.box_rows,
                box_columns=self.puzzle.box_columns,
            ),
            **self._check(instance),
        }

    def _check(self, instance: _SudokuInstance) -> dict[str, Any]:
        violations = _constraint_violations(
            instance.grid,
            self.puzzle.box_rows,
            self.puzzle.box_columns,
        )
        filled = _filled_count(instance.grid)
        return {
            "filled_count": filled,
            "empty_count": self.puzzle.size**2 - filled,
            "total_cells": self.puzzle.size**2,
            "constraint_violations": violations,
            "violation_count": len(violations),
            "consistent": not violations,
            "complete": filled == self.puzzle.size**2,
        }

    def _summary(self, instance: _SudokuInstance) -> dict[str, Any]:
        report = self._check(instance)
        return {
            key: report[key]
            for key in (
                "filled_count",
                "empty_count",
                "total_cells",
                "violation_count",
                "consistent",
                "complete",
            )
        }


def render_grid(
    grid: list[list[int]],
    *,
    box_rows: int,
    box_columns: int,
) -> str:
    """Render a coordinate-labelled Sudoku board with visible box boundaries."""

    size = len(grid)
    width = max(1, len(str(size)))
    groups = [
        " ".join(f"{column:>{width}}" for column in range(start, start + box_columns))
        for start in range(0, size, box_columns)
    ]
    header = " " * (width + 2) + " | ".join(groups)
    rule = "-" * len(header)
    lines = [header, rule]
    for row, values in enumerate(grid):
        if row and row % box_rows == 0:
            lines.append(rule)
        cells = ["." if value == 0 else str(value) for value in values]
        chunks = [
            " ".join(f"{cell:>{width}}" for cell in cells[start : start + box_columns])
            for start in range(0, size, box_columns)
        ]
        lines.append(f"{row:>{width}}  " + " | ".join(chunks))
    return "\n".join(lines)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _encode_command(operation: str, payload: dict[str, int]) -> str:
    return json.dumps(
        {"operation": operation, **payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_command(encoded: str) -> tuple[str, dict[str, int]]:
    try:
        value = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Sudoku replay command") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid Sudoku replay command")
    operation = value.pop("operation", None)
    if operation not in {"place", "erase"}:
        raise ValueError("invalid Sudoku replay operation")
    return operation, value


def _copy_grid(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def _filled_count(grid: list[list[int]]) -> int:
    return sum(value != 0 for row in grid for value in row)


def _constraint_violations(
    grid: list[list[int]],
    box_rows: int,
    box_columns: int,
) -> list[dict[str, Any]]:
    size = len(grid)
    violations: list[dict[str, Any]] = []

    def inspect(unit: str, index: int | list[int], cells: list[tuple[int, int]]) -> None:
        by_value: dict[int, list[list[int]]] = {}
        for row, column in cells:
            value = grid[row][column]
            if value:
                by_value.setdefault(value, []).append([row, column])
        for value in sorted(by_value):
            locations = by_value[value]
            if len(locations) > 1:
                violations.append(
                    {
                        "unit": unit,
                        "index": index,
                        "value": value,
                        "cells": locations,
                    }
                )

    for row in range(size):
        inspect("row", row, [(row, column) for column in range(size)])
    for column in range(size):
        inspect("column", column, [(row, column) for row in range(size)])
    for row_start in range(0, size, box_rows):
        for column_start in range(0, size, box_columns):
            inspect(
                "box",
                [row_start // box_rows, column_start // box_columns],
                [
                    (row, column)
                    for row in range(row_start, row_start + box_rows)
                    for column in range(column_start, column_start + box_columns)
                ],
            )
    return violations


SudokuSearchEnvironmentAdapter = SudokuSearchAdapter

__all__ = [
    "MAX_SEQUENCE_LENGTH",
    "SudokuSearchAdapter",
    "SudokuSearchConfig",
    "SudokuSearchEnvironmentAdapter",
    "SudokuSearchPuzzle",
    "render_grid",
]
