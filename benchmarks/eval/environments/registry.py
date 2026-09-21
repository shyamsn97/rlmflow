"""One versioned environment factory shared by every benchmark runtime."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.base import EnvironmentAdapter
from benchmarks.eval.environments.mazebench import MazeBenchAdapter, MazeBenchConfig
from benchmarks.eval.environments.minimaze import MiniMazeAdapter, MiniMazeConfig
from benchmarks.eval.environments.sudoku_search import (
    SudokuSearchAdapter,
    SudokuSearchConfig,
)
from rlmflow.tools import tool

PARITY_CONTRACT_VERSION = "environment-parity-v1"
Perform = Callable[[str, Mapping[str, Any] | None], dict[str, Any]]
BoundTools = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvironmentBuild:
    adapter: EnvironmentAdapter
    max_actions: int
    candidate_budget: int
    contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvironmentDefinition:
    kind: str
    factory: Callable[[Mapping[str, Any], Path], EnvironmentAdapter]
    operations: tuple[OperationSpec, ...]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    annotation: Any


@dataclass(frozen=True, slots=True)
class OperationSpec:
    tool_name: str
    operation: str
    description: str
    parameters: tuple[ParameterSpec, ...] = ()


def build_environment(
    environment: Mapping[str, Any],
    *,
    artifact_dir: str | Path,
) -> EnvironmentBuild:
    kind = str(environment.get("kind", ""))
    try:
        definition = DEFINITIONS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark environment kind {kind!r}") from exc
    output = Path(artifact_dir).expanduser().resolve()
    max_actions = int(environment.get("max_actions", _default_max_actions(kind)))
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    candidate_budget = int(environment.get("candidate_budget", 0))
    if candidate_budget < 0:
        raise ValueError("candidate_budget must not be negative")
    return EnvironmentBuild(
        adapter=definition.factory(environment, output),
        max_actions=max_actions,
        candidate_budget=candidate_budget,
        contract=environment_contract(kind),
    )


def bind_tools(kind: str, perform: Perform, *, proxy: bool) -> BoundTools:
    try:
        definition = DEFINITIONS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark environment kind {kind!r}") from exc
    bound: BoundTools = {}
    for spec in definition.operations:
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    parameter.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=parameter.annotation,
                )
                for parameter in spec.parameters
            ],
            return_annotation=dict[str, Any],
        )

        def make_invoke(operation: str, tool_signature: inspect.Signature):
            def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
                payload = dict(tool_signature.bind(*args, **kwargs).arguments)
                return perform(operation, payload or None)

            invoke.__signature__ = tool_signature  # type: ignore[attr-defined]
            return invoke

        invoke = make_invoke(spec.operation, signature)

        invoke.__name__ = spec.tool_name
        invoke.__qualname__ = spec.tool_name
        if proxy:
            bound[spec.tool_name] = tool(spec.description, proxy=True)(invoke)
        else:
            bound[spec.tool_name] = {"tool": invoke, "description": spec.description}
    return bound


def environment_contract(kind: str) -> dict[str, Any]:
    try:
        definition = DEFINITIONS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark environment kind {kind!r}") from exc
    return {
        "version": PARITY_CONTRACT_VERSION,
        "kind": kind,
        "operations": [spec.operation for spec in definition.operations],
        "action_budget": "shared-row",
        "candidate_budget": "shared-row",
        "fresh_state": "fork-immediate-parent-accepted-prefix",
        "selection": "explicit-root-only",
        "native_scorer": "selected-instance",
        "official_handle_dag_credit": False,
    }


def _mazebench_factory(
    environment: Mapping[str, Any],
    output: Path,
) -> EnvironmentAdapter:
    return MazeBenchAdapter(
        MazeBenchConfig(
            artifact_dir=output,
            game_id=str(environment.get("game_id", "maze")),
            level=str(environment.get("level", "HxI")),
            view=str(environment.get("view", "top-diagonal")),
            yaw=int(environment.get("yaw", 0)),
        )
    )


def _minimaze_factory(
    environment: Mapping[str, Any],
    output: Path,
) -> EnvironmentAdapter:
    from benchmarks.eval.tasks.minimaze import FROZEN_MINIMAZE_MANIFEST

    task_id = str(environment["task_id"])
    spec = next((item for item in FROZEN_MINIMAZE_MANIFEST if item.id == task_id), None)
    if spec is None:
        raise ValueError(f"unknown frozen MiniMaze task {task_id!r}")
    return MiniMazeAdapter(MiniMazeConfig(spec=spec, artifact_dir=output))


def _sudoku_factory(
    environment: Mapping[str, Any],
    output: Path,
) -> EnvironmentAdapter:
    from benchmarks.eval.tasks.sudoku_search import ALL_PUZZLES

    puzzle_id = str(environment["puzzle_id"])
    puzzle = next((item for item in ALL_PUZZLES if item.id == puzzle_id), None)
    if puzzle is None:
        raise ValueError(f"unknown frozen Sudoku search task {puzzle_id!r}")
    return SudokuSearchAdapter(SudokuSearchConfig(puzzle=puzzle, artifact_dir=output))


DEFINITIONS: dict[str, EnvironmentDefinition] = {
    "mazebench": EnvironmentDefinition(
        "mazebench",
        _mazebench_factory,
        (
            OperationSpec("maze_observe", "observe", "Inspect the private sanitized ASCII maze."),
            OperationSpec(
                "maze_action",
                "action",
                "Apply one documented MazeBench command.",
                (ParameterSpec("action", str),),
            ),
            OperationSpec(
                "maze_action_sequence",
                "action_sequence",
                "Apply a bounded MazeBench command list.",
                (ParameterSpec("actions", list[str]),),
            ),
        ),
    ),
    "minimaze": EnvironmentDefinition(
        "minimaze",
        _minimaze_factory,
        (
            OperationSpec("observe", "observe", "Observe the private fogged maze."),
            OperationSpec(
                "action",
                "action",
                "Move up, down, left, or right.",
                (ParameterSpec("action", str),),
            ),
            OperationSpec(
                "action_sequence",
                "action_sequence",
                "Apply a bounded movement list.",
                (ParameterSpec("actions", list[str]),),
            ),
        ),
    ),
    "sudoku_search": EnvironmentDefinition(
        "sudoku_search",
        _sudoku_factory,
        (
            OperationSpec("observe", "observe", "Observe the private Sudoku grid."),
            OperationSpec(
                "place",
                "place",
                "Place one value at zero-based coordinates.",
                (
                    ParameterSpec("row", int),
                    ParameterSpec("column", int),
                    ParameterSpec("value", int),
                ),
            ),
            OperationSpec(
                "erase",
                "erase",
                "Erase one editable cell.",
                (ParameterSpec("row", int), ParameterSpec("column", int)),
            ),
            OperationSpec(
                "place_sequence",
                "place_sequence",
                "Apply a bounded placement list.",
                (ParameterSpec("placements", list[dict[str, int]]),),
            ),
            OperationSpec(
                "check",
                "check",
                "Reveal visible violations and fill progress.",
            ),
        ),
    ),
}


def _default_max_actions(kind: str) -> int:
    del kind
    return 256


__all__ = [
    "DEFINITIONS",
    "PARITY_CONTRACT_VERSION",
    "EnvironmentBuild",
    "EnvironmentDefinition",
    "OperationSpec",
    "ParameterSpec",
    "bind_tools",
    "build_environment",
    "environment_contract",
]
