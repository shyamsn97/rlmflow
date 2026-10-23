"""Deterministic, offline ASCII mazes for visual branch-and-select evaluation."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from benchmarks.eval.environments.environment import Environment, Observation
from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics
from benchmarks.eval.telemetry import EnvironmentMetricsView
from benchmarks.eval.types import Prediction, Score

ACTIONS = ("up", "down", "left", "right")
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_ACTION_SEQUENCE = 128
SCORE_WEIGHTS = {
    "gem_completion": 0.60,
    "coverage": 0.25,
    "path_efficiency": 0.15,
}


class MiniMazeProtocolError(ValueError):
    """A MiniMaze operation or action did not match the public protocol."""


@dataclass(frozen=True, slots=True)
class MiniMazeSpec:
    """One frozen maze layout.

    ``S`` is the start, ``*`` is a gem, lowercase letters are reusable keys,
    and matching uppercase letters are locked doors.
    """

    id: str
    rows: tuple[str, ...]
    fog_radius: int = 2
    action_budget: int = 48
    candidate_budget: int = 6

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MiniMaze ids must be non-empty")
        if len(self.rows) < 3 or len({len(row) for row in self.rows}) != 1:
            raise ValueError(f"{self.id}: maze rows must form a rectangle")
        if len(self.rows[0]) < 3:
            raise ValueError(f"{self.id}: maze must be at least 3x3")
        if any(not row.startswith("#") or not row.endswith("#") for row in self.rows):
            raise ValueError(f"{self.id}: maze must have a wall boundary")
        if set(self.rows[0] + self.rows[-1]) != {"#"}:
            raise ValueError(f"{self.id}: maze must have a wall boundary")
        cells = "".join(self.rows)
        allowed = set("#.S* abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        unknown = set(cells) - allowed
        if unknown:
            raise ValueError(f"{self.id}: unsupported cells {sorted(unknown)!r}")
        if cells.count("S") != 1:
            raise ValueError(f"{self.id}: maze must contain exactly one start")
        if "*" not in cells:
            raise ValueError(f"{self.id}: maze must contain at least one gem")
        keys = {cell for cell in cells if cell.islower()}
        doors = {cell for cell in cells if cell.isupper() and cell != "S"}
        if not keys or not doors or {door.lower() for door in doors} - keys:
            raise ValueError(f"{self.id}: every maze needs matching keys and doors")
        if self.fog_radius < 1:
            raise ValueError("fog_radius must be positive")
        for name, value in (
            ("action_budget", self.action_budget),
            ("candidate_budget", self.candidate_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])


@dataclass(frozen=True, slots=True)
class MiniMazeConfig:
    spec: MiniMazeSpec
    artifact_dir: str | Path | None = None


@dataclass
class _MiniMazeInstance:
    position: tuple[int, int]
    keys: set[str] = field(default_factory=set)
    gems: set[tuple[int, int]] = field(default_factory=set)
    seen: set[tuple[int, int]] = field(default_factory=set)
    visited: set[tuple[int, int]] = field(default_factory=set)
    commands: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    parent_instance_id: str | None = None
    replay_actions: int = 0


@dataclass(frozen=True, slots=True)
class MiniMazeObservation(Observation):
    task_id: str
    ascii: str
    legend: dict[str, str]
    position: tuple[int, int]
    keys: tuple[str, ...]
    gems_collected: int
    total_gems: int
    coverage: float
    completed: bool
    legal_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "task_id": self.task_id,
            "ascii": self.ascii,
            "legend": self.legend,
            "position": list(self.position),
            "keys": list(self.keys),
            "gems_collected": self.gems_collected,
            "total_gems": self.total_gems,
            "coverage": self.coverage,
            "completed": self.completed,
            "legal_actions": list(self.legal_actions),
        }


@dataclass(frozen=True, slots=True)
class MiniMazeMetrics:
    values: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


def normalize_action(value: object, *, index: int | None = None) -> str:
    """Validate and canonicalize one movement command."""

    label = "action" if index is None else f"actions[{index}]"
    if not isinstance(value, str) or not value.strip():
        raise MiniMazeProtocolError(f"{label} must be a non-empty string")
    action = value.strip().lower()
    if action not in ACTION_DELTAS:
        raise MiniMazeProtocolError(f"{label} must be one of {', '.join(ACTIONS)}")
    return action


class MiniMazeEnvironment(Environment[MiniMazeObservation, MiniMazeMetrics]):
    """Small deterministic maze with replayable private branch state."""

    kind = "minimaze"
    operations = ("observe", "action", "action_sequence")
    tool_operations = (
        ("observe", "observe", "Observe the private fogged maze.", ()),
        (
            "action",
            "action",
            "Move up, down, left, or right.",
            (("action", str),),
        ),
        (
            "action_sequence",
            "action_sequence",
            "Apply a bounded movement list.",
            (("actions", list[str]),),
        ),
    )

    def __init__(
        self,
        config: MiniMazeConfig | MiniMazeSpec,
        *,
        artifact_dir: str | Path | None = None,
        description: str | None = None,
        environment_id: str | None = None,
    ) -> None:
        if isinstance(config, MiniMazeConfig):
            self.spec = config.spec
            configured_artifacts = config.artifact_dir
        else:
            self.spec = config
            configured_artifacts = artifact_dir
        self.artifact_dir = (
            Path(configured_artifacts).expanduser().resolve()
            if configured_artifacts is not None
            else None
        )
        self._instances: dict[str, _MiniMazeInstance] = {}
        self.id = environment_id or f"minimaze_{self.spec.id}"
        self.action_budget = self.spec.action_budget
        self.candidate_budget = self.spec.candidate_budget
        self.description = description or (
            "Navigate the fog-of-war MiniMaze and collect every gem. "
            "Use observe, action, and action_sequence."
        )
        self._start = self._find("S")[0]
        self._all_gems = frozenset(self._find("*"))
        self._traversable = frozenset(
            (row, column)
            for row, line in enumerate(self.spec.rows)
            for column, cell in enumerate(line)
            if cell != "#"
        )
        self._optimal_actions = self._shortest_completion()
        if self._optimal_actions is None:
            raise ValueError(f"{self.spec.id}: all gems must be reachable")

    def start(self, instance_id: str) -> MiniMazeObservation:
        if instance_id in self._instances:
            raise ValueError(f"MiniMaze instance {instance_id!r} already exists")
        instance = _MiniMazeInstance(position=self._start)
        instance.visited.add(self._start)
        self._reveal(instance)
        self._instances[instance_id] = instance
        observation = self._observation(instance)
        instance.events.append(
            {
                "index": 0,
                "kind": "observation",
                "replay": False,
                "observation": observation.to_dict(),
            }
        )
        return observation

    def observe(self, instance_id: str) -> MiniMazeObservation:
        return self._observation(self._instance(instance_id))

    def fork(
        self,
        instance_id: str,
        *,
        parent_instance_id: str,
        command_prefix: tuple[str, ...],
    ) -> MiniMazeObservation:
        if parent_instance_id not in self._instances:
            raise KeyError(f"unknown immediate parent MiniMaze instance {parent_instance_id!r}")
        self.start(instance_id)
        instance = self._instance(instance_id)
        instance.parent_instance_id = parent_instance_id
        for index, raw_action in enumerate(command_prefix):
            action = normalize_action(raw_action, index=index)
            self._apply(instance, action, replay=True)
            instance.replay_actions += 1
        observation = self._observation(instance)
        return observation

    def execute(
        self,
        instance_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        max_actions: int,
    ) -> MiniMazeObservation:
        instance = self._instance(instance_id)
        if operation == "observe":
            if payload:
                raise MiniMazeProtocolError("observe takes no payload")
            observation = self._observation(instance)
            instance.events.append(
                {
                    "index": len(instance.events),
                    "kind": "observation",
                    "replay": False,
                    "observation": observation.to_dict(),
                }
            )
            return observation

        if operation == "action":
            if set(payload) != {"action"}:
                raise MiniMazeProtocolError("action requires exactly the `action` field")
            requested = [normalize_action(payload["action"])]
        elif operation == "action_sequence":
            if set(payload) != {"actions"}:
                raise MiniMazeProtocolError("action_sequence requires exactly the `actions` field")
            values = payload["actions"]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise MiniMazeProtocolError("actions must be a sequence of movement strings")
            if not 1 <= len(values) <= MAX_ACTION_SEQUENCE:
                raise MiniMazeProtocolError(
                    f"actions must contain 1..{MAX_ACTION_SEQUENCE} movements"
                )
            requested = [normalize_action(value, index=index) for index, value in enumerate(values)]
        else:
            raise MiniMazeProtocolError(f"unknown MiniMaze operation {operation!r}")

        selected = requested[:max_actions]
        if not selected:
            raise RuntimeError("the shared MiniMaze action budget is exhausted")
        for action in selected:
            self._apply(instance, action, replay=False)
            if self._complete(instance):
                break
        observation = self._observation(instance)
        return observation

    def accepted_prefix(self, instance_id: str) -> tuple[str, ...]:
        return tuple(self._instance(instance_id).commands)

    def metrics(self, instance_id: str) -> MiniMazeMetrics:
        instance = self._instance(instance_id)
        gems_collected = len(instance.gems)
        total_gems = len(self._all_gems)
        gem_completion = gems_collected / total_gems
        seen_walkable = len(instance.seen & self._traversable)
        coverage = seen_walkable / len(self._traversable)
        actions = len(instance.commands)
        if not actions:
            path_efficiency = 0.0
        elif gems_collected == total_gems:
            path_efficiency = min(1.0, self._optimal_actions / actions)
        else:
            path_efficiency = min(1.0, max(0, len(instance.visited) - 1) / actions)
        score = (
            SCORE_WEIGHTS["gem_completion"] * gem_completion
            + SCORE_WEIGHTS["coverage"] * coverage
            + SCORE_WEIGHTS["path_efficiency"] * path_efficiency
        )
        return MiniMazeMetrics(
            {
                "score": min(1.0, max(0.0, score)),
                "task_id": self.spec.id,
                "completed": gems_collected == total_gems,
                "gems_collected": gems_collected,
                "total_gems": total_gems,
                "gem_completion": gem_completion,
                "coverage": coverage,
                "path_efficiency": path_efficiency,
                "cells_seen": seen_walkable,
                "total_traversable_cells": len(self._traversable),
                "cells_visited": len(instance.visited),
                "keys_collected": sorted(instance.keys),
                "doors_unlocked": sorted(
                    {
                        cell
                        for row in self.spec.rows
                        for cell in row
                        if cell.isupper() and cell != "S" and cell.lower() in instance.keys
                    }
                ),
                "accepted_actions": actions,
                "live_actions": actions - instance.replay_actions,
                "replay_actions": instance.replay_actions,
                "optimal_completion_actions": self._optimal_actions,
                "position": list(instance.position),
                "score_weights": dict(SCORE_WEIGHTS),
            }
        )

    def candidate_metrics(self, instance_id: str) -> tuple[float, dict[str, Any]]:
        metrics = self.metrics(instance_id).to_dict()
        public = {
            key: metrics[key]
            for key in (
                "task_id",
                "completed",
                "gems_collected",
                "total_gems",
                "gem_completion",
                "coverage",
                "path_efficiency",
                "cells_seen",
                "cells_visited",
                "keys_collected",
                "doors_unlocked",
                "accepted_actions",
                "position",
            )
        }
        return float(metrics["score"]), public

    def report(
        self,
        prediction: Prediction,
        runtime: RuntimeMetrics,
    ) -> EnvironmentReport[MiniMazeMetrics]:
        view = EnvironmentMetricsView.from_prediction(prediction)
        raw_environment = prediction.metrics.get("environment", {})
        if not isinstance(raw_environment, dict):
            raw_environment = {}
        native = view.native_float("score", bounded=True)
        selected = not view.selection_required or view.selection_made
        primary = view.primary
        score = Score(
            value=native if selected else 0.0,
            correct=bool(primary.get("completed") and selected),
            details={
                **view.common_details(),
                "native_progress_score": native,
                "completed": bool(primary.get("completed")),
            },
        )
        return EnvironmentReport(
            benchmark="",
            environment_id=self.id,
            runner="",
            model="",
            seed=None,
            answer=prediction.answer,
            score=score,
            description_sha256="",
            observation_sha256="",
            runtime=runtime,
            environment=MiniMazeMetrics(dict(raw_environment)),
            artifacts=prediction.artifacts,
            error=prediction.error,
        )

    def artifacts(self, instance_id: str) -> dict[str, str]:
        if self.artifact_dir is None:
            return {}
        self._instance(instance_id)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / "minimaze_replay.json"
        payload = {
            "schema_version": 1,
            "kind": self.kind,
            "task_id": self.spec.id,
            "layout": list(self.spec.rows),
            "fog_radius": self.spec.fog_radius,
            "selected_instance_id": instance_id,
            "instances": {
                current_id: {
                    "selected": current_id == instance_id,
                    "parent_instance_id": current.parent_instance_id,
                    "commands": list(current.commands),
                    "replay_actions": current.replay_actions,
                    "observations": [
                        event["observation"]
                        for event in current.events
                        if event["kind"] in {"observation", "action"}
                    ],
                    "actions": [
                        {
                            key: value
                            for key, value in event.items()
                            if key in {"index", "action", "replay", "from", "to", "outcome"}
                        }
                        for event in current.events
                        if event["kind"] == "action"
                    ],
                    "metrics": self.metrics(current_id).to_dict(),
                }
                for current_id, current in self._instances.items()
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"minimaze_replay": str(path)}

    def _instance(self, instance_id: str) -> _MiniMazeInstance:
        try:
            return self._instances[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown MiniMaze instance {instance_id!r}") from exc

    def _find(self, token: str) -> list[tuple[int, int]]:
        return [
            (row, column)
            for row, line in enumerate(self.spec.rows)
            for column, cell in enumerate(line)
            if cell == token
        ]

    def _cell(self, position: tuple[int, int]) -> str:
        return self.spec.rows[position[0]][position[1]]

    def _apply(
        self,
        instance: _MiniMazeInstance,
        action: str,
        *,
        replay: bool,
    ) -> dict[str, Any]:
        before = instance.position
        delta = ACTION_DELTAS[action]
        target = (before[0] + delta[0], before[1] + delta[1])
        cell = self._cell(target)
        outcome = "moved"
        if cell == "#":
            target = before
            outcome = "wall"
        elif cell.isupper() and cell != "S" and cell.lower() not in instance.keys:
            target = before
            outcome = f"locked:{cell}"
        else:
            instance.position = target
            instance.visited.add(target)
            if cell.islower() and cell not in instance.keys:
                instance.keys.add(cell)
                outcome = f"key:{cell}"
            if cell == "*" and target not in instance.gems:
                instance.gems.add(target)
                outcome = "gem"
        instance.commands.append(action)
        self._reveal(instance)
        observation = self._observation(instance)
        event = {
            "index": len(instance.events),
            "kind": "action",
            "action": action,
            "replay": replay,
            "from": list(before),
            "to": list(instance.position),
            "outcome": outcome,
            "observation": observation.to_dict(),
        }
        instance.events.append(event)
        return event

    def _reveal(self, instance: _MiniMazeInstance) -> None:
        player_row, player_column = instance.position
        for row in range(self.spec.height):
            for column in range(self.spec.width):
                if abs(row - player_row) + abs(column - player_column) <= self.spec.fog_radius:
                    instance.seen.add((row, column))

    def _observation(self, instance: _MiniMazeInstance) -> MiniMazeObservation:
        rendered: list[str] = []
        for row, line in enumerate(self.spec.rows):
            output: list[str] = []
            for column, cell in enumerate(line):
                position = (row, column)
                if position == instance.position:
                    output.append("@")
                elif position not in instance.seen:
                    output.append("?")
                elif cell == "S":
                    output.append(".")
                elif cell == "*" and position in instance.gems:
                    output.append(".")
                elif cell.islower() and cell in instance.keys:
                    output.append(".")
                elif cell.isupper() and cell != "S" and cell.lower() in instance.keys:
                    output.append("/")
                elif cell == " ":
                    output.append(".")
                else:
                    output.append(cell)
            rendered.append("".join(output))
        metrics = self._visible_metrics(instance)
        return MiniMazeObservation(
            task_id=self.spec.id,
            ascii="\n".join(rendered),
            legend={
                "@": "player",
                "#": "wall",
                ".": "explored floor",
                "?": "fog",
                "*": "gem",
                "a-z": "key",
                "A-Z": "locked matching door",
                "/": "unlocked door",
            },
            position=instance.position,
            keys=tuple(sorted(instance.keys)),
            gems_collected=len(instance.gems),
            total_gems=len(self._all_gems),
            coverage=metrics["coverage"],
            completed=self._complete(instance),
            legal_actions=ACTIONS,
        )

    def _visible_metrics(self, instance: _MiniMazeInstance) -> dict[str, float]:
        seen_walkable = len(instance.seen & self._traversable)
        return {"coverage": seen_walkable / len(self._traversable)}

    def _complete(self, instance: _MiniMazeInstance) -> bool:
        return instance.gems == self._all_gems

    def _shortest_completion(self) -> int | None:
        """Return the exact shortest all-gem route in the static key-door world."""

        gem_indexes = {position: index for index, position in enumerate(sorted(self._all_gems))}
        all_gems = (1 << len(gem_indexes)) - 1
        initial = (self._start, frozenset(), 0)
        queue = deque([(initial, 0)])
        seen = {initial}
        while queue:
            (position, keys, gems), distance = queue.popleft()
            if gems == all_gems:
                return distance
            for delta in ACTION_DELTAS.values():
                target = (position[0] + delta[0], position[1] + delta[1])
                cell = self._cell(target)
                if cell == "#" or (cell.isupper() and cell != "S" and cell.lower() not in keys):
                    continue
                next_keys = keys | {cell} if cell.islower() else keys
                next_gems = gems | (1 << gem_indexes[target]) if target in gem_indexes else gems
                state = (target, frozenset(next_keys), next_gems)
                if state not in seen:
                    seen.add(state)
                    queue.append((state, distance + 1))
        return None


MiniMazeAdapter = MiniMazeEnvironment
MiniMazeEnvironmentAdapter = MiniMazeEnvironment

__all__ = [
    "ACTIONS",
    "ACTION_DELTAS",
    "MAX_ACTION_SEQUENCE",
    "SCORE_WEIGHTS",
    "MiniMazeAdapter",
    "MiniMazeConfig",
    "MiniMazeEnvironment",
    "MiniMazeEnvironmentAdapter",
    "MiniMazeMetrics",
    "MiniMazeObservation",
    "MiniMazeProtocolError",
    "MiniMazeSpec",
    "normalize_action",
]
