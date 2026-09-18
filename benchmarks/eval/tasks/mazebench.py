"""Official MazeBench world as a bounded recursive-environment benchmark."""

from __future__ import annotations

from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.environments.mazebench import (
    DEFAULT_LEVEL,
    DEFAULT_VIEW,
    DEFAULT_YAW,
    GAME_WON_GEM_COUNT,
    PACKAGE_VERSION,
    UPSTREAM_REVISION,
    preflight_mazebench,
)
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset(
    "mazebench",
    tags=["interactive", "environment", "visual"],
)
class MazeBenchDataset(Dataset):
    """One fixed ASCII start in MazeBench's official 100-gem world."""

    def __init__(
        self,
        max_actions: int = 256,
        level: str = DEFAULT_LEVEL,
        view: str = DEFAULT_VIEW,
        yaw: int = DEFAULT_YAW,
        preflight: bool = True,
    ) -> None:
        if not isinstance(max_actions, int) or isinstance(max_actions, bool) or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        if level != DEFAULT_LEVEL or view != DEFAULT_VIEW or yaw != DEFAULT_YAW:
            raise ValueError(
                "the official MazeBench adapter uses the fixed HxI/top-diagonal/yaw-0 start"
            )
        self.max_actions = max_actions
        self.level = level
        self.view = view
        self.yaw = yaw
        self.run_preflight = preflight

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        if limit is not None and limit < 1:
            return []
        if self.run_preflight:
            preflight_mazebench()
        return [
            Example(
                id="delegation_mazebench_hxi",
                prompt=(
                    "Play MazeBench from the fixed HxI top-diagonal ASCII start. Inspect "
                    "`INPUTS['environment_observation']`, then use `maze_observe()`, "
                    "`maze_action(action)`, or `maze_action_sequence(actions)`. Build a "
                    "compact map as you explore and collect as many unique gems as possible. "
                    "The native win condition is exactly 100 gems. Root actions determine "
                    "the score; child trajectories are private probes, and all trajectories "
                    "share one action budget. Finish with a short route summary."
                ),
                expected={"game_won_gem_count": GAME_WON_GEM_COUNT},
                metadata={
                    "source": "mazebench/MazeBenchEngine",
                    "source_id": "official-world:H×I",
                    "revision": UPSTREAM_REVISION,
                    "package": f"mazebench=={PACKAGE_VERSION}",
                    "license": "MIT",
                    "domain": "mazebench",
                    "route_hypothesis": "dependent",
                    "native_scorer": "mazebench_primary_gems",
                    "environment": {
                        "kind": "mazebench",
                        "game_id": "maze",
                        "level": self.level,
                        "view": self.view,
                        "yaw": self.yaw,
                        "observation_mode": "ascii",
                        "game_won_gem_count": GAME_WON_GEM_COUNT,
                        "max_actions": self.max_actions,
                        "package_version": PACKAGE_VERSION,
                        "revision": UPSTREAM_REVISION,
                    },
                },
            )
        ]

    def score(self, example: Example, prediction: Prediction) -> Score:
        del example
        environment = prediction.metrics.get("environment")
        primary = environment.get("primary") if isinstance(environment, dict) else {}
        primary = primary if isinstance(primary, dict) else {}
        gems = _nonnegative_int(primary.get("gems_collected"))
        game_won = bool(primary.get("game_won")) and gems >= GAME_WON_GEM_COUNT
        details: dict[str, Any] = {
            "gems_collected": gems,
            "game_won": game_won,
            "win_condition_gems": GAME_WON_GEM_COUNT,
            "rooms_visited": _nonnegative_int(primary.get("rooms_visited")),
            "novel_block_positions": _nonnegative_int(primary.get("novel_block_positions")),
            "unique_states": _nonnegative_int(primary.get("unique_states")),
            "primary_actions": _nonnegative_int(
                environment.get("primary_actions") if isinstance(environment, dict) else 0
            ),
            "probe_actions": _nonnegative_int(
                environment.get("probe_actions") if isinstance(environment, dict) else 0
            ),
            "instances_forked": _nonnegative_int(
                environment.get("instances_forked") if isinstance(environment, dict) else 0
            ),
        }
        return Score(value=float(gems), correct=game_won, details=details)


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "GAME_WON_GEM_COUNT",
    "MazeBenchDataset",
    "PACKAGE_VERSION",
    "UPSTREAM_REVISION",
]
