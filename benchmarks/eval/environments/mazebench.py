"""Pinned MazeBench adapter using its bundled sanitized Node interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.base import EnvironmentAdapter, EnvironmentResult

PACKAGE_VERSION = "0.2.19"
UPSTREAM_REVISION = "07aa03a5c0ee8b4e0b025a52793055e89cb68bfe"
GAME_WON_GEM_COUNT = 100
DEFAULT_LEVEL = "HxI"
DEFAULT_VIEW = "top-diagonal"
DEFAULT_YAW = 0
MAX_ACTION_LENGTH = 128
MAX_ACTION_SEQUENCE = 1_000
RUNTIME_SHA256 = {
    "codex-play.js": "ff195645393e8293e18b354bfab734ec8083b617028af282f8ca983f3e7a1a71",
    "maze-bridge.js": "3410efcf60a9979faed63b025b8501db2a5b2ff85b74c87ba513850ff82edf21",
}

_DIRECTIONS = "up|down|left|right"
_ACTION_PATTERN = re.compile(
    rf"^(?:{_DIRECTIONS}|rotate camera (?:{_DIRECTIONS})|undo|reset|"
    r"go to level [A-Z] [A-Z]|quit)$",
    re.IGNORECASE,
)
_OBSERVATION_KEYS = {
    "observation_mode",
    "current_room",
    "current_view",
    "yaw",
    "gem_count",
    "visited_levels",
    "player_dead",
    "game_won",
    "game_lost",
    "level",
    "ascii_legend",
    "death_message",
    "allowed_commands",
}


def preflight_mazebench() -> Path:
    """Resolve and validate the exact optional MazeBench runtime."""
    try:
        installed = metadata.version("mazebench")
        from mazebench_cli import __version__, resolve_root
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "MazeBench benchmarks require `mazebench==0.2.19` and Node.js; "
            "install the environment benchmark dependencies first"
        ) from exc
    if installed != PACKAGE_VERSION or __version__ != PACKAGE_VERSION:
        raise RuntimeError(
            f"MazeBench requires mazebench=={PACKAGE_VERSION}; found "
            f"distribution {installed!r}, runtime {__version__!r}"
        )
    node = os.environ.get("MAZEBENCH_NODE", "node")
    if shutil.which(node) is None:
        raise RuntimeError("MazeBench requires Node.js; `node` was not found on PATH")
    try:
        root = Path(resolve_root()).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"MazeBench could not resolve its bundled runtime: {exc}") from exc
    required = {name: root / "scripts" / name for name in RUNTIME_SHA256}
    if not all(path.is_file() for path in required.values()):
        raise RuntimeError("MazeBench's bundled authoritative JavaScript runtime is incomplete")
    mismatched = [
        name
        for name, path in required.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != RUNTIME_SHA256[name]
    ]
    if mismatched:
        raise RuntimeError(
            "MazeBench's JavaScript runtime does not match the pinned upstream "
            f"revision {UPSTREAM_REVISION}: {', '.join(mismatched)}"
        )
    return root


def _canonical_action(value: object, *, index: int | None = None) -> str:
    label = "action" if index is None else f"actions[{index}]"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    action = " ".join(value.strip().split())
    if len(action) > MAX_ACTION_LENGTH:
        raise ValueError(f"{label} exceeds {MAX_ACTION_LENGTH} characters")
    if not _ACTION_PATTERN.fullmatch(action):
        raise ValueError(f"{label} is not an accepted MazeBench command")
    return action.lower()


def _public_observation(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in _OBSERVATION_KEYS if key in value}


def _summary(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: observation[key]
        for key in (
            "current_room",
            "current_view",
            "yaw",
            "gem_count",
            "visited_levels",
            "player_dead",
            "game_won",
            "game_lost",
        )
        if key in observation
    }


def _strip_private(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        str(key): _strip_private(nested)
        for key, nested in value.items()
        if not str(key).startswith("_")
        and "checkpoint" not in str(key).casefold()
        and str(key).casefold() != "scorecard"
    }


@dataclass(frozen=True, slots=True)
class MazeBenchConfig:
    artifact_dir: Path
    game_id: str = "maze"
    level: str = DEFAULT_LEVEL
    view: str = DEFAULT_VIEW
    yaw: int = DEFAULT_YAW
    allow_quit: bool = False
    render_video: bool = True
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_dir",
            self.artifact_dir.expanduser().resolve(),
        )
        if self.game_id != "maze":
            raise ValueError("the benchmark adapter uses MazeBench's official `maze` world")
        if self.level not in {"HxI", "level_HxI"}:
            raise ValueError("the benchmark adapter uses the fixed HxI start")
        if self.view != DEFAULT_VIEW or self.yaw != DEFAULT_YAW:
            raise ValueError("the benchmark adapter uses the fixed top-diagonal/yaw-0 view")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")


class MazeBenchAdapter(EnvironmentAdapter):
    """Host-side adapter over ``scripts/codex-play.js``."""

    kind = "mazebench"

    def __init__(self, config: MazeBenchConfig) -> None:
        self.config = config
        self._root: Path | None = None
        self._instances: dict[str, Path] = {}
        self._finalized: set[str] = set()
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = preflight_mazebench()
        return self._root

    def start(self, instance_id: str) -> EnvironmentResult:
        state = self._new_state_path(instance_id)
        args = [
            "start",
            "--repo-root",
            str(self.root),
            "--state",
            str(state),
            "--game",
            self.config.game_id,
            "--level",
            self.config.level,
            "--view",
            self.config.view,
            "--yaw",
            str(self.config.yaw),
            "--max-actions",
            "unlimited",
        ]
        if not self.config.allow_quit:
            args.append("--no-quit")
        observation = _public_observation(self._run(args))
        self._instances[instance_id] = state
        return EnvironmentResult(observation, observation_summary=_summary(observation))

    def fork(
        self,
        instance_id: str,
        *,
        parent_instance_id: str,
        command_prefix: tuple[str, ...],
    ) -> EnvironmentResult:
        if parent_instance_id not in self._instances:
            raise KeyError(f"unknown parent environment instance {parent_instance_id!r}")
        result = self.start(instance_id)
        if command_prefix:
            self._run(
                ["action-sequence", "--state", str(self._instances[instance_id])],
                input_value={"actions": list(command_prefix), "primary": False},
            )
            observation = self._observation(instance_id)
            result = EnvironmentResult(observation, observation_summary=_summary(observation))
        return result

    def execute(
        self,
        instance_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        max_actions: int,
    ) -> EnvironmentResult:
        state = self._state_path(instance_id)
        if operation == "observe":
            observation = _public_observation(self._run(["observe", "--state", str(state)]))
            return EnvironmentResult(observation, observation_summary=_summary(observation))
        if operation == "action":
            if max_actions < 1:
                raise RuntimeError("the shared environment action budget is exhausted")
            action = _canonical_action(payload.get("action"))
            before = self._actions(instance_id)
            raw = self._run(["action", "--state", str(state), action])
            accepted = self._accepted_since(instance_id, before)
            observation = _public_observation(raw)
            return EnvironmentResult(
                {
                    "observation": observation,
                    "accepted": bool(accepted),
                    "actions_remaining": max(0, max_actions - len(accepted)),
                },
                accepted_actions=accepted,
                observation_summary=_summary(observation),
            )
        if operation == "action_sequence":
            requested = payload.get("actions")
            if not isinstance(requested, list) or not requested:
                raise ValueError("actions must contain at least one item")
            if len(requested) > MAX_ACTION_SEQUENCE:
                raise ValueError(f"actions must contain at most {MAX_ACTION_SEQUENCE} items")
            actions = [
                _canonical_action(action, index=index) for index, action in enumerate(requested)
            ]
            selected = actions[:max_actions]
            if not selected:
                raise RuntimeError("the shared environment action budget is exhausted")
            before = self._actions(instance_id)
            raw = self._run(
                ["action-sequence", "--state", str(state)],
                input_value={"actions": selected, "primary": self._is_primary(instance_id)},
            )
            accepted = self._accepted_since(instance_id, before)
            observation = self._observation(instance_id)
            steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
            public_steps = [
                {
                    key: step[key]
                    for key in (
                        "index",
                        "action",
                        "recorded",
                        "action_count_before",
                        "action_count_after",
                        "error",
                    )
                    if key in step
                }
                for step in steps
                if isinstance(step, dict)
            ]
            return EnvironmentResult(
                {
                    "requested_count": len(actions),
                    "attempted_count": int(raw.get("attempted_count") or 0),
                    "completed_count": len(accepted),
                    "stopped_early": len(accepted) < len(actions),
                    "stop_reason": (
                        "shared_budget_exhausted"
                        if len(selected) < len(actions) and len(accepted) == len(selected)
                        else str(raw.get("stop_reason") or "completed")
                    ),
                    "steps": public_steps,
                    "final_observation": observation,
                    "actions_remaining": max(0, max_actions - len(accepted)),
                },
                accepted_actions=accepted,
                observation_summary=_summary(observation),
            )
        raise ValueError(f"unknown MazeBench operation {operation!r}")

    def primary_metrics(self, instance_id: str) -> dict[str, Any]:
        session = self._session(instance_id)
        status = session.get("lastStatus") if isinstance(session.get("lastStatus"), dict) else {}
        actions = session.get("actions") if isinstance(session.get("actions"), list) else []
        hashes = {
            str(item.get("status", {}).get("board_state_hash"))
            for item in actions
            if isinstance(item, dict)
            and isinstance(item.get("status"), dict)
            and item["status"].get("board_state_hash")
        }
        initial = session.get("initial")
        if isinstance(initial, dict) and initial.get("board_state_hash"):
            hashes.add(str(initial["board_state_hash"]))
        gems = max(0, int(status.get("gem_count") or 0))
        return {
            "score": float(gems),
            "gems_collected": gems,
            "rooms_visited": len(status.get("visited_levels") or []),
            "novel_block_positions": max(0, int(status.get("novel_push_count") or 0)),
            "block_pushes": max(0, int(status.get("push_count") or 0)),
            "unique_states": len(hashes),
            "accepted_actions": len(actions),
            "game_won": gems >= GAME_WON_GEM_COUNT,
            "win_condition_gems": GAME_WON_GEM_COUNT,
        }

    def artifacts(self, instance_id: str) -> dict[str, str]:
        state = self._state_path(instance_id)
        if instance_id not in self._finalized:
            self._run(
                ["finalize", "--state", str(state)],
                trusted_env={"MAZEBENCH_TRUSTED_FINALIZE": "1"},
            )
            self._finalized.add(instance_id)
        session = self._session(instance_id)
        replay_path = self.config.artifact_dir / "maze_replay.json"
        scorecard_path = self.config.artifact_dir / "maze_scorecard.json"
        actions = session.get("actions") if isinstance(session.get("actions"), list) else []
        replay = {
            "source": "mazebench",
            "package_version": PACKAGE_VERSION,
            "upstream_revision": UPSTREAM_REVISION,
            "game_id": self.config.game_id,
            "start_level": self.config.level,
            "view": self.config.view,
            "yaw": self.config.yaw,
            "game_won_gem_count": GAME_WON_GEM_COUNT,
            "initial": _public_observation(session.get("initial")),
            "actions": [
                {
                    "turn": action.get("turn"),
                    "timestamp": action.get("timestamp"),
                    "command": action.get("command_text"),
                    "observation": _public_observation(action.get("status")),
                }
                for action in actions
                if isinstance(action, dict)
            ],
        }
        scorecard = session.get("scorecard")
        if not isinstance(scorecard, dict):
            scorecard = {}
        replay_path.write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
        scorecard_path.write_text(
            json.dumps(_strip_private(scorecard), indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            "maze_replay": str(replay_path),
            "maze_scorecard": str(scorecard_path),
        }
        if video := self._render_video(state):
            artifacts["maze_video"] = str(video)
        return artifacts

    def close(self) -> None:
        return None

    def _new_state_path(self, instance_id: str) -> Path:
        path = self.config.artifact_dir / "instances" / instance_id / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        return path

    def _state_path(self, instance_id: str) -> Path:
        try:
            return self._instances[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown MazeBench instance {instance_id!r}") from exc

    def _session(self, instance_id: str) -> dict[str, Any]:
        try:
            value = json.loads(self._state_path(instance_id).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MazeBench private session state is unavailable") from exc
        if not isinstance(value, dict):
            raise TypeError("MazeBench private session state is invalid")
        return value

    def _actions(self, instance_id: str) -> int:
        actions = self._session(instance_id).get("actions")
        return len(actions) if isinstance(actions, list) else 0

    def _accepted_since(self, instance_id: str, before: int) -> tuple[str, ...]:
        actions = self._session(instance_id).get("actions")
        if not isinstance(actions, list):
            return ()
        return tuple(
            str(action.get("command_text") or "")
            for action in actions[before:]
            if isinstance(action, dict)
            and action.get("valid", True) is not False
            and not action.get("error")
            and action.get("command_text")
        )

    def _observation(self, instance_id: str) -> dict[str, Any]:
        return _public_observation(
            self._run(["observe", "--state", str(self._state_path(instance_id))])
        )

    def _is_primary(self, instance_id: str) -> bool:
        return bool(self._instances) and next(iter(self._instances)) == instance_id

    def _render_video(self, state: Path) -> Path | None:
        if not self.config.render_video or shutil.which("ffmpeg") is None:
            return None
        script = self.root / "scripts" / "maze-export-replay.js"
        if not script.is_file():
            return None
        node = os.environ.get("MAZEBENCH_NODE", "node")
        browser = os.environ.get("MAZEBENCH_BROWSER")
        if not browser:
            browser = next(
                (
                    candidate
                    for candidate in (
                        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                        "/Applications/Chromium.app/Contents/MacOS/Chromium",
                        "/usr/bin/chromium",
                        "/usr/bin/google-chrome",
                    )
                    if Path(candidate).is_file()
                ),
                None,
            )
        if browser is None:
            return None
        playwright = subprocess.run(
            [node, "-e", "require.resolve('playwright-core')"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if playwright.returncode:
            return None
        completed = subprocess.run(
            [
                node,
                str(script),
                str(state),
                "--out-dir",
                str(self.config.artifact_dir),
                "--video",
                "--fast",
                "--browser",
                browser,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=max(self.config.timeout_seconds, 600),
            check=False,
        )
        path = self.config.artifact_dir / "maze_replay.mp4"
        return path if completed.returncode == 0 and path.is_file() else None

    def _run(
        self,
        args: list[str],
        *,
        input_value: dict[str, Any] | None = None,
        trusted_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        env = {
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
            **(trusted_env or {}),
        }
        node = os.environ.get("MAZEBENCH_NODE", "node")
        completed = subprocess.run(
            [node, str(self.root / "scripts" / "codex-play.js"), *args],
            cwd=self.root,
            env=env,
            input=json.dumps(input_value) if input_value is not None else None,
            text=True,
            capture_output=True,
            timeout=max(10, self.config.timeout_seconds),
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise RuntimeError((detail[-1] if detail else "MazeBench command failed")[:500])
        text = completed.stdout.lstrip()
        if not text:
            return {}
        try:
            value, _end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MazeBench returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError("MazeBench returned an invalid response")
        return value


__all__ = [
    "DEFAULT_LEVEL",
    "DEFAULT_VIEW",
    "DEFAULT_YAW",
    "GAME_WON_GEM_COUNT",
    "MazeBenchAdapter",
    "MazeBenchConfig",
    "PACKAGE_VERSION",
    "RUNTIME_SHA256",
    "UPSTREAM_REVISION",
    "preflight_mazebench",
]
