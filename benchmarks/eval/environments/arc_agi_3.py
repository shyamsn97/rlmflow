"""Offline ARC-AGI-3 adapter for the benchmark episode API.

``arc_agi`` and ``arcengine`` are optional and imported only when an episode
starts. Games are loaded exclusively from ``ARC3_CORPUS`` in OFFLINE mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.base import EnvironmentAdapter, EnvironmentResult

WRAPPER_REVISION = "398d4dd63cf01d00adbea41c13437ba0b8ad40fc"
ARC_AGI_VERSION = "0.9.9"
ARCENGINE_VERSION = "0.9.3"
DEFAULT_ACTION_BUDGET = 500
DEFAULT_ARC3_CORPUS = Path.home() / ".cache" / "rlmflow" / "arc3" / "environment_files"
MAX_BATCH_ACTIONS = 20
GRID_MIN = 0
GRID_MAX = 63

FROZEN_PUBLIC_GAME_IDS = (
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
)

# Movement, click-grid transformation, and click-only orchestration.
PILOT_GAME_IDS = (
    "ls20-9607627b",
    "ft09-0d8bbf25",
    "vc33-5430563c",
)

REVISION = WRAPPER_REVISION
FROZEN_IDS = FROZEN_PUBLIC_GAME_IDS
PILOT_IDS = PILOT_GAME_IDS
VALID_ACTION_NAMES = frozenset(
    {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}
)


class ArcAgi3DependencyError(RuntimeError):
    """The pinned optional ARC runtime is unavailable or has the wrong version."""


class ArcAgi3ProtocolError(ValueError):
    """A host operation or action violated the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ArcAgi3Action:
    name: str
    data: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "data": dict(self.data)}

    def encode(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ArcAgi3ReplayEvent:
    """One action or returned frame, including deterministic replay traffic."""

    index: int
    kind: str
    replay: bool
    counted: bool
    action: dict[str, Any]
    state: str
    levels_completed: int | None
    frame: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArcAgi3Config:
    """Pinned game selection and host-side episode limits."""

    game_id: str
    seed: int = 0
    corpus: str | os.PathLike[str] | None = None
    max_actions: int = DEFAULT_ACTION_BUDGET
    artifact_dir: str | os.PathLike[str] | None = None
    metadata_hashes: Mapping[str, str] | None = None


@dataclass
class _GameInstance:
    arcade: Any
    env: Any
    game_action: Any
    response: Any
    terminal: str = "ACTIVE"
    commands: list[str] = field(default_factory=list)
    events: list[ArcAgi3ReplayEvent] = field(default_factory=list)
    live_actions: int = 0
    replay_actions: int = 0


def preflight_arc3_corpus(
    game_ids: Sequence[str] = FROZEN_PUBLIC_GAME_IDS,
    *,
    corpus: str | os.PathLike[str] | None = None,
) -> Path:
    """Require a complete pre-existing local corpus; never download games."""

    configured = corpus if corpus is not None else os.environ.get("ARC3_CORPUS")
    path = Path(configured).expanduser() if configured else DEFAULT_ARC3_CORPUS
    if not path.is_dir():
        if configured:
            raise RuntimeError(f"ARC3_CORPUS is not a directory: {path}")
        raise RuntimeError(
            "ARC-AGI-3 public games are not installed. Run `make "
            "eval-arc3-setup`, or set ARC3_CORPUS to a pre-existing "
            f"environment_files directory (checked {path})"
        )
    available = _corpus_game_ids(path)
    missing = [game_id for game_id in game_ids if game_id not in available]
    if missing:
        raise RuntimeError(
            f"ARC3_CORPUS is missing {len(missing)} frozen game(s): {', '.join(missing)}"
        )
    return path.resolve()


def download_arc3_corpus(
    game_ids: Sequence[str] = FROZEN_PUBLIC_GAME_IDS,
    *,
    corpus: str | os.PathLike[str] | None = None,
) -> Path:
    """Explicitly download the public games into the local offline cache."""

    unknown = sorted(set(game_ids) - set(FROZEN_PUBLIC_GAME_IDS))
    if unknown:
        raise ValueError(f"cannot download unpinned ARC game(s): {', '.join(unknown)}")
    target = Path(corpus).expanduser() if corpus is not None else DEFAULT_ARC3_CORPUS
    arc_agi, _arcengine = _require_pinned_dependencies()
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.NORMAL,
        environments_dir=str(target),
        recordings_dir=str(target.parent / "recordings"),
    )
    available = {str(info.game_id) for info in arcade.get_environments()}
    unavailable = [game_id for game_id in game_ids if game_id not in available]
    if unavailable:
        raise RuntimeError(
            "ARC public registry is missing pinned game(s): " + ", ".join(unavailable)
        )
    for game_id in game_ids:
        environment = arcade.make(game_id, seed=0, save_recording=False)
        if environment is None:
            raise RuntimeError(f"failed to download ARC-AGI-3 game {game_id}")
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    return preflight_arc3_corpus(game_ids, corpus=target)


def _corpus_game_ids(corpus: Path) -> set[str]:
    game_ids: set[str] = set()
    for path in corpus.rglob("metadata.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("game_id"), str):
            game_ids.add(value["game_id"])
        if len(path.parent.name) == 8 and len(path.parent.parent.name) == 4:
            game_ids.add(f"{path.parent.parent.name}-{path.parent.name}")
    return game_ids


def corpus_metadata_hashes(
    corpus: Path,
    game_ids: Sequence[str],
) -> dict[str, str]:
    """Hash canonical metadata for the selected full suffixed game IDs."""
    wanted = set(game_ids)
    hashes: dict[str, str] = {}
    for path in corpus.rglob("metadata.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        game_id = value.get("game_id")
        if not isinstance(game_id, str) and len(path.parent.name) == 8:
            game_id = f"{path.parent.parent.name}-{path.parent.name}"
        if game_id in wanted:
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            hashes[game_id] = hashlib.sha256(canonical).hexdigest()
    missing = wanted - set(hashes)
    if missing:
        raise RuntimeError(
            "ARC3_CORPUS metadata could not be hashed for: " + ", ".join(sorted(missing))
        )
    return hashes


def validate_action(value: ArcAgi3Action | Mapping[str, Any] | str) -> ArcAgi3Action:
    """Normalize one action and enforce ACTION6's exact coordinate shape."""

    if isinstance(value, ArcAgi3Action):
        action = value
    elif isinstance(value, str):
        action = ArcAgi3Action(value.upper())
    elif isinstance(value, Mapping):
        unknown = set(value) - {"name", "data"}
        if unknown:
            raise ArcAgi3ProtocolError(f"unknown action field(s): {', '.join(sorted(unknown))}")
        name, data = value.get("name"), value.get("data", {})
        if not isinstance(name, str) or not isinstance(data, Mapping):
            raise ArcAgi3ProtocolError("actions require string `name` and object `data`")
        action = ArcAgi3Action(name.upper(), dict(data))
    else:
        raise ArcAgi3ProtocolError("each action must be a name or an action object")

    name, data = action.name.upper(), dict(action.data)
    if name not in VALID_ACTION_NAMES:
        raise ArcAgi3ProtocolError(f"invalid ARC-AGI-3 action {action.name!r}")
    if name == "ACTION6":
        if set(data) != {"x", "y"}:
            raise ArcAgi3ProtocolError("ACTION6 requires exactly integer `x` and `y`")
        for key in ("x", "y"):
            coordinate = data[key]
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, int)
                or not GRID_MIN <= coordinate <= GRID_MAX
            ):
                raise ArcAgi3ProtocolError(
                    f"ACTION6 {key} must be an integer in {GRID_MIN}..{GRID_MAX}"
                )
    elif data:
        raise ArcAgi3ProtocolError(f"{name} requires an empty `data` object")
    return ArcAgi3Action(name, data)


class ArcAgi3Adapter(EnvironmentAdapter):
    """Host-owned ARC Arcade with one private deterministic game per graph agent."""

    kind = "arc_agi_3"
    operations = ("observe", "status", "act")

    def __init__(
        self,
        game_id: str | ArcAgi3Config,
        *,
        seed: int = 0,
        corpus: str | os.PathLike[str] | None = None,
        max_actions: int = DEFAULT_ACTION_BUDGET,
        artifact_dir: str | os.PathLike[str] | None = None,
        metadata_hashes: Mapping[str, str] | None = None,
        arcade_factory: Callable[..., Any] | None = None,
    ) -> None:
        if isinstance(game_id, ArcAgi3Config):
            config = game_id
            game_id, seed, corpus, max_actions, artifact_dir, metadata_hashes = (
                config.game_id,
                config.seed,
                config.corpus,
                config.max_actions,
                config.artifact_dir,
                config.metadata_hashes,
            )
        if game_id not in FROZEN_PUBLIC_GAME_IDS:
            raise ValueError(f"{game_id!r} is not in the frozen ARC-AGI-3 public manifest")
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions <= 0:
            raise ValueError("max_actions must be a positive integer")
        self.game_id = game_id
        self.seed = seed
        self.max_actions = max_actions
        self.corpus = preflight_arc3_corpus((game_id,), corpus=corpus)
        actual_hashes = corpus_metadata_hashes(self.corpus, (game_id,))
        if metadata_hashes is not None and dict(metadata_hashes) != actual_hashes:
            raise RuntimeError(f"ARC3_CORPUS metadata hash mismatch for pinned game {game_id}")
        self.metadata_hashes = actual_hashes
        self.artifact_dir = (
            Path(artifact_dir).expanduser().resolve() if artifact_dir is not None else None
        )
        self.arcade_factory = arcade_factory
        self._instances: dict[str, _GameInstance] = {}

    def start(self, instance_id: str) -> EnvironmentResult:
        instance = self._open_instance()
        self._instances[instance_id] = instance
        return self._result(instance, actions_remaining=self.max_actions)

    def fork(
        self,
        instance_id: str,
        *,
        parent_instance_id: str,
        command_prefix: tuple[str, ...],
    ) -> EnvironmentResult:
        if parent_instance_id not in self._instances:
            raise RuntimeError(f"unknown immediate parent instance {parent_instance_id}")
        instance = self._open_instance()
        for encoded in command_prefix:
            action = validate_action(json.loads(encoded))
            response = self._apply(instance, action)
            if response is None:
                raise RuntimeError(f"replay failed on {action.name} for {self.game_id}")
            instance.response = response
            instance.commands.append(encoded)
            instance.replay_actions += 1
            self._record(instance, action, response, replay=True)
        instance.terminal = "WIN" if _state_of(instance.response) == "WIN" else "ACTIVE"
        self._instances[instance_id] = instance
        return self._result(instance)

    def execute(
        self,
        instance_id: str,
        operation: str,
        payload: dict[str, Any],
        *,
        max_actions: int,
    ) -> EnvironmentResult:
        instance = self._instance(instance_id)
        if operation in {"observe", "status"}:
            if payload:
                raise ArcAgi3ProtocolError(f"{operation} takes no payload")
            return self._result(instance, actions_remaining=max_actions)
        if operation != "act" or set(payload) != {"actions"}:
            raise ArcAgi3ProtocolError("only observe, status, and act are available")
        actions = payload["actions"]
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise ArcAgi3ProtocolError("act requires a sequence of actions")
        if not 1 <= len(actions) <= MAX_BATCH_ACTIONS:
            raise ArcAgi3ProtocolError(
                f"act requires 1..{MAX_BATCH_ACTIONS} actions, got {len(actions)}"
            )
        if len(actions) > max_actions:
            raise ArcAgi3ProtocolError(
                f"act requested {len(actions)} actions with only {max_actions} remaining"
            )
        normalized = [validate_action(action) for action in actions]
        if instance.terminal != "ACTIVE":
            if normalized[0].name != "RESET":
                return self._result(instance, actions_remaining=max_actions)
            normalized = normalized[:1]
            instance.terminal = "ACTIVE"

        before = getattr(instance.response, "levels_completed", None)
        accepted: list[str] = []
        for action in normalized:
            try:
                self._validate_available(instance, action)
            except ArcAgi3ProtocolError:
                # A prior accepted action changed the legal action space. Stop
                # at that settled state instead of mutating past invalid input.
                if accepted:
                    break
                raise
            response = self._apply(instance, action)
            if response is None:
                raise RuntimeError(f"{action.name} failed for {self.game_id}")
            encoded = action.encode()
            accepted.append(encoded)
            instance.commands.append(encoded)
            instance.response = response
            instance.live_actions += 1
            self._record(instance, action, response, replay=False)

            state = _state_of(response)
            if state == "WIN":
                instance.terminal = "WIN"
                break
            if state == "GAME_OVER":
                instance.terminal = "GAME_OVER"
                break
            if getattr(response, "levels_completed", None) != before:
                break
            if len(accepted) >= max_actions:
                instance.terminal = "ACTION_CAP"
                break
        if len(accepted) >= max_actions and instance.terminal == "ACTIVE":
            instance.terminal = "ACTION_CAP"
        return self._result(
            instance,
            accepted=tuple(accepted),
            actions_remaining=max(0, max_actions - len(accepted)),
        )

    def primary_metrics(self, instance_id: str) -> dict[str, Any]:
        instance = self._instance(instance_id)
        scorecard = self._scorecard(instance)
        run = _scorecard_run(scorecard, self.game_id)
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "operation_mode": "OFFLINE",
            "terminal": instance.terminal,
            "state": _state_of(instance.response),
            "levels_completed": getattr(instance.response, "levels_completed", None),
            "native_score": _native_score(scorecard, self.game_id),
            "native_scorer": "arc_agi_scorecard_rhae",
            "live_actions": instance.live_actions,
            "replay_actions": sum(child.replay_actions for child in self._instances.values()),
            "actions": run.get("actions", instance.live_actions),
            "resets": run.get("resets", 0),
            "per_level_action_counts": run.get("level_actions", []),
            "frame_events": sum(event.kind == "frame" for event in instance.events),
            "action_events": sum(event.kind == "action" for event in instance.events),
        }

    def artifacts(self, instance_id: str) -> dict[str, str]:
        if self.artifact_dir is None:
            return {}
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        primary = self._instance(instance_id)
        replay_path = self.artifact_dir / "arc_replay.json"
        scorecard_path = self.artifact_dir / "arc_scorecard.json"
        replay = {
            "schema_version": 1,
            "kind": "arc",
            "game_id": self.game_id,
            "seed": self.seed,
            "wrapper_revision": WRAPPER_REVISION,
            "arc_agi_version": ARC_AGI_VERSION,
            "arcengine_version": ARCENGINE_VERSION,
            "corpus_metadata_hashes": self.metadata_hashes,
            "instances": {
                child_id: {
                    "primary": child_id == instance_id,
                    "live_actions": child.live_actions,
                    "replay_setup_actions": child.replay_actions,
                    "events": [event.to_dict() for event in child.events],
                }
                for child_id, child in self._instances.items()
            },
        }
        replay_path.write_text(
            json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        scorecard_path.write_text(
            json.dumps(self._scorecard(primary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "arc_replay": str(replay_path),
            "arc_scorecard": str(scorecard_path),
        }

    def close(self) -> None:
        for instance in self._instances.values():
            close = getattr(instance.env, "close", None)
            if callable(close):
                close()

    def _open_instance(self) -> _GameInstance:
        arcade, game_action = self._new_arcade()
        env = arcade.make(
            self.game_id,
            seed=self.seed,
            save_recording=False,
            include_frame_data=True,
        )
        if env is None:
            raise RuntimeError(f"Arcade.make failed for {self.game_id}")
        response = env.reset()
        if response is None:
            raise RuntimeError(f"initial RESET failed for {self.game_id}")
        instance = _GameInstance(
            arcade=arcade,
            env=env,
            game_action=game_action,
            response=response,
        )
        self._record(instance, ArcAgi3Action("RESET"), response, replay=False, counted=False)
        return instance

    def _new_arcade(self) -> tuple[Any, Any]:
        if self.arcade_factory is not None:
            try:
                return self.arcade_factory(self.corpus), None
            except TypeError:
                return self.arcade_factory(environments_dir=str(self.corpus)), None
        arc_agi, arcengine = _require_pinned_dependencies()
        return (
            arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.OFFLINE,
                environments_dir=str(self.corpus),
            ),
            arcengine.GameAction,
        )

    def _instance(self, instance_id: str) -> _GameInstance:
        try:
            return self._instances[instance_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown ARC-AGI-3 instance {instance_id}") from exc

    def _validate_available(self, instance: _GameInstance, action: ArcAgi3Action) -> None:
        if action.name == "RESET":
            return
        available = _available_actions(instance.response)
        if action.name not in available:
            raise ArcAgi3ProtocolError(
                f"{action.name} is unavailable; current actions are {available}"
            )

    def _apply(self, instance: _GameInstance, action: ArcAgi3Action) -> Any:
        if action.name == "RESET":
            return instance.env.reset()
        direct = getattr(instance.env, action.name.lower(), None)
        if callable(direct):
            if action.name == "ACTION6":
                return direct(action.data["x"], action.data["y"])
            return direct()
        sdk_action = (
            action.name
            if instance.game_action is None
            else getattr(instance.game_action, action.name)
        )
        return instance.env.step(sdk_action, dict(action.data) or None)

    def _record(
        self,
        instance: _GameInstance,
        action: ArcAgi3Action,
        response: Any,
        *,
        replay: bool,
        counted: bool = True,
    ) -> None:
        state = _state_of(response)
        levels = getattr(response, "levels_completed", None)
        instance.events.append(
            ArcAgi3ReplayEvent(
                len(instance.events), "action", replay, counted, action.to_dict(), state, levels
            )
        )
        for frame in getattr(response, "frame", None) or []:
            instance.events.append(
                ArcAgi3ReplayEvent(
                    len(instance.events),
                    "frame",
                    replay,
                    counted,
                    action.to_dict(),
                    state,
                    levels,
                    _plain(frame),
                )
            )

    def _result(
        self,
        instance: _GameInstance,
        *,
        accepted: tuple[str, ...] = (),
        actions_remaining: int | None = None,
    ) -> EnvironmentResult:
        value = _observation(instance, self.game_id)
        if actions_remaining is not None:
            value["actions_remaining"] = actions_remaining
            value["actions_taken"] = max(0, self.max_actions - actions_remaining)
        return EnvironmentResult(
            value=value,
            accepted_actions=accepted,
            observation_summary={
                "state": value["state"],
                "terminal": value["terminal"],
                "levels_completed": value["levels_completed"],
                "available_actions": value["available_actions"],
            },
        )

    def _scorecard(self, instance: _GameInstance) -> dict[str, Any]:
        value = instance.arcade.get_scorecard()
        return {} if value is None else _plain(value)


def _require_pinned_dependencies() -> tuple[Any, Any]:
    for distribution, expected in (
        ("arc-agi", ARC_AGI_VERSION),
        ("arcengine", ARCENGINE_VERSION),
    ):
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise ArcAgi3DependencyError(
                f"install `arc-agi=={ARC_AGI_VERSION}` and `arcengine=={ARCENGINE_VERSION}`"
            ) from exc
        if installed != expected:
            raise ArcAgi3DependencyError(
                f"ARC-AGI-3 requires {distribution}=={expected}, found {installed}"
            )
    return import_module("arc_agi"), import_module("arcengine")


def _state_of(response: Any) -> str:
    state = getattr(response, "state", None)
    return str(getattr(state, "name", state or "NOT_FINISHED"))


def _available_actions(response: Any) -> list[str]:
    result: list[str] = []
    for value in getattr(response, "available_actions", None) or []:
        name = getattr(value, "name", None)
        raw = getattr(value, "value", value)
        if isinstance(name, str):
            result.append(name)
        elif isinstance(raw, int) and 1 <= raw <= 7:
            result.append(f"ACTION{raw}")
        elif isinstance(raw, str):
            result.append(raw.upper())
    return result


def _plain(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _plain(model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _observation(instance: _GameInstance, game_id: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "state": _state_of(instance.response),
        "levels_completed": getattr(instance.response, "levels_completed", None),
        "win_levels": getattr(instance.response, "win_levels", None),
        "available_actions": _available_actions(instance.response),
        "frame": _plain(getattr(instance.response, "frame", None) or []),
        "terminal": instance.terminal,
    }


def _native_score(scorecard: Mapping[str, Any], game_id: str) -> float:
    for environment in scorecard.get("environments", []):
        if not isinstance(environment, Mapping):
            continue
        environment_id = str(environment.get("id", ""))
        if game_id.startswith(environment_id) or environment_id.startswith(game_id):
            score = environment.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return float(score)
    score = scorecard.get("score")
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0


def _scorecard_run(scorecard: Mapping[str, Any], game_id: str) -> dict[str, Any]:
    for environment in scorecard.get("environments", []):
        if not isinstance(environment, Mapping):
            continue
        environment_id = str(environment.get("id", ""))
        if not (game_id.startswith(environment_id) or environment_id.startswith(game_id)):
            continue
        runs = environment.get("runs", [])
        if not isinstance(runs, list):
            return {}
        candidates = [run for run in runs if isinstance(run, dict)]
        return max(
            candidates,
            key=lambda run: (
                int(run.get("levels_completed", 0) or 0),
                float(run.get("score", 0.0) or 0.0),
            ),
            default={},
        )
    return {}


ArcAgi3EnvironmentAdapter = ArcAgi3Adapter
ArcadeOfflineAdapter = ArcAgi3Adapter

__all__ = [
    "ARCENGINE_VERSION",
    "ARC_AGI_VERSION",
    "DEFAULT_ACTION_BUDGET",
    "DEFAULT_ARC3_CORPUS",
    "FROZEN_IDS",
    "FROZEN_PUBLIC_GAME_IDS",
    "MAX_BATCH_ACTIONS",
    "PILOT_GAME_IDS",
    "PILOT_IDS",
    "REVISION",
    "VALID_ACTION_NAMES",
    "WRAPPER_REVISION",
    "ArcAgi3Action",
    "ArcAgi3Adapter",
    "ArcAgi3Config",
    "ArcAgi3DependencyError",
    "ArcAgi3EnvironmentAdapter",
    "ArcAgi3ProtocolError",
    "ArcAgi3ReplayEvent",
    "ArcadeOfflineAdapter",
    "corpus_metadata_hashes",
    "download_arc3_corpus",
    "preflight_arc3_corpus",
    "validate_action",
]
