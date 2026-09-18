"""Runtime-neutral environment episode state.

Runtime bindings identify owners and launch boundaries. This module owns every
scientific invariant: private instances, prefix replay, shared budgets,
candidate snapshots, selection, metrics, and artifact finalization.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from benchmarks.eval.candidates import CandidateLedger, CandidateRecord
from benchmarks.eval.environments.base import EnvironmentAdapter, EnvironmentResult


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat()


@dataclass(frozen=True, slots=True)
class EpisodeOwner:
    id: str
    path: str
    instance_id: str
    parent_instance_id: str | None
    depth: int
    primary: bool
    created_at: float
    fork_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.id,
            "agent_path": self.path,
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
            "primary": self.primary,
            "probe": not self.primary,
            "depth": self.depth,
            "created_at": _timestamp(self.created_at),
            "fork_action_count": self.fork_action_count,
        }


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    owner_id: str
    owner_path: str
    exec_action_id: str
    instance_id: str
    operation: str
    started_at: float
    finished_at: float
    accepted_actions: tuple[str, ...]
    observation_summary: dict[str, Any]
    primary: bool
    error: str | None = None
    call_id: int = 0
    result: dict[str, Any] | None = None

    @property
    def accepted_action_count(self) -> int:
        return len(self.accepted_actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.owner_id,
            "agent_path": self.owner_path,
            "exec_action_id": self.exec_action_id,
            "instance_id": self.instance_id,
            "operation": self.operation,
            "started_at": _timestamp(self.started_at),
            "finished_at": _timestamp(self.finished_at),
            "duration_ms": round((self.finished_at - self.started_at) * 1_000, 3),
            "accepted_actions": list(self.accepted_actions),
            "accepted_action_count": self.accepted_action_count,
            "observation_summary": dict(self.observation_summary),
            "primary": self.primary,
            "probe": not self.primary,
            **({"error": self.error} if self.error else {}),
        }


class EpisodeCore:
    """One thread-safe environment episode shared by every runtime binding."""

    def __init__(
        self,
        adapter: EnvironmentAdapter,
        *,
        max_actions: int,
        episode_id: str,
        candidate_budget: int = 0,
    ) -> None:
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        self.adapter = adapter
        self.max_actions = max_actions
        self.episode_id = episode_id
        self.candidates = (
            CandidateLedger(max_evaluations=candidate_budget, episode_id=episode_id)
            if candidate_budget
            else None
        )
        self.owners: dict[str, EpisodeOwner] = {}
        self.prefixes: dict[str, tuple[str, ...]] = {}
        self.events: list[EpisodeEvent] = []
        self.initial: EnvironmentResult | None = None
        self._final_metrics: dict[str, Any] | None = None
        self._final_artifacts: dict[str, str] | None = None
        self._closed = False
        self._lock = threading.RLock()

    @property
    def accepted_action_count(self) -> int:
        return sum(event.accepted_action_count for event in self.events)

    @property
    def actions_remaining(self) -> int:
        return max(0, self.max_actions - self.accepted_action_count)

    @property
    def initial_observation(self) -> dict[str, Any]:
        if self.initial is None:
            raise RuntimeError("environment episode has not started")
        return dict(self.initial.value)

    @property
    def root_owner(self) -> EpisodeOwner:
        try:
            return next(owner for owner in self.owners.values() if owner.primary)
        except StopIteration as exc:
            raise RuntimeError("environment episode has no root owner") from exc

    @property
    def scored_instance_id(self) -> str:
        selected = self.candidates.selected if self.candidates is not None else None
        return selected.instance_id if selected is not None else self.root_owner.instance_id

    def start(self, owner_id: str, *, path: str, depth: int = 0) -> EpisodeOwner:
        with self._lock:
            if self.owners and owner_id not in self.owners:
                raise RuntimeError("an environment episode belongs to exactly one root")
            if owner_id in self.owners:
                return self.owners[owner_id]
            owner = self._owner(owner_id, path, None, depth, True, 0)
            self.initial = self.adapter.start(owner.instance_id)
            if self.initial.accepted_actions:
                raise ValueError("environment start must not consume model actions")
            self.owners[owner_id] = owner
            self.prefixes[owner_id] = ()
            return owner

    def prefix_at(self, owner_id: str, exec_action_id: str, call_id: int) -> tuple[str, ...]:
        prefix = list(self.prefixes.get(owner_id, ()))
        after_launch = [
            action
            for event in self.events
            if event.owner_id == owner_id
            and event.exec_action_id == exec_action_id
            and event.call_id >= call_id
            for action in event.accepted_actions
        ]
        if after_launch:
            del prefix[-len(after_launch) :]
        return tuple(prefix)

    def fork(
        self,
        owner_id: str,
        *,
        parent_owner_id: str,
        path: str,
        depth: int,
        exec_action_id: str,
        call_id: int = 0,
        prefix: tuple[str, ...] | None = None,
    ) -> EpisodeOwner:
        with self._lock:
            if owner_id in self.owners:
                return self.owners[owner_id]
            parent = self.owners[parent_owner_id]
            accepted_prefix = self.prefixes[parent_owner_id] if prefix is None else prefix
            owner = self._owner(
                owner_id,
                path,
                parent.instance_id,
                depth,
                False,
                len(accepted_prefix),
            )
            started = time.time()
            result = EnvironmentResult({})
            error = None
            try:
                result = self.adapter.fork(
                    owner.instance_id,
                    parent_instance_id=parent.instance_id,
                    command_prefix=accepted_prefix,
                )
                if result.accepted_actions:
                    raise ValueError("fork replay must not consume the shared action budget")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.events.append(
                    EpisodeEvent(
                        owner.id,
                        owner.path,
                        exec_action_id,
                        owner.instance_id,
                        "fork",
                        started,
                        time.time(),
                        (),
                        dict(result.observation_summary),
                        False,
                        error,
                        call_id,
                        dict(result.value),
                    )
                )
            self.owners[owner_id] = owner
            self.prefixes[owner_id] = accepted_prefix
            return owner

    def perform(
        self,
        owner_id: str,
        operation: str,
        payload: dict[str, Any] | None,
        *,
        exec_action_id: str,
        call_id: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            owner = self.owners[owner_id]
            started = time.time()
            result = EnvironmentResult({})
            error = None
            try:
                result = self.adapter.execute(
                    owner.instance_id,
                    operation,
                    dict(payload or {}),
                    max_actions=self.actions_remaining,
                )
                if len(result.accepted_actions) > self.actions_remaining:
                    raise RuntimeError("environment adapter exceeded the shared action budget")
            except Exception as exc:  # noqa: BLE001 - every tool failure is row telemetry
                error = f"{type(exc).__name__}: {exc}"
            event = EpisodeEvent(
                owner.id,
                owner.path,
                exec_action_id,
                owner.instance_id,
                operation,
                started,
                time.time(),
                tuple(result.accepted_actions),
                dict(result.observation_summary),
                owner.primary,
                error,
                call_id,
                dict(result.value),
            )
            self.events.append(event)
            if result.accepted_actions:
                self.prefixes[owner_id] = (
                    *self.prefixes[owner_id],
                    *result.accepted_actions,
                )
            if error:
                raise RuntimeError(error)
            return dict(result.value)

    def submit_candidate(
        self,
        owner_id: str,
        *,
        label: str,
        proposal: Any = None,
        parent_candidate_id: str | None = None,
    ) -> CandidateRecord:
        if self.candidates is None:
            raise RuntimeError("candidate search is not enabled")
        with self._lock:
            if self.candidates.evaluations_remaining < 1:
                raise RuntimeError("candidate evaluation budget exhausted")
            owner = self.owners[owner_id]
            index = self.candidates.max_evaluations - self.candidates.evaluations_remaining
            prefix = self.prefixes[owner_id]
            snapshot_id = self._instance_id(f"{owner_id}:candidate:{index}")
            result = self.adapter.fork(
                snapshot_id,
                parent_instance_id=owner.instance_id,
                command_prefix=prefix,
            )
            if result.accepted_actions:
                raise ValueError("candidate snapshot must not consume actions")
            score, public = self.adapter.candidate_metrics(snapshot_id)
            return self.candidates.submit(
                owner_id=owner.id,
                owner_path=owner.path,
                owner_depth=owner.depth,
                instance_id=snapshot_id,
                label=label,
                proposal=proposal,
                parent_candidate_id=parent_candidate_id,
                environment_prefix=prefix,
                state_material={"prefix": prefix, "metrics": public},
                score=score,
                public_summary=public,
                replay_artifacts={
                    "environment_replay": f"environment_replay.json#instance={snapshot_id}"
                },
            )

    def select_candidate(self, candidate_id: str, *, owner_id: str) -> CandidateRecord:
        if self.candidates is None:
            raise RuntimeError("candidate search is not enabled")
        return self.candidates.select(
            candidate_id,
            actor_is_root=self.owners[owner_id].primary,
        )

    def metrics(self, *, parity_contract: dict[str, Any]) -> dict[str, Any]:
        selected = self.candidates.selected if self.candidates is not None else None
        primary = (
            self._final_metrics
            if self._final_metrics is not None
            else self.adapter.primary_metrics(self.scored_instance_id)
        )
        return {
            "kind": self.adapter.kind,
            "parity_contract": parity_contract,
            "episode_id": self.episode_id,
            "action_budget": self.max_actions,
            "accepted_actions": self.accepted_action_count,
            "total_actions": self.accepted_action_count,
            "actions_remaining": self.actions_remaining,
            "primary_actions": sum(
                event.accepted_action_count for event in self.events if event.primary
            ),
            "probe_actions": sum(
                event.accepted_action_count for event in self.events if not event.primary
            ),
            "instances_forked": max(0, len(self.owners) - 1),
            "instances": [owner.to_dict() for owner in self.owners.values()],
            "events": [
                {"index": index, **event.to_dict()} for index, event in enumerate(self.events)
            ],
            "primary": dict(primary),
            "selection_compliance": (
                float(selected is not None) if self.candidates is not None else None
            ),
            **(
                {
                    "candidate_search": {
                        **self.candidates.to_dict(),
                        "selection_required": True,
                        "selection_made": selected is not None,
                    }
                }
                if self.candidates is not None
                else {}
            ),
        }

    def artifacts(self) -> dict[str, str]:
        if self._final_artifacts is not None:
            return dict(self._final_artifacts)
        return self.adapter.artifacts(self.scored_instance_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._final_metrics = self.adapter.primary_metrics(self.scored_instance_id)
                self._final_artifacts = self.adapter.artifacts(self.scored_instance_id)
            finally:
                self._closed = True
                self.adapter.close()

    def _owner(
        self,
        owner_id: str,
        path: str,
        parent_instance_id: str | None,
        depth: int,
        primary: bool,
        fork_action_count: int,
    ) -> EpisodeOwner:
        return EpisodeOwner(
            id=owner_id,
            path=path,
            instance_id=self._instance_id(owner_id),
            parent_instance_id=parent_instance_id,
            depth=depth,
            primary=primary,
            created_at=time.time(),
            fork_action_count=fork_action_count,
        )

    def _instance_id(self, source: str) -> str:
        digest = hashlib.sha256(f"{self.episode_id}\0{source}".encode()).hexdigest()[:16]
        return f"env_{digest}"


__all__ = ["EpisodeCore", "EpisodeEvent", "EpisodeOwner"]
