"""Thin upstream-RLM ownership binding for the shared episode state."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.base import candidate_protocol_message
from benchmarks.eval.environments.environment import Environment
from benchmarks.eval.environments.episode import EpisodeCore
from benchmarks.eval.environments.registry import (
    action_budget,
    bind_tools,
    build_environment,
    candidate_budget,
)


class OfficialEnvironmentEpisode:
    """Map upstream recursive RLM instances onto runtime-neutral owners."""

    ROOT_OWNER = "official:root"

    def __init__(
        self,
        adapter: Environment,
        *,
        max_actions: int,
        episode_id: str,
        candidate_budget: int = 0,
        expected_observation_sha256: str | None = None,
    ) -> None:
        self.core = EpisodeCore(
            adapter,
            max_actions=max_actions,
            episode_id=episode_id,
            candidate_budget=candidate_budget,
            expected_observation_sha256=expected_observation_sha256,
        )
        self._rlm_owners: dict[int, str] = {}
        self._thread = threading.local()
        self._counter = 0

    @property
    def adapter(self) -> Environment:
        return self.core.adapter

    @property
    def candidates(self):
        return self.core.candidates

    @property
    def max_actions(self) -> int:
        return self.core.max_actions

    @property
    def events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.core.events]

    @property
    def accepted_action_count(self) -> int:
        return self.core.accepted_action_count

    @property
    def actions_remaining(self) -> int:
        return self.core.actions_remaining

    @property
    def initial_observation(self) -> dict[str, Any]:
        return self.core.initial_observation

    @property
    def current_owner_id(self) -> str:
        return str(getattr(self._thread, "owner_id", self.ROOT_OWNER))

    def start(self) -> dict[str, Any]:
        self.core.start(self.ROOT_OWNER, path="root", depth=0)
        return self.initial_observation

    def register_rlm(self, rlm: object, owner_id: str | None = None) -> None:
        selected = owner_id or self.current_owner_id
        if selected not in self.core.owners:
            raise RuntimeError(f"unknown official environment owner {selected!r}")
        self._rlm_owners[id(rlm)] = selected

    def owner_for_rlm(self, rlm: object) -> str:
        try:
            return self._rlm_owners[id(rlm)]
        except KeyError as exc:
            raise RuntimeError("official RLM has no environment owner") from exc

    @contextmanager
    def activate(self, owner_id: str) -> Iterator[None]:
        previous = getattr(self._thread, "owner_id", None)
        self._thread.owner_id = owner_id
        try:
            yield
        finally:
            if previous is None:
                self._thread.__dict__.pop("owner_id", None)
            else:
                self._thread.owner_id = previous

    def fork(self, parent_owner_id: str, prompt: str, depth: int) -> str:
        self._counter += 1
        digest = hashlib.sha256(
            f"{self.core.episode_id}\0{parent_owner_id}\0{self._counter}\0{prompt}".encode()
        ).hexdigest()[:12]
        owner_id = f"official:{digest}"
        parent = self.core.owners[parent_owner_id]
        self.core.fork(
            owner_id,
            parent_owner_id=parent_owner_id,
            path=f"{parent.path}/{self._counter}",
            depth=depth,
            exec_action_id=f"{parent_owner_id}:fork:{self._counter}",
        )
        return owner_id

    def perform(self, operation: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        owner_id = self.current_owner_id
        result = self.core.perform(
            owner_id,
            operation,
            dict(payload or {}),
            exec_action_id=f"{owner_id}:environment:{len(self.core.events)}",
        )
        if self.candidates is not None:
            result["_candidate_protocol"] = candidate_protocol_message(
                is_root=self.core.owners[owner_id].primary,
                selected=self.candidates.selected is not None,
            )
        return result

    def tools(self) -> dict[str, dict[str, Any]]:
        try:
            tools = bind_tools(
                self.adapter.kind,
                lambda operation, payload=None: self.perform(operation, payload),
                proxy=False,
            )
        except ValueError:
            if self.candidates is None:
                raise
            tools = {}
        if self.candidates is not None:
            tools.update(self._candidate_tools())
        return tools

    @contextmanager
    def instrument(self, rlm_type: type) -> Iterator[None]:
        original_init = rlm_type.__init__
        original_subcall = rlm_type._subcall
        episode = self

        def instrumented_init(rlm_self: object, *args: Any, **kwargs: Any) -> None:
            original_init(rlm_self, *args, **kwargs)
            episode.register_rlm(rlm_self)

        def instrumented_subcall(
            rlm_self: object,
            prompt: str,
            model: str | None = None,
        ) -> Any:
            next_depth = int(rlm_self.depth) + 1
            if next_depth >= int(rlm_self.max_depth):
                return original_subcall(rlm_self, prompt, model)
            child_owner = episode.fork(episode.owner_for_rlm(rlm_self), prompt, next_depth)
            with episode.activate(child_owner):
                return original_subcall(rlm_self, prompt, model)

        rlm_type.__init__ = instrumented_init
        rlm_type._subcall = instrumented_subcall
        try:
            yield
        finally:
            rlm_type.__init__ = original_init
            rlm_type._subcall = original_subcall

    def metrics(self) -> dict[str, Any]:
        return self.core.metrics()

    def artifacts(self) -> dict[str, str]:
        return self.core.artifacts()

    def close(self) -> None:
        self.core.close()

    def write_discovery_tree(self, path: str | Path) -> Path | None:
        if self.candidates is None:
            return None
        return self.candidates.write(path)

    def _candidate_tools(self) -> dict[str, dict[str, Any]]:
        assert self.candidates is not None

        def submit_candidate(
            label: str,
            proposal: Any = None,
            parent_candidate_id: str | None = None,
        ) -> dict[str, Any]:
            owner_id = self.current_owner_id
            record = self.core.submit_candidate(
                owner_id,
                label=label,
                proposal=proposal,
                parent_candidate_id=parent_candidate_id,
            )
            return {
                **record.to_dict(public=True),
                "_candidate_protocol": candidate_protocol_message(
                    is_root=self.core.owners[owner_id].primary,
                    selected=False,
                ),
            }

        def list_candidates() -> list[dict[str, Any]]:
            return self.candidates.public_records()

        def select_candidate(candidate_id: str) -> dict[str, Any]:
            record = self.core.select_candidate(
                candidate_id,
                owner_id=self.current_owner_id,
            )
            return {
                "selected_candidate_id": record.id,
                "label": record.label,
                "public_summary": dict(record.public_summary),
            }

        return {
            "submit_candidate": {
                "tool": submit_candidate,
                "description": "Evaluate and register the active branch as a named candidate.",
            },
            "list_candidates": {
                "tool": list_candidates,
                "description": "List public summaries for all evaluated candidates.",
            },
            "select_candidate": {
                "tool": select_candidate,
                "description": "Root only: choose the candidate used for native scoring.",
            },
        }


def open_official_environment(
    environment: Mapping[str, Any],
    *,
    artifact_dir: str | Path,
    episode_id: str,
) -> OfficialEnvironmentEpisode:
    built = build_environment(environment, artifact_dir=artifact_dir)
    episode = OfficialEnvironmentEpisode(
        built,
        max_actions=action_budget(built),
        episode_id=episode_id,
        candidate_budget=candidate_budget(built),
        expected_observation_sha256=(
            str(environment["observation_sha256"])
            if environment.get("observation_sha256")
            else None
        ),
    )
    episode.start()
    return episode

__all__ = ["OfficialEnvironmentEpisode", "open_official_environment"]
