"""Minimal environment interface shared by every benchmark problem."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import JsonValue

from benchmarks.eval.reporting import EnvironmentReport, RuntimeMetrics
from benchmarks.eval.types import Prediction

ObservationT = TypeVar("ObservationT", bound="Observation")
MetricsT = TypeVar("MetricsT")


class Observation(ABC):
    """The complete public state delivered identically to every runner."""

    @abstractmethod
    def to_dict(self) -> dict[str, JsonValue]:
        """Return JSON-safe public state with no hidden scoring data."""


def canonical_observation_json(observation: Observation) -> str:
    """Serialize one observation deterministically for runtime delivery."""
    return json.dumps(
        observation.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_observation_bytes(observation: Observation) -> bytes:
    return canonical_observation_json(observation).encode("utf-8")


def observation_sha256(observation: Observation) -> str:
    return hashlib.sha256(canonical_observation_bytes(observation)).hexdigest()


class Environment(ABC, Generic[ObservationT, MetricsT]):
    """One complete scored benchmark problem."""

    id: str
    description: str

    @abstractmethod
    def start(self, instance_id: str) -> ObservationT:
        """Create fresh state and return the canonical initial observation."""

    def observe(self, instance_id: str) -> ObservationT:
        raise NotImplementedError(f"{type(self).__name__} is not interactive")

    def execute(
        self,
        instance_id: str,
        operation: str,
        payload: dict[str, JsonValue],
        *,
        max_actions: int,
    ) -> ObservationT:
        del instance_id, operation, payload, max_actions
        raise NotImplementedError(f"{type(self).__name__} is not interactive")

    def fork(
        self,
        instance_id: str,
        *,
        parent_instance_id: str,
        command_prefix: tuple[str, ...],
    ) -> ObservationT:
        del instance_id, parent_instance_id, command_prefix
        raise NotImplementedError(f"{type(self).__name__} does not support forks")

    def accepted_prefix(self, instance_id: str) -> tuple[str, ...]:
        del instance_id
        return ()

    @abstractmethod
    def metrics(self, instance_id: str) -> MetricsT:
        """Return environment-owned measurements for one scored instance."""

    @abstractmethod
    def report(
        self,
        prediction: Prediction,
        runtime: RuntimeMetrics,
    ) -> EnvironmentReport[MetricsT]:
        """Score a prediction and attach this environment's metrics."""


__all__ = [
    "Environment",
    "MetricsT",
    "Observation",
    "ObservationT",
    "canonical_observation_bytes",
    "canonical_observation_json",
    "observation_sha256",
]
