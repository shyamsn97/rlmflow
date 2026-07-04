"""Child-launch bookkeeping for recursive Flow runs.

This module is graph-adjacent but not a scheduler. It normalizes the result of a
``launch_subagents`` call into a small launch record and tracks the live future
that the parent coroutine awaits.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from rflow.graph import ChildHandle
from rflow.tools.builtins import _launch_id_from_names


@dataclass(slots=True)
class ChildLaunch:
    """Durable metadata for one ``launch_subagents`` batch."""

    launch_id: str
    agent_ids: list[str]
    positions: list[int]
    results: list[object]
    launch_specs: list[dict[str, Any]]
    launch_names: list[str]


@dataclass(slots=True)
class LaunchWaiter:
    """Live future for a parent coroutine waiting on child results."""

    parent_agent_id: str
    launch: ChildLaunch
    future: asyncio.Future[list[object]]


class SpawnChild(Protocol):
    def __call__(
        self,
        parent_agent_id: str,
        name: str,
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Any | None = None,
        *,
        strict_name: bool = False,
    ) -> ChildHandle | str: ...


def prepare_child_launch(
    *,
    parent_agent_id: str,
    specs: list[dict[str, Any]],
    launch_names: list[str],
    spawn_child: SpawnChild,
) -> ChildLaunch:
    """Spawn child graphs and return normalized launch metadata."""

    results: list[object] = [None] * len(specs)
    agent_ids: list[str] = []
    positions: list[int] = []
    launch_specs: list[dict[str, Any]] = []
    committed_names: list[str] = []

    for i, spec in enumerate(specs):
        name = str(spec.get("name", launch_names[i]))
        spawned = spawn_child(
            parent_agent_id,
            name,
            spec["query"],
            spec.get("inputs"),
            spec.get("model", "default"),
            spec.get("output_schema"),
            strict_name=True,
        )
        if isinstance(spawned, str):
            results[i] = spawned
            continue
        agent_ids.append(spawned.agent_id)
        positions.append(i)
        launch_specs.append(dict(spec))
        committed_names.append(name)

    return ChildLaunch(
        launch_id=_launch_id_from_names(committed_names, agent_ids),
        agent_ids=agent_ids,
        positions=positions,
        results=results,
        launch_specs=launch_specs,
        launch_names=committed_names,
    )


def complete_launch(
    waiter: LaunchWaiter,
    child_result: Callable[[str], object],
) -> list[object]:
    """Fill a waiter's result list in original spec order."""

    for pos, agent_id in zip(waiter.launch.positions, waiter.launch.agent_ids):
        waiter.launch.results[pos] = child_result(agent_id)
    return waiter.launch.results


__all__ = [
    "ChildLaunch",
    "LaunchWaiter",
    "SpawnChild",
    "complete_launch",
    "prepare_child_launch",
]
