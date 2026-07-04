"""Typed live notifications for graph changes.

These are not the source of truth. A :class:`~rflow.graph.graph.Graph` remains
the durable state; events are small hints for live UIs, recorders, and
controllers to re-read the changed part of the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class GraphNodeCommitted:
    """Normal execution committed a new node to one agent trajectory."""

    type: Literal["node_committed"]
    agent_id: str
    node_id: str
    node_type: str
    seq: int
    global_step: int | None


@dataclass(frozen=True, slots=True)
class GraphEdited:
    """An explicit graph edit changed existing graph state."""

    type: Literal["graph_edited"]
    affected_agents: tuple[str, ...]
    reason: str


GraphEvent = GraphNodeCommitted | GraphEdited


__all__ = ["GraphEdited", "GraphEvent", "GraphNodeCommitted"]
