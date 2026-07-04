"""Graph action events for minimal rflow.

The recursive ``Graph`` is the durable source of truth. Graph actions are the
state transitions applied to that graph and streamed to observers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rflow.minimal.graph import Graph, Node


@dataclass(frozen=True, slots=True)
class GraphCreated:
    type: Literal["graph_created"]
    graph: Graph


@dataclass(frozen=True, slots=True)
class AppendNode:
    type: Literal["append_node"]
    agent_id: str
    node_type: str
    node: Node


@dataclass(frozen=True, slots=True)
class ReplaceNode:
    type: Literal["replace_node"]
    agent_id: str
    node_id: str
    node_type: str
    node: Node


@dataclass(frozen=True, slots=True)
class RemoveNode:
    type: Literal["remove_node"]
    agent_id: str
    node_id: str
    subtree: bool = False


@dataclass(frozen=True, slots=True)
class AddChild:
    type: Literal["add_child"]
    parent_agent_id: str
    child: Graph


@dataclass(frozen=True, slots=True)
class RemoveChild:
    type: Literal["remove_child"]
    parent_agent_id: str
    child_agent_id: str


GraphAction = GraphCreated | AppendNode | ReplaceNode | RemoveNode | AddChild | RemoveChild
Event = GraphAction


__all__ = [
    "AddChild",
    "AppendNode",
    "Event",
    "GraphAction",
    "GraphCreated",
    "RemoveChild",
    "RemoveNode",
    "ReplaceNode",
]
