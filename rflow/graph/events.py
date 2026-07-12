"""Graph action events for minimal rflow.

The recursive ``Graph`` is the durable source of truth. Graph actions are the
state transitions applied to that graph and streamed to observers. Every action
subclasses :class:`Event`, so observers can dispatch with ``isinstance`` instead
of string tags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rflow.graph.graph import DoneOutput, ExecOutput, Graph, Node


class Event:
    """Base class for graph action events streamed to observers."""

    __slots__ = ()
    #: Discriminator tag (kept for wire/JSON); prefer ``isinstance`` in code.
    type: str
    #: Id of the graph this event belongs to.
    graph_id: str


@dataclass(frozen=True, slots=True)
class GraphCreated(Event):
    type: Literal["graph_created"]
    graph: Graph

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id


@dataclass(frozen=True, slots=True)
class AppendNode(Event):
    type: Literal["append_node"]
    agent_id: str
    node_type: str
    node: Node
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class ReplaceNode(Event):
    type: Literal["replace_node"]
    agent_id: str
    node_id: str
    node_type: str
    node: Node
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class RemoveNode(Event):
    type: Literal["remove_node"]
    agent_id: str
    node_id: str
    subtree: bool = False
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class AddChild(Event):
    type: Literal["add_child"]
    parent_agent_id: str
    child: Graph
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class RemoveChild(Event):
    type: Literal["remove_child"]
    parent_agent_id: str
    child_agent_id: str
    graph_id: str = ""


GraphAction = (
    GraphCreated | AppendNode | ReplaceNode | RemoveNode | AddChild | RemoveChild
)

StepUntil = (
    Literal["next", "idle", "done", "finished", "supervising", "error"]
    | Callable[[Event, Graph], bool]
)


def is_rest(event: Event) -> bool:
    """Whether an event leaves its agent at a resting node (clean output / done).

    ``error_output`` is *not* a rest point, so an ``idle`` step keeps advancing past
    an error until the agent recovers to a clean ``exec_output``.
    """
    return isinstance(event, AppendNode) and isinstance(
        event.node, (ExecOutput, DoneOutput)
    )


def is_node(event: Event, node_cls: type[Node] | tuple[type[Node], ...]) -> bool:
    """Whether ``event`` appends a node of the given subclass(es)."""
    return isinstance(event, AppendNode) and isinstance(event.node, node_cls)


__all__ = [
    "AddChild",
    "AppendNode",
    "Event",
    "GraphAction",
    "GraphCreated",
    "RemoveChild",
    "RemoveNode",
    "ReplaceNode",
    "StepUntil",
    "is_node",
    "is_rest",
]
