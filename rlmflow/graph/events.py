"""Graph action events for minimal rlmflow.

The recursive ``Graph`` is the durable source of truth. Graph actions are the
state transitions applied to that graph and streamed to observers. Every action
subclasses :class:`Event`, so observers can dispatch with ``isinstance`` instead
of string tags.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from rlmflow.graph.graph import DoneOutput, ExecOutput, Graph, Node


class Event:
    """Base class for graph action events streamed to observers.

    Every event exposes the same node view: ``node`` (the node the event is
    about) plus ``node_id`` / ``node_type`` derived from it, so they can never
    drift. Events not about a node leave ``node`` as ``None``.
    """

    __slots__ = ()
    #: Discriminator tag (kept for wire/JSON); prefer ``isinstance`` in code.
    type: str
    #: Id of the graph this event belongs to.
    graph_id: str
    #: The node this event is about; a field on node events, a property on
    #: graph/child events, and ``None`` when the event is not about a node.
    node: Node | None

    @property
    def node_id(self) -> str | None:
        return self.node.id if self.node is not None else None

    @property
    def node_type(self) -> str | None:
        return self.node.type if self.node is not None else None


@dataclass(frozen=True, slots=True)
class GraphCreated(Event):
    type: Literal["graph_created"]
    graph: Graph

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id

    @property
    def node(self) -> Node | None:
        return self.graph.nodes[0] if self.graph.nodes else None


@dataclass(frozen=True, slots=True)
class AppendNode(Event):
    type: Literal["append_node"]
    agent_id: str
    node: Node
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class InsertNode(Event):
    type: Literal["insert_node"]
    agent_id: str
    node: Node
    index: int
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class ReplaceNode(Event):
    type: Literal["replace_node"]
    agent_id: str
    node: Node  # the new node
    replaced_node: Node  # the node it replaces
    graph_id: str = ""

    @property
    def replaced_node_id(self) -> str:
        return self.replaced_node.id


@dataclass(frozen=True, slots=True)
class RemoveNode(Event):
    type: Literal["remove_node"]
    agent_id: str
    node: Node  # the node being removed
    subtree: bool = False
    graph_id: str = ""


@dataclass(frozen=True, slots=True)
class AddChild(Event):
    type: Literal["add_child"]
    parent_agent_id: str
    child: Graph
    graph_id: str = ""

    @property
    def node(self) -> Node | None:
        return self.child.nodes[0] if self.child.nodes else None


@dataclass(frozen=True, slots=True)
class RemoveChild(Event):
    type: Literal["remove_child"]
    parent_agent_id: str
    child: Graph
    graph_id: str = ""

    @property
    def child_agent_id(self) -> str:
        return self.child.agent_id

    @property
    def node(self) -> Node | None:
        return self.child.nodes[0] if self.child.nodes else None


GraphAction = (
    GraphCreated
    | AppendNode
    | InsertNode
    | ReplaceNode
    | RemoveNode
    | AddChild
    | RemoveChild
)

StepUntil = (
    Literal["next", "idle", "done", "finished", "supervising", "error"]
    | Callable[[Event, Graph], bool]
)


def is_append(event: Event) -> bool:
    """Whether ``event`` appends a fresh node to an agent's trajectory."""
    return isinstance(event, AppendNode)


def is_idle(event: Event) -> bool:
    """Whether an event leaves its agent at a resting node (clean output / done).

    ``error_output`` is *not* a rest point, so an ``idle`` step keeps advancing past
    an error until the agent recovers to a clean ``exec_output``.
    """
    return isinstance(event, AppendNode) and isinstance(
        event.node, (ExecOutput, DoneOutput)
    )


__all__ = [
    "AddChild",
    "AppendNode",
    "Event",
    "GraphAction",
    "GraphCreated",
    "InsertNode",
    "RemoveChild",
    "RemoveNode",
    "ReplaceNode",
    "StepUntil",
    "is_append",
    "is_idle",
]
