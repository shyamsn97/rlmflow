"""Graph action events for minimal rflow.

The recursive ``Graph`` is the durable source of truth. Graph actions are the
state transitions applied to that graph and streamed to observers. Every action
subclasses :class:`Event`, so observers can dispatch with ``isinstance`` instead
of string tags.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

from rflow.minimal.graph.graph import (
    DoneOutput,
    ErrorOutput,
    Graph,
    Node,
    SupervisingOutput,
)


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


GraphAction = GraphCreated | AppendNode | ReplaceNode | RemoveNode | AddChild | RemoveChild

StepUntil = (
    Literal["event", "node", "done", "finished", "idle", "supervising", "error"]
    | Callable[[Event, Graph], bool]
    | int
)

#: Step-boundary alias -> the node subclass an ``AppendNode`` must carry to match.
_NODE_BOUNDARIES: dict[str, type[Node]] = {
    "done": DoneOutput,
    "finished": DoneOutput,
    "idle": DoneOutput,
    "supervising": SupervisingOutput,
    "error": ErrorOutput,
}


def reached(
    until: StepUntil,
    n: int | None,
    event: Event,
    events: list[Event],
    graph: Graph,
) -> bool:
    """Whether a ``Flow.step`` boundary has been reached given the events so far."""
    if isinstance(until, int):
        return len(events) >= until
    if callable(until):
        return bool(until(event, graph)) or (n is not None and len(events) >= n)
    if until == "event":
        return len(events) >= (n or 1)
    if until == "node":
        return sum(isinstance(e, AppendNode) for e in events) >= (n or 1)
    if until in _NODE_BOUNDARIES:
        return isinstance(event, AppendNode) and isinstance(event.node, _NODE_BOUNDARIES[until])
    raise ValueError(f"unknown step boundary: {until!r}")


class EventStream:
    """The live event channel + driver task for one Flow run.

    Owns the async queue that graph actions are emitted into and the task that
    drives the run. The Flow owns the graph and decides *what* to emit; this owns
    the plumbing of *delivering* events and shutting the task down cleanly.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self, coro: Coroutine[Any, Any, None]) -> None:
        self._task = asyncio.create_task(coro)

    def emit(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def next(self) -> Event | None:
        """Next event, or ``None`` once the run is done (re-raising task errors)."""
        if self._task is None:
            return None
        if self._task.done() and self._queue.empty():
            await self._task
            return None
        # Race the queue against the task: a run that errors (or returns) without
        # emitting a final event can't wedge us on an event that never arrives.
        get = asyncio.ensure_future(self._queue.get())
        await asyncio.wait({get, self._task}, return_when=asyncio.FIRST_COMPLETED)
        if get.done():
            return get.result()
        get.cancel()
        with suppress(asyncio.CancelledError):
            await get
        if not self._queue.empty():
            return self._queue.get_nowait()
        await self._task
        return None

    async def aclose(self) -> None:
        """Cancel the driver task and wait for it to unwind."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task


__all__ = [
    "AddChild",
    "AppendNode",
    "Event",
    "EventStream",
    "GraphAction",
    "GraphCreated",
    "RemoveChild",
    "RemoveNode",
    "ReplaceNode",
    "StepUntil",
    "reached",
]
