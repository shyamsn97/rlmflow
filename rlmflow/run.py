"""One ``Flow.run_streaming``: the turns in flight, the nodes that landed, who waits for a change."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow


class Run:
    """Exists only while a Flow drives a graph; ``flow.run`` is ``None`` otherwise.

    Turns live in ``flow.tasks``. Anything that changes the graph calls ``wake()``;
    ``wait_for(predicate)`` rechecks its predicate after every wake.
    """

    def __init__(self, flow: Flow, g: Graph) -> None:
        self.flow, self.g = flow, g
        self.loop = asyncio.get_running_loop()
        self._landed: list[Node] = []  # finished nodes not yet yielded
        self._changed = asyncio.Condition()
        self._version = 0  # bumped by every wake
        self._seen = 0  # the version when ``advance`` last ran
        self._notifying: set[asyncio.Task[None]] = set()
        self._toolkit_tools: list[str] = []

    async def __aenter__(self) -> Run:
        self._toolkit_tools = await self.flow._open_toolkits()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.flow._cancel(list(self.flow.tasks))
        for agent in self.g.agents():
            self.flow.deliver_results(self.g, agent)
        await self.flow._close_toolkits(self._toolkit_tools)

    # -- what the run loop asks ----------------------------------------------

    def advance(self) -> None:
        """Reap finished turns (a top-level failure raises), then start new ones."""
        self._seen = self._version
        tasks = self.flow.tasks
        for agent_id, task in list(tasks.items()):
            if task.done():
                del tasks[agent_id]
                if not task.cancelled() and (error := task.exception()) is not None:
                    raise error
        for agent in self.g.agents():
            if agent.id not in tasks and self.flow.ready_for_turn(self.g, agent):
                task = asyncio.ensure_future(self.flow._turn(self.g, agent))
                task.add_done_callback(lambda _task: self.wake())
                tasks[agent.id] = task

    @property
    def landed(self) -> bool:
        return bool(self._landed)

    @property
    def busy(self) -> bool:
        return bool(self.flow.tasks)

    def drain(self) -> Iterator[Step]:
        """The nodes that finished since the last call, skipping any since removed."""
        while self._landed:
            node = self._landed.pop(0)
            if node.id in self.g:
                yield Step(self.g, node.agent, node)

    async def wait_for_change(self) -> None:
        """Sleep until something wakes the run after the last ``advance``."""
        await self.wait_for(lambda: self._version != self._seen)

    # -- waking ----------------------------------------------------------------

    def land(self, node: Node) -> None:
        self._landed.append(node)
        self.wake()

    def wake(self) -> None:
        """The graph changed; safe from any thread."""
        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        if here is self.loop:
            self._notify()
        else:
            self.loop.call_soon_threadsafe(self._notify)

    def _notify(self) -> None:
        self._version += 1
        task = self.loop.create_task(self._broadcast())
        self._notifying.add(task)
        task.add_done_callback(self._notifying.discard)

    async def _broadcast(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        """Wait until ``predicate()`` holds, rechecking whenever the graph changes."""
        async with self._changed:
            await self._changed.wait_for(predicate)
