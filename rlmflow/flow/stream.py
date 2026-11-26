"""Internal state for one live ``Flow.stream`` call."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from typing import Any

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node


class Stream:
    """One graph being streamed: an internal record ``Flow`` reads and writes.

    A Flow keeps one per live graph, by graph id. It holds no Flow and calls
    nothing outside itself; starting steps, asking hooks and opening toolkits
    are Flow's. Anything that changes the graph calls ``wake()``, and
    ``wait_for(predicate)`` rechecks its predicate after every wake.
    """

    def __init__(self, g: Graph, hooks: list[Any]) -> None:
        self.g = g
        self.hooks = hooks
        self.loop = asyncio.get_running_loop()
        self.started_at = time.monotonic()
        #: The task running each run's step, by task node id.
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        #: Tools from the toolkits this stream opened, by name.
        self.tools: dict[str, Any] = {}
        #: The toolkit copies this stream opened, in opening order.
        self.toolkits: list[Any] = []
        #: The hooks whose ``start`` ran, in order.
        self.started_hooks: list[Any] = []
        self.timers: list[asyncio.TimerHandle] = []
        #: Timer callbacks that came due and haven't run yet.
        self.due: list[Callable[[], Any]] = []
        self._landed: list[Node] = []  # finished nodes not yet yielded
        self._changed = asyncio.Condition()
        self._version = 0  # bumped by every wake
        self._seen = 0  # the version when the loop last looked
        self._notifying: set[asyncio.Task[None]] = set()

    @property
    def landed(self) -> bool:
        return bool(self._landed)

    def land(self, node: Node) -> None:
        """Queue a finished node to be yielded, and wake."""
        self._landed.append(node)
        self.wake()

    def drain(self) -> Iterator[Node]:
        """The nodes that finished since the last call, skipping any since removed."""
        while self._landed:
            node = self._landed.pop(0)
            if node.id in self.g:
                yield node

    def mark_seen(self) -> None:
        """The loop is looking now; ``wait_for_change`` waits for a wake after this."""
        self._seen = self._version

    def reap(self) -> None:
        """Forget finished steps; a top-level failure raises."""
        for run_id, task in list(self.tasks.items()):
            if task.done():
                del self.tasks[run_id]
                if not task.cancelled() and (error := task.exception()) is not None:
                    raise error

    async def cancel(self, run_ids: list[str]) -> None:
        """Cancel the steps of ``run_ids`` that are running and wait for them to stop."""
        tasks = [self.tasks.pop(i) for i in dict.fromkeys(run_ids) if i in self.tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def call_later(self, delay: float, fn: Callable[[], Any]) -> None:
        """Queue ``fn`` as due after ``delay`` seconds, and wake."""

        def due() -> None:
            self.due.append(fn)
            self.wake()

        self.timers.append(self.loop.call_later(delay, due))

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    # -- waking ----------------------------------------------------------------

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

    async def wait_for_change(self) -> None:
        """Sleep until something wakes the stream after the loop last looked."""
        await self.wait_for(lambda: self._version != self._seen)

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        """Wait until ``predicate()`` holds, rechecking whenever the graph changes."""
        async with self._changed:
            await self._changed.wait_for(predicate)

    def _notify(self) -> None:
        self._version += 1
        task = self.loop.create_task(self._broadcast())
        self._notifying.add(task)
        task.add_done_callback(self._notifying.discard)

    async def _broadcast(self) -> None:
        async with self._changed:
            self._changed.notify_all()
