"""Per-agent async queues for Flow scheduling.

``TaskQueue`` is deliberately graph-free: each opaque id has an event queue and
at most one running task. Scheduling is explicit: callers add the next task for an
id, or do not add one. That is enough to model ``until`` boundaries without
parking coroutine frames.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

Work = Callable[[], Awaitable[Any]]


class TaskQueue:
    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self.queued: dict[str, Work] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.notify = asyncio.Event()

    def queue(self, agent_id: str) -> asyncio.Queue:
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
        return self.queues[agent_id]

    def running(self, agent_id: str) -> bool:
        task = self.tasks.get(agent_id)
        return task is not None and not task.done()

    def add(self, agent_id: str, work: Work) -> None:
        """Schedule work for an id, or remember it if that id is still running."""
        self.queue(agent_id)
        if self.running(agent_id):
            self.queued[agent_id] = work
            return
        self.start(agent_id, work)

    def stop(self, agent_id: str) -> None:
        """Drop any queued follow-up work for an id."""
        self.queued.pop(agent_id, None)

    def start(self, agent_id: str, work: Work) -> asyncio.Task:
        task = self.tasks[agent_id] = asyncio.create_task(work())
        task.add_done_callback(lambda task, id=agent_id: self.done(id))
        self.notify.set()
        return task

    def done(self, agent_id: str) -> None:
        if agent_id in self.queued:
            self.start(agent_id, self.queued.pop(agent_id))
            return
        self.notify.set()

    def emit(self, agent_id: str, item: Any) -> None:
        self.queue(agent_id).put_nowait(item)
        self.notify.set()

    def has_events(self) -> bool:
        return any(not queue.empty() for queue in self.queues.values())

    async def changed(self) -> None:
        await self.notify.wait()
        self.notify.clear()

    async def stream(self):
        """Yield ``(agent_id, item)`` from all per-id queues as items arrive."""
        while True:
            drained = False
            for agent_id, queue in list(self.queues.items()):
                while not queue.empty():
                    yield agent_id, queue.get_nowait()
                    drained = True
            if drained:
                continue
            if self.finished:
                return
            self.notify.clear()
            if any(not queue.empty() for queue in self.queues.values()):
                continue
            await self.notify.wait()

    @property
    def finished(self) -> bool:
        return (
            all(task.done() for task in self.tasks.values())
            and not self.queued
            and all(queue.empty() for queue in self.queues.values())
        )

    async def aclose(self) -> None:
        self.queued.clear()
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def exception(self) -> BaseException | None:
        for task in self.tasks.values():
            if task.done() and not task.cancelled() and task.exception() is not None:
                return task.exception()
        return None


__all__ = ["TaskQueue"]
