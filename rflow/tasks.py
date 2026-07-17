"""Async task scheduling for Flow.

``TaskQueue`` is deliberately graph-free: it tracks at most one running task per
opaque id, an optional queued follow-up per id, and a single stream mailbox of
emitted ``StreamItem``s. Scheduling is explicit: callers add the next task for an
id, or do not add one. That is enough to model ``until`` boundaries without
parking coroutine frames.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

Work = Callable[[], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class StreamItem:
    agent_id: str
    item: Any


class TaskQueue:
    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self.queued: dict[str, Work] = {}
        self.stream: deque[StreamItem] = deque()
        self.notify: asyncio.Event | None = None
        self.notify_loop: asyncio.AbstractEventLoop | None = None

    def event(self) -> asyncio.Event:
        loop = asyncio.get_running_loop()
        if self.notify is None or self.notify_loop is not loop:
            self.notify = asyncio.Event()
            self.notify_loop = loop
        return self.notify

    def wake(self) -> None:
        if self.notify is not None:
            self.notify.set()

    def running(self, agent_id: str) -> bool:
        task = self.tasks.get(agent_id)
        return task is not None and not task.done()

    def add(self, agent_id: str, work: Work) -> None:
        """Schedule work for an id, or remember it if that id is still running."""
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
        self.wake()
        return task

    def done(self, agent_id: str) -> None:
        if agent_id in self.queued:
            self.start(agent_id, self.queued.pop(agent_id))
            return
        self.wake()

    def emit(self, agent_id: str, item: Any) -> None:
        self.stream.append(StreamItem(agent_id=agent_id, item=item))
        self.wake()

    def has_events(self) -> bool:
        return bool(self.stream)

    async def changed(self) -> None:
        notify = self.event()
        await notify.wait()
        notify.clear()

    async def next(self) -> StreamItem | None:
        """Return the next emitted item, or ``None`` when the frontier settles."""
        while True:
            if self.stream:
                return self.stream.popleft()
            if self.settled:
                return None
            await self.changed()

    async def items(self):
        """Yield emitted stream items until the current frontier settles."""
        while True:
            item = await self.next()
            if item is None:
                break
            yield item

    @property
    def settled(self) -> bool:
        return (
            all(task.done() for task in self.tasks.values())
            and not self.queued
            and not self.stream
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


__all__ = ["StreamItem", "TaskQueue"]
