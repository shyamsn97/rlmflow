"""Small async task queue used by Flow's live scheduler.

This module intentionally knows nothing about graphs, prompts, LLMs, REPLs, or
``launch_subagents``. It provides only the runtime primitive Flow needs:

* submit keyed async work;
* run submitted work with a global concurrency limit;
* let a running task park while it awaits something else, freeing its slot.

The graph-specific part of scheduling lives in ``Flow``. Flow decides which
agent ids to submit, how child launches are recorded, and when child futures
resolve. This queue just keeps async work moving without nested pools.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

_current_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rflow_scheduler_current_key", default=None
)


@dataclass(slots=True)
class WorkItem:
    """One keyed async task submitted to the scheduler."""

    key: str
    run: Callable[[], Awaitable[None]]


class AsyncTaskQueue:
    """A minimal work-conserving async task queue.

    ``park(...)`` is the important operation for recursive agents. When parent
    code does ``await launch_subagents(...)``, Flow will await the child future
    through ``queue.park(future)``. The parent coroutine stays alive, but the
    queue stops counting it against ``max_concurrency`` so child agents can run.
    """

    def __init__(self, *, max_concurrency: int = 8) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self._pending: deque[WorkItem] = deque()
        self._pending_keys: set[str] = set()
        self._active: dict[asyncio.Task[None], str] = {}
        self._parked: set[str] = set()
        self._wake: asyncio.Event | None = None

    @property
    def pending_keys(self) -> frozenset[str]:
        return frozenset(self._pending_keys)

    @property
    def active_keys(self) -> frozenset[str]:
        return frozenset(self._active.values())

    @property
    def parked_keys(self) -> frozenset[str]:
        return frozenset(self._parked)

    @property
    def running_keys(self) -> frozenset[str]:
        return frozenset(k for k in self._active.values() if k not in self._parked)

    @property
    def idle(self) -> bool:
        return not self._pending and not self._active

    def contains(self, key: str) -> bool:
        """Whether ``key`` is pending or currently active."""

        return key in self._pending_keys or key in self.active_keys

    def submit(self, key: str, run: Callable[[], Awaitable[None]]) -> bool:
        """Submit one keyed async task.

        Returns ``False`` if that key is already pending or active.
        """

        if self.contains(key):
            return False
        self._pending.append(WorkItem(key=key, run=run))
        self._pending_keys.add(key)
        self.wake()
        return True

    def submit_many(
        self, items: list[tuple[str, Callable[[], Awaitable[None]]]]
    ) -> int:
        """Submit many tasks and return the number accepted."""

        return sum(1 for key, run in items if self.submit(key, run))

    async def run_until_idle(self) -> None:
        """Run submitted work until no task is pending or active."""

        if self._wake is None:
            self._wake = asyncio.Event()
        try:
            while True:
                self._reap_done()
                self._start_ready()
                if self.idle:
                    return
                await self._wait_for_progress()
        finally:
            if self.idle:
                self._wake = None

    async def park(self, awaitable: Awaitable[T]) -> T:
        """Await ``awaitable`` without consuming a scheduler concurrency slot."""

        key = _current_key.get()
        if key is None:
            return await awaitable

        self._parked.add(key)
        self.wake()
        try:
            return await awaitable
        finally:
            self._parked.discard(key)
            self.wake()

    def wake(self) -> None:
        """Wake ``run_until_idle`` after new work/progress is available."""

        if self._wake is not None:
            self._wake.set()

    def _start_ready(self) -> None:
        slots = self.max_concurrency - len(self.running_keys)
        while slots > 0 and self._pending:
            item = self._pending.popleft()
            self._pending_keys.discard(item.key)
            task = asyncio.create_task(self._run(item))
            self._active[task] = item.key
            slots -= 1

    async def _run(self, item: WorkItem) -> None:
        token = _current_key.set(item.key)
        try:
            await item.run()
        finally:
            _current_key.reset(token)
            self.wake()

    async def _wait_for_progress(self) -> None:
        waitables: set[asyncio.Task[Any]] = set(self._active)
        wake_task: asyncio.Task[bool] | None = None
        if self._wake is not None:
            wake_task = asyncio.create_task(self._wake.wait())
            waitables.add(wake_task)
        if not waitables:
            return

        done, _pending = await asyncio.wait(
            waitables, return_when=asyncio.FIRST_COMPLETED
        )
        if wake_task is not None and wake_task in done and self._wake is not None:
            self._wake.clear()
        elif wake_task is not None:
            wake_task.cancel()

        self._reap_done(done)

    def _reap_done(self, candidates: set[asyncio.Task[Any]] | None = None) -> None:
        tasks = candidates if candidates is not None else set(self._active)
        for task in list(tasks):
            key = self._active.get(task)
            if key is None or not task.done():
                continue
            self._active.pop(task, None)
            self._parked.discard(key)
            task.result()


__all__ = ["AsyncTaskQueue", "WorkItem"]
