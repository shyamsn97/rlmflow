"""Scheduling: the step queue, its pools, and where a streamed run stops."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from rlmflow.graph.nodes import CODE, DONE, MESSAGE, WAITING, Node, Step, running_step


class Pool(ABC):
    """Where compute runs and how many compute calls may run at once."""

    @abstractmethod
    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one compute call under this backend's placement policy."""

    @abstractmethod
    def stream(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run one streaming compute call under this backend's policy."""

    def close(self) -> None:
        """Release whatever the backend holds."""


class ThreadPool(Pool):
    """Offload blocking compute to threads."""

    def __init__(self, workers: int | None = None) -> None:
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers else None
        self._async_slots = asyncio.Semaphore(workers) if workers else None

    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            functools.partial(fn, *args, **kwargs),
        )
        return await result if inspect.isawaitable(result) else result

    async def stream(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        if inspect.isasyncgenfunction(fn) or inspect.iscoroutinefunction(fn):
            if self._async_slots is None:
                async for item in self._iterate_async(fn, args, kwargs):
                    yield item
                return
            async with self._async_slots:
                async for item in self._iterate_async(fn, args, kwargs):
                    yield item
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stopped = threading.Event()

        def emit(kind: str, value: Any = None) -> None:
            if stopped.is_set():
                return
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
            except RuntimeError:
                stopped.set()

        def produce() -> None:
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result) or hasattr(result, "__aiter__"):
                    raise TypeError("synchronous stream function returned an async stream")
                for item in result:
                    if stopped.is_set():
                        break
                    emit("item", item)
            except BaseException as exc:  # noqa: BLE001 - cross-thread propagation
                emit("error", exc)
            finally:
                emit("end")

        future = loop.run_in_executor(self._executor, produce)
        try:
            while True:
                kind, value = await queue.get()
                if kind == "item":
                    yield value
                elif kind == "error":
                    raise value
                else:
                    break
        finally:
            stopped.set()
            if future.done():
                await asyncio.gather(future, return_exceptions=True)

    @staticmethod
    async def _iterate_async(
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if hasattr(result, "__aiter__"):
            async for item in result:
                yield item
            return
        for item in result:
            yield item

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Run one compute call at a time in the event-loop thread."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async with self._lock:
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

    async def stream(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async with self._lock:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                async for item in result:
                    yield item
                return
            for item in result:
                yield item


@dataclass(slots=True)
class StepResult:
    """One finished step: who stepped, from where, and what landed."""

    step: Step
    created: Node | None
    error: BaseException | None = None

    @property
    def agent(self) -> Node:
        return self.step.agent


class TaskQueue:
    """Run one step per agent at a time and report finished steps in order."""

    def __init__(self) -> None:
        self.running: dict[str, tuple[Step, asyncio.Task[None]]] = {}
        self.done: asyncio.Queue[StepResult | None] = asyncio.Queue()
        self.changed = asyncio.Condition()
        self._loop = asyncio.get_running_loop()

    def __bool__(self) -> bool:
        return bool(self.running)

    def __contains__(self, agent: Node) -> bool:
        return agent.id in self.running

    def submit(self, step: Step, fn: Callable[[Step], Awaitable[StepResult]]) -> None:
        """Run ``fn(step)`` unless this agent already has a step in flight."""
        if step.agent.id not in self.running:
            self.running[step.agent.id] = (step, asyncio.create_task(self._run(step, fn)))

    async def _run(self, step: Step, fn: Callable[[Step], Awaitable[StepResult]]) -> None:
        with running_step(step):
            result = await fn(step)
        if not isinstance(result, StepResult):
            raise TypeError(f"task returned {type(result).__name__}, expected StepResult")
        self.done.put_nowait(result)

    def wake(self) -> None:
        """Ask the driver to look for new work; safe from any thread."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self.done.put_nowait(None)
        else:
            self._loop.call_soon_threadsafe(self.done.put_nowait, None)

    async def next(self) -> StepResult | None:
        """The next finished step, or ``None`` when only woken."""
        result = await self.done.get()
        if result is not None:
            self.running.pop(result.agent.id, None)
        return result

    async def notify(self) -> None:
        """Wake waiters after the driver has processed a change."""
        async with self.changed:
            self.changed.notify_all()

    async def join(self, predicate: Callable[[], bool]) -> None:
        """Wait until ``predicate()`` holds after some processed change."""
        async with self.changed:
            await self.changed.wait_for(predicate)

    async def cancel(self, agents: Iterable[Node] | None = None) -> None:
        """Cancel the steps of these agents, or of every agent when omitted."""
        keys = set(self.running) if agents is None else {agent.id for agent in agents}
        tasks = [task for key, (_step, task) in self.running.items() if key in keys]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for key in keys:
            self.running.pop(key, None)


Boundary: TypeAlias = Callable[[Step], bool]
StepUntil: TypeAlias = Literal["next", "idle", "done", "finished"] | Boundary

_FULL_RUN = frozenset({"done", "finished"})
_BOUNDARIES: dict[str, Boundary] = {
    # Messages are written by senders, not steps.
    "next": lambda step: step.node.type != MESSAGE,
    "idle": lambda step: step.node.type in (CODE, WAITING, DONE),
}


def resolve_boundary(until: StepUntil) -> Boundary | None:
    """Resolve a built-in or callable boundary."""
    if until in _FULL_RUN:
        return None
    if callable(until):
        return until
    try:
        return _BOUNDARIES[until]
    except (KeyError, TypeError):
        raise ValueError(f"unknown until boundary: {until!r}") from None


__all__ = [
    "Boundary",
    "Pool",
    "SequentialPool",
    "StepResult",
    "StepUntil",
    "TaskQueue",
    "ThreadPool",
    "resolve_boundary",
]
