"""Concurrency primitives used by Flow."""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from rlmflow.graph.nodes import Node

T = TypeVar("T")


class _ResultsDone:
    pass


_RESULTS_DONE = _ResultsDone()


class CurrentResults:
    """Results published by one running task."""

    def __init__(self, stop: Callable[[Node], bool] | None = None) -> None:
        self.queue: asyncio.Queue[Node | _ResultsDone] = asyncio.Queue()
        self.stop = stop
        self.stopped = False
        self.task: asyncio.Task[Any] | None = None

    def publish(self, result: Node) -> None:
        self.queue.put_nowait(result)
        if self.stop is not None and self.stop(result):
            self.stopped = True

    def bind(self, task: asyncio.Task[Any]) -> None:
        if self.task is not None:
            raise RuntimeError("results already bound to a task")
        self.task = task
        task.add_done_callback(self._finish)

    def _finish(self, _task: asyncio.Task[Any]) -> None:
        self.queue.put_nowait(_RESULTS_DONE)

    async def __aiter__(self) -> AsyncIterator[Node]:
        while True:
            result = await self.queue.get()
            if isinstance(result, _ResultsDone):
                return
            yield result

    async def wait(self) -> Any:
        if self.task is None:
            raise RuntimeError("results are not bound to a task")
        return await self.task

    async def close(self) -> None:
        if self.task is None:
            return
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class Pool(ABC):
    """Run blocking leaf calls under a configurable policy."""

    @abstractmethod
    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke a callable and return its result."""

    def close(self) -> None:
        """Release resources held by the pool."""


class ThreadPool(Pool):
    """Offload synchronous calls to a bounded thread pool."""

    def __init__(self, workers: int | None = None) -> None:
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers is not None else None

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))
        return await result if inspect.isawaitable(result) else result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Run one call at a time without worker threads."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result


def _first_error(group: BaseExceptionGroup) -> BaseException:
    error: BaseException = group.exceptions[0]
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


class TaskQueue:
    """Track top-level runs and execute child batches concurrently."""

    def __init__(self) -> None:
        self.tasks: set[asyncio.Task[Any]] = set()
        self.runs: set[str] = set()

    def track(self, coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def submit(
        self,
        coroutine: Coroutine[Any, Any, Any],
        results: CurrentResults,
        *,
        key: str,
    ) -> CurrentResults:
        if key in self.runs:
            coroutine.close()
            raise RuntimeError("root is already streaming")
        self.runs.add(key)
        task = self.track(coroutine)
        task.add_done_callback(lambda _task: self.runs.discard(key))
        results.bind(task)
        return results

    async def run_all(self, coroutines: Iterable[Coroutine[Any, Any, Any]]) -> None:
        try:
            async with asyncio.TaskGroup() as tasks:
                for coroutine in coroutines:
                    tasks.create_task(coroutine)
        except BaseExceptionGroup as group:
            raise _first_error(group) from None

    async def close(self) -> None:
        while self.tasks:
            tasks = tuple(self.tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["CurrentResults", "Pool", "SequentialPool", "TaskQueue", "ThreadPool"]
