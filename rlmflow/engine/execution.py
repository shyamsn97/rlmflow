"""Pooled compute and the small queue that runs graph leaves."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rlmflow.graph.nodes import AgentStart, Node, running_step


class Pool(ABC):
    """Where compute runs and how many compute calls may run at once."""

    @abstractmethod
    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        key: object | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one compute call under this backend's placement policy."""

    @abstractmethod
    def stream(
        self,
        fn: Callable[..., Any],
        *args: Any,
        key: object | None = None,
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
        key: object | None = None,
        **kwargs: Any,
    ) -> Any:
        del key
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
        key: object | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        del key
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
        key: object | None = None,
        **kwargs: Any,
    ) -> Any:
        del key
        async with self._lock:
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

    async def stream(
        self,
        fn: Callable[..., Any],
        *args: Any,
        key: object | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        del key
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
class Transition:
    """A submitted node and what it created."""

    submitted: Node
    created: Node
    error: BaseException | None = None

    @property
    def is_agent_start(self) -> bool:
        return isinstance(self.created, AgentStart)


class TaskQueue:
    """Run graph leaves and report their completed transitions."""

    def __init__(self) -> None:
        self.running: dict[int, tuple[Node, asyncio.Task[None]]] = {}
        self.done: asyncio.Queue[Transition] = asyncio.Queue()
        self.changed = asyncio.Condition()

    def __bool__(self) -> bool:
        return bool(self.running)

    def submit(
        self,
        node: Node,
        fn: Callable[[Node], Any],
        *,
        publish: bool = False,
    ) -> asyncio.Task[None]:
        """Run ``fn(node)`` once; optionally publish the node opening the work."""
        key = id(node)
        active = self.running.get(key)
        if active is not None:
            return active[1]

        task = asyncio.create_task(self._run(node, fn))
        self.running[key] = (node, task)
        if publish:
            self.done.put_nowait(Transition(submitted=node, created=node))
        return task

    async def _run(self, node: Node, fn: Callable[[Node], Any]) -> None:
        with running_step(node):
            result = fn(node)
            transition = await result if inspect.isawaitable(result) else result
        if not isinstance(transition, Transition):
            raise TypeError(f"task returned {type(transition).__name__}, expected Transition")

        # Publish before waking a parent, so a child boundary gets first refusal.
        self.done.put_nowait(transition)

        agent = transition.created.parent_agent
        if agent is not None and agent.terminal:
            async with self.changed:
                self.changed.notify_all()

    async def next(self) -> Transition:
        """Return the next graph transition."""
        transition = await self.done.get()
        if not transition.is_agent_start:
            self.running.pop(id(transition.submitted), None)
        return transition

    async def join(self, child: AgentStart) -> Any:
        """Wait for a complete child, then read its durable graph result."""
        async with self.changed:
            await self.changed.wait_for(lambda: child.terminal)
        return child.result()

    async def cancel(self, nodes: Iterable[Node] | None = None) -> None:
        """Cancel transitions for these nodes, or every transition when omitted."""
        keys = set(self.running) if nodes is None else {id(node) for node in nodes}
        tasks = [task for key, (_node, task) in self.running.items() if key in keys]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for key in keys:
            self.running.pop(key, None)


__all__ = [
    "Pool",
    "SequentialPool",
    "TaskQueue",
    "ThreadPool",
    "Transition",
]
