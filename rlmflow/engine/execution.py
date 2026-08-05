"""Pooled compute and the small queue that runs graph leaves."""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from rlmflow.graph.nodes import AgentStart, Node


class Pool(ABC):
    """Where compute runs and how many compute calls may run at once."""

    async def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run lightweight async orchestration without consuming a compute slot."""
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    @abstractmethod
    async def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        key: object | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one compute call under this backend's placement policy."""

    def close(self) -> None:
        """Release whatever the backend holds."""


class ThreadPool(Pool):
    """Offload blocking compute to threads."""

    def __init__(self, workers: int | None = None) -> None:
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers else None

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
    """Run leaves through a pool and report their completed transitions."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool
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
        transition = await self.pool.run(fn, node)
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
