"""Pools bound blocking leaf work — the one place a real thread is ever held.

A ``Pool`` caps how many *blocking* leaf calls run at once (a sync
``client.chat`` or any other blocking callable). It is a small strategy object:
swap the subclass to change how leaf work is scheduled.

- :class:`ThreadPool` — offload blocking calls to a bounded thread pool.
- :class:`SequentialPool` — run one leaf call at a time (deterministic; debug).
- (future) a Ray-backed pool for distributing leaf calls across a cluster.

Agent scheduling is **not** a pool concern: each agent turn is an unbounded
``asyncio`` task on Flow's ``TaskQueue``. Nothing that *waits on another unit of
the same pool* is ever submitted here, so a pool cannot deadlock — it only ever
holds a thread for a single leaf call that runs to completion on its own.

Async callables never touch a pool's threads: they are awaited on the event loop
and self-bound via their own connection pool.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class Pool(ABC):
    """Run blocking leaf calls under a policy defined by subclasses."""

    @abstractmethod
    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn`` under the pool's policy and return its result.

        Coroutine functions are awaited directly (never a thread). Plain
        functions run according to the pool; if such a function still returns an
        awaitable, that is awaited too.
        """

    def close(self) -> None:
        """Release any resources the pool holds (default: nothing)."""


class ThreadPool(Pool):
    """Offload blocking leaf calls to a bounded thread pool.

    ``workers`` is the cap: at most that many blocking calls run at once. Pass
    ``None`` to leave blocking calls on asyncio's default executor (effectively
    unbounded). A single ``ThreadPool`` can be shared across flows to give them
    one global thread budget.
    """

    def __init__(self, workers: int | None = None) -> None:
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers
        self._executor = (
            ThreadPoolExecutor(max_workers=workers) if workers is not None else None
        )

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )
        return await result if inspect.isawaitable(result) else result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)


class SequentialPool(Pool):
    """Run one leaf call at a time (no worker threads).

    Serializes every call behind a lock, so at most one leaf runs at once — sync
    calls run inline (blocking the loop for their duration) and async calls are
    awaited one after another. Deterministic; useful for debugging.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if inspect.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result


__all__ = ["Pool", "SequentialPool", "ThreadPool"]
