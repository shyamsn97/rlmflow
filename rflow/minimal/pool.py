"""Small async pools for minimal Flow scheduling."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from contextvars import ContextVar
from typing import TypeVar

T = TypeVar("T")
_ACTIVE_POOLS: ContextVar[frozenset[int]] = ContextVar(
    "minimal_rflow_active_pools", default=frozenset()
)


class AsyncPool:
    """Bound concurrent async work with a semaphore.

    If work scheduled through the pool schedules more work on the same pool, the
    nested work runs directly. That avoids deadlocking a parent that is waiting
    for children while holding a pool slot.
    """

    def __init__(self, max_concurrency: int | None = None) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self._semaphore = None if max_concurrency is None else asyncio.Semaphore(max_concurrency)

    async def run(self, work: Awaitable[T]) -> T:
        if self._semaphore is None or id(self) in _ACTIVE_POOLS.get():
            return await work
        async with self._semaphore:
            active = _ACTIVE_POOLS.get()
            token = _ACTIVE_POOLS.set(active | {id(self)})
            try:
                return await work
            finally:
                _ACTIVE_POOLS.reset(token)

    async def gather(self, *work: Awaitable[T]) -> list[T]:
        tasks = [asyncio.create_task(self.run(item)) for item in work]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


class SequentialPool:
    """Run async work one item at a time."""

    max_concurrency = 1

    async def run(self, work: Awaitable[T]) -> T:
        return await work

    async def gather(self, *work: Awaitable[T]) -> list[T]:
        results: list[T] = []
        for index, item in enumerate(work):
            try:
                results.append(await item)
            except BaseException:
                for leftover in work[index + 1 :]:
                    if inspect.iscoroutine(leftover):
                        leftover.close()
                raise
        return results


__all__ = ["AsyncPool", "SequentialPool"]
