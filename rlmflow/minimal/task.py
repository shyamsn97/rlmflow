"""A queue of calls in flight, each tagged with a key its caller groups by."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any


class TaskQueue:
    """Run calls concurrently and hold each key's result.

    What a key is, and what the calls do, is not its business: keys are compared by
    identity and never inspected.
    """

    def __init__(self) -> None:
        self.running: dict[asyncio.Task[Any], Any] = {}
        self.results: dict[int, asyncio.Future[Any]] = {}
        self.wake: asyncio.Future[None] | None = None

    def result(self, key: Any) -> asyncio.Future[Any]:
        """The future this key's work reports to, opened on first ask."""
        if id(key) not in self.results:
            self.results[id(key)] = asyncio.get_running_loop().create_future()
        return self.results[id(key)]

    def completed(self, key: Any) -> bool:
        result = self.results.get(id(key))
        return result is not None and result.done()

    def complete(self, key: Any, value: Any = None, error: BaseException | None = None) -> None:
        """Report this key's outcome, unless something already did."""
        result = self.result(key)
        if result.done():
            return
        if error is None:
            result.set_result(value)
        else:
            result.set_exception(error)

    def forget(self, key: Any) -> None:
        self.results.pop(id(key), None)

    def submit(self, key: Any, fn: Callable[..., Awaitable[Any]], *args: Any) -> asyncio.Task[Any]:
        handle = asyncio.create_task(fn(*args))
        self.running[handle] = key
        handle.add_done_callback(self._woke)
        return handle

    def busy(self, key: Any) -> bool:
        """Is a call tagged with this key still running?"""
        return any(running is key for running in self.running.values())

    def pending(self, belongs: Callable[[Any], bool]) -> list[asyncio.Task[Any]]:
        return [handle for handle, key in self.running.items() if belongs(key)]

    async def wave(self) -> list[tuple[Any, asyncio.Task[Any]]]:
        """Wait until at least one call lands, and hand back the ones that did.

        Waiting on a fresh future rather than on the handles themselves, because a
        call submitted from inside a running call still has to wake us.
        """
        while not any(handle.done() for handle in self.running):
            self.wake = asyncio.get_running_loop().create_future()
            await self.wake
        done = [handle for handle in self.running if handle.done()]
        return [(self.running.pop(handle), handle) for handle in done]

    async def cancel(self, handles: Iterable[asyncio.Task[Any]]) -> None:
        handles = list(handles)
        for handle in handles:
            handle.cancel()
            self.running.pop(handle, None)
        await asyncio.gather(*handles, return_exceptions=True)

    async def close(self) -> None:
        while self.running:
            await self.cancel(list(self.running))

    def _woke(self, _handle: asyncio.Task[Any]) -> None:
        if self.wake is not None and not self.wake.done():
            self.wake.set_result(None)


__all__ = ["TaskQueue"]
