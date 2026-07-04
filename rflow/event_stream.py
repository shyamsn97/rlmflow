"""Small async event stream used by Flow.

Events are delivery hints, not state. The graph remains the source of truth.
This module only owns fanout and backpressure for live consumers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

T = TypeVar("T")


class EventStream(Generic[T]):
    """A minimal one-producer/many-consumer async event stream."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[T | None]] = set()

    def publish(self, event: T) -> None:
        """Publish one event to all current subscribers."""

        for queue in list(self._subscribers):
            queue.put_nowait(event)

    def close(self) -> None:
        """Stop all current subscribers."""

        for queue in list(self._subscribers):
            queue.put_nowait(None)
        self._subscribers.clear()

    async def subscribe(self) -> AsyncIterator[T]:
        """Yield events published after subscription."""

        queue: asyncio.Queue[T | None] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._subscribers.discard(queue)


__all__ = ["EventStream"]
