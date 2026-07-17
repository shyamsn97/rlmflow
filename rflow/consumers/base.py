"""Small stream-consumer primitives for graph event streams."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress

from rflow.graph import Graph
from rflow.graph.events import Event


class StreamConsumer:
    """Base class for objects that react to streamed graph events."""

    def handle(self, event: Event, graph: Graph | None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources or flush final state."""


class ConsumerGroup(StreamConsumer):
    """Fan out each event to a list of consumers."""

    def __init__(self, consumers: Iterable[StreamConsumer] = ()) -> None:
        self.consumers = list(consumers)

    def append(self, consumer: StreamConsumer) -> None:
        self.consumers.append(consumer)

    def handle(self, event: Event, graph: Graph | None) -> None:
        for consumer in self.consumers:
            consumer.handle(event, graph)

    def close(self) -> None:
        for consumer in reversed(self.consumers):
            with suppress(Exception):
                consumer.close()


__all__ = ["ConsumerGroup", "StreamConsumer"]
