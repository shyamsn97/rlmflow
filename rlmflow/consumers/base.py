"""Small consumer primitives for published Node streams."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress

from rlmflow.graph import Node


class StreamConsumer:
    """Base class for objects that react to published Nodes."""

    def handle(self, node: Node) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources or flush final state."""


class ConsumerGroup(StreamConsumer):
    """Fan out each Node to a list of consumers."""

    def __init__(self, consumers: Iterable[StreamConsumer] = ()) -> None:
        self.consumers = list(consumers)

    def append(self, consumer: StreamConsumer) -> None:
        self.consumers.append(consumer)

    def handle(self, node: Node) -> None:
        for consumer in self.consumers:
            consumer.handle(node)

    def close(self) -> None:
        for consumer in reversed(self.consumers):
            with suppress(Exception):
                consumer.close()


__all__ = ["ConsumerGroup", "StreamConsumer"]
