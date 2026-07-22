"""Graph checkpoint consumers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph import Graph
from rlmflow.graph.events import Event


class GraphCheckpointer(StreamConsumer):
    """Persist the current graph snapshot while a stream advances."""

    def __init__(
        self,
        path: str | Path,
        *,
        every_s: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.every_s = every_s
        self.metadata = metadata
        self.last = 0.0
        self.latest: Graph | None = None

    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is None:
            return
        self.latest = graph
        now = time.monotonic()
        if self.every_s and now - self.last < self.every_s:
            return
        self.save(graph)

    def save(self, graph: Graph) -> None:
        graph.save(self.path, metadata=self.metadata)
        self.last = time.monotonic()

    def close(self) -> None:
        if self.latest is not None:
            self.save(self.latest)


__all__ = ["GraphCheckpointer"]
