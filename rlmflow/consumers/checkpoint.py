"""Persist Node trees while they stream."""

from __future__ import annotations

import time
from pathlib import Path

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph.nodes import AgentStart, DoneOutput, ErrorOutput, Node


class GraphCheckpointer(StreamConsumer):
    """Periodically save one streamed run and always flush its final revision."""

    def __init__(
        self,
        path: str | Path,
        *,
        interval_s: float | None = 2.0,
        interval_nodes: int | None = 50,
    ) -> None:
        if interval_s is not None and interval_s <= 0:
            raise ValueError("interval_s must be positive or None")
        if interval_nodes is not None and interval_nodes <= 0:
            raise ValueError("interval_nodes must be positive or None")
        self.path = Path(path)
        self.interval_s = interval_s
        self.interval_nodes = interval_nodes
        self.last_flush_at = time.monotonic()
        self.latest: AgentStart | None = None
        self.dirty_nodes = 0
        self.saved_revision = -1

    def handle(self, node: Node) -> None:
        root = node.root
        if root is None:
            return
        if self.latest is not None and root is not self.latest:
            raise ValueError("one GraphCheckpointer cannot write multiple runs")
        self.latest = root
        self.dirty_nodes += 1
        if isinstance(node, (DoneOutput, ErrorOutput)) or self._due():
            self.flush()

    def _due(self) -> bool:
        if self.interval_nodes is not None and self.dirty_nodes >= self.interval_nodes:
            return True
        return (
            self.interval_s is not None and time.monotonic() - self.last_flush_at >= self.interval_s
        )

    def flush(self) -> None:
        root = self.latest
        if root is None:
            return
        revision = root.stats.revision
        if revision == self.saved_revision:
            return
        root.save(self.path)
        self.saved_revision = revision
        self.dirty_nodes = 0
        self.last_flush_at = time.monotonic()

    def close(self) -> None:
        self.flush()


__all__ = ["GraphCheckpointer"]
