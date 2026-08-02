"""Persist Node trees while they stream."""

from __future__ import annotations

import time
from pathlib import Path

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph.nodes import AgentStart, Node


class GraphCheckpointer(StreamConsumer):
    """Save the current root periodically and once more when the stream closes."""

    def __init__(self, path: str | Path, *, every_s: float = 0.0) -> None:
        self.path = Path(path)
        self.every_s = every_s
        self.last = 0.0
        self.latest: AgentStart | None = None

    def handle(self, node: Node) -> None:
        root = node.root
        if root is None:
            return
        self.latest = root
        now = time.monotonic()
        if self.every_s and now - self.last < self.every_s:
            return
        self.save(root)

    def save(self, root: AgentStart) -> None:
        root.save(self.path)
        self.last = time.monotonic()

    def close(self) -> None:
        if self.latest is not None:
            self.save(self.latest)


__all__ = ["GraphCheckpointer"]
