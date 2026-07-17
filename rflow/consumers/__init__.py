"""Composable consumers for ``Flow.run_streaming`` event streams."""

from rflow.consumers.base import ConsumerGroup, StreamConsumer
from rflow.consumers.checkpoint import GraphCheckpointer
from rflow.consumers.sync import WorkspaceSync
from rflow.consumers.ui import LiveTreeRenderer, render_tree

__all__ = [
    "ConsumerGroup",
    "GraphCheckpointer",
    "LiveTreeRenderer",
    "StreamConsumer",
    "WorkspaceSync",
    "render_tree",
    "tui",
]


def __getattr__(name: str):
    if name == "tui":
        from rflow.consumers.tui import tui

        return tui
    raise AttributeError(name)
