"""Composable consumers for ``Flow.run_streaming`` event streams."""

from rflow.consumers.base import ConsumerGroup, StreamConsumer
from rflow.consumers.checkpoint import GraphCheckpointer
from rflow.consumers.sync import WorkspaceSync
from rflow.consumers.ui import LiveGraphTree, LiveTreeRenderer, render_tree

__all__ = [
    "ConsumerGroup",
    "FlowTUI",
    "GraphCheckpointer",
    "LiveGraphTree",
    "LiveTreeRenderer",
    "StreamConsumer",
    "WorkspaceSync",
    "render_tree",
    "tui",
]


def __getattr__(name: str):
    if name in ("FlowTUI", "tui"):
        from rflow.consumers import tui as _tui

        return getattr(_tui, name)
    raise AttributeError(name)
