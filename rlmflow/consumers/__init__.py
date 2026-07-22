"""Composable consumers for ``Flow.run_streaming`` event streams."""

from rlmflow.consumers.base import ConsumerGroup, StreamConsumer
from rlmflow.consumers.checkpoint import GraphCheckpointer
from rlmflow.consumers.sync import WorkspaceSync
from rlmflow.consumers.ui import LiveGraphTree, LiveTreeRenderer, render_tree

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
        from rlmflow.consumers import tui as _tui

        return getattr(_tui, name)
    raise AttributeError(name)
