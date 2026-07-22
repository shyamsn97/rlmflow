"""Compatibility facade for stream UI consumers.

Use :mod:`rflow.consumers.ui` for new imports.
"""

from rflow.consumers.ui import LiveGraphTree, LiveTreeRenderer, _clip, render_tree

__all__ = ["LiveGraphTree", "LiveTreeRenderer", "_clip", "render_tree"]
