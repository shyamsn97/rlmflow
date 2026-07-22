"""Compatibility facade for stream UI consumers.

Use :mod:`rlmflow.consumers.ui` for new imports.
"""

from rlmflow.consumers.ui import LiveGraphTree, LiveTreeRenderer, _clip, render_tree

__all__ = ["LiveGraphTree", "LiveTreeRenderer", "_clip", "render_tree"]
