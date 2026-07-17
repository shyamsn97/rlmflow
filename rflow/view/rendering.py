"""Compatibility facade for stream UI consumers.

Use :mod:`rflow.consumers.ui` for new imports.
"""

from rflow.consumers.ui import LiveTreeRenderer, _clip, render_tree

__all__ = ["LiveTreeRenderer", "_clip", "render_tree"]
