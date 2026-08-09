"""Render a run: as a figure, a timeline, a stepper, a browser app, or frames.

Because the run is a typed graph, every view is a render of that graph::

    from rlmflow import persistence
    from rlmflow.view import save_html, save_svg, steps

    graph = persistence.load("runs/coding/graph")
    for step in steps(graph):
        print(step.index, step.title, step.summary)
    save_svg(graph, "graph.svg")
    save_html(graph, "run.html")     # single self-contained stepper

Or from the terminal, without writing any of that::

    rlmflow view runs/coding/graph
    rlmflow view runs/coding/graph --html run.html

The figure, the timeline, the stepper, and ``replay`` need nothing beyond the
standard library. Two views do, and say so when reached: ``open_viewer`` wants
Gradio (``rlmflow[viewer]``), and ``save_frames``/``save_gif`` want a rasteriser
(``rlmflow[image]``).
"""

from typing import TYPE_CHECKING, Any

from rlmflow.view.figure import (
    NODE_COLORS,
    NODE_SHAPES,
    figure_title,
    graph_svg,
    node_color,
    node_shape,
)
from rlmflow.view.html import render_html, save_html, save_svg, step_svg
from rlmflow.view.replay import as_root, render_steps, replay, snapshot
from rlmflow.view.steps import Step, node_detail, steps, summarize, timeline

if TYPE_CHECKING:  # the two that carry dependencies, typed but not imported
    from rlmflow.view.app import open_viewer
    from rlmflow.view.images import save_frames, save_gif

__all__ = [
    "NODE_COLORS",
    "NODE_SHAPES",
    "Step",
    "as_root",
    "figure_title",
    "graph_svg",
    "node_color",
    "node_detail",
    "node_shape",
    "open_viewer",
    "render_html",
    "render_steps",
    "replay",
    "save_frames",
    "save_gif",
    "save_html",
    "save_svg",
    "snapshot",
    "step_svg",
    "steps",
    "summarize",
    "timeline",
]

# Held back so importing rlmflow does not pull Gradio or a rasteriser into a run
# that only ever wanted a Flow.
_OPTIONAL = {
    "open_viewer": "rlmflow.view.app",
    "save_frames": "rlmflow.view.images",
    "save_gif": "rlmflow.view.images",
}


def __getattr__(name: str) -> Any:
    if name in _OPTIONAL:
        from importlib import import_module

        return getattr(import_module(_OPTIONAL[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
