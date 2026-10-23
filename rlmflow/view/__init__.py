"""Render a run: as a text tree, a figure, a timeline, a stepper, a browser app, or frames.

Because the run is a typed graph, every view is a render of that graph::

    from rlmflow import Graph
    from rlmflow.view import save_html, save_svg, steps

    graph = Graph.load("runs/coding/graph")
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

from rlmflow.view.figure import (
    figure_title,
    graph_svg,
    node_color,
    run_title,
    save_frames,
    save_gif,
    step_svg,
)
from rlmflow.view.html import open_viewer, render_html, save_html, save_svg
from rlmflow.view.text import render_forest, render_tree
from rlmflow.view.timeline import (
    Step,
    as_graph,
    node_detail,
    render_steps,
    replay,
    snapshot,
    steps,
    timeline,
)

__all__ = [
    "Step",
    "as_graph",
    "figure_title",
    "graph_svg",
    "node_color",
    "node_detail",
    "open_viewer",
    "render_forest",
    "render_html",
    "render_steps",
    "render_tree",
    "replay",
    "run_title",
    "save_frames",
    "save_gif",
    "save_html",
    "save_svg",
    "snapshot",
    "step_svg",
    "steps",
    "timeline",
]
