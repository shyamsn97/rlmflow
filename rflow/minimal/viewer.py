"""Minimal graph viewer built on ``render_tree``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rflow.minimal.graph import Graph
from rflow.minimal.rendering import render_tree

ViewSource = Graph | Iterable[Graph] | str | Path


def _graphs_from(source: ViewSource) -> list[Graph]:
    if isinstance(source, Graph):
        return [source]
    if isinstance(source, str | Path):
        return [Graph.load(source)]
    return list(source)


def open_viewer(source: ViewSource, **launch_kwargs: object) -> object:
    """Open a tiny Gradio viewer, or print trees if Gradio is unavailable."""

    graphs = _graphs_from(source)
    if not graphs:
        raise ValueError("open_viewer needs at least one Graph")
    should_launch = bool(launch_kwargs.pop("launch", True))

    try:
        import gradio as gr
    except ImportError:
        for index, graph in enumerate(graphs):
            print(f"\n=== graph {index} ===")
            print(render_tree(graph))
        return graphs

    labels = [f"{index}: {graph.agent_id}" for index, graph in enumerate(graphs)]

    def show(label: str) -> str:
        index = labels.index(label)
        graph = graphs[index]
        result = graph.result()
        usage = graph.usage()
        return (
            f"{render_tree(graph)}\n\n"
            f"result: {result or '<unfinished>'}\n"
            f"tokens: {usage.input_tokens + usage.output_tokens}"
        )

    with gr.Blocks(title="rflow.minimal viewer") as demo:
        gr.Markdown("# rflow.minimal viewer")
        selector = gr.Dropdown(labels, value=labels[-1], label="Snapshot")
        tree = gr.Textbox(show(labels[-1]), lines=24, label="Graph")
        selector.change(show, inputs=selector, outputs=tree)

    if not should_launch:
        return demo
    return demo.launch(**launch_kwargs)


__all__ = ["ViewSource", "open_viewer"]
