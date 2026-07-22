"""Rendering a minimal Graph."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow import Graph, open_viewer, render_tree


def build_graph() -> Graph:
    spec = importlib.util.spec_from_file_location(
        "graph_01_query", Path(__file__).with_name("01_query.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load 01_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


def main() -> None:
    graph = build_graph()
    out_dir = Path(__file__).resolve().parents[1] / "_runs" / "graph-render"
    path = graph.save(out_dir)

    print(render_tree(graph))
    print(f"\nGraph saved to {path}")
    open_viewer(graph, launch=False)


if __name__ == "__main__":
    main()
