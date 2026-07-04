"""Navigating a minimal Graph."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rflow.minimal import render_tree


def build_graph():
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

    print(render_tree(graph))
    print("\nwalk:", [agent.agent_id for agent in graph.walk()])
    print("children:", list(graph.children))
    print("root.test result:", graph.children["root.test"].result())
    print("len(root nodes):", len(graph.nodes))


if __name__ == "__main__":
    main()
