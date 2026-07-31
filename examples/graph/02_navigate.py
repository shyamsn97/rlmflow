"""Navigating a minimal Node trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow.view import render_tree


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
    print("\nwalk:", [node.id for node in graph.walk()])
    print("agents:", list(graph.agent_ids()))
    print("children:", list(graph.child_agents("root")))
    print("root.test result:", graph.agent_result("root.test"))
    print("len(root nodes):", len(graph.transcript("root")))


if __name__ == "__main__":
    main()
