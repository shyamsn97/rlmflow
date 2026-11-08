"""Navigating a graph: each agent's history, its children, and their results."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow import Graph, Node
from rlmflow.view.text import kind


def build_graph():
    spec = importlib.util.spec_from_file_location(
        "graph_01_query", Path(__file__).with_name("01_query.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load 01_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


def print_tree(g: Graph, agent: Node, depth: int = 0) -> None:
    """One indented line per node of ``agent``'s history, then each child below it."""
    pad = "  " * depth
    print(f"{pad}{g.path(agent)} [{g.status(agent)}]")
    for node in g.history(agent):
        print(f"{pad}  {kind(node)}")
    for child in g.children(agent):
        print_tree(g, child, depth + 1)


def main() -> None:
    g, root = build_graph()

    print_tree(g, root)
    print("\nagents:", [g.path(agent) for agent in g.subtree(root)])
    print("children:", [g.path(child) for child in g.children(root)])
    for child in g.children(root):
        print(f"{g.path(child)} result:", g.result(child))
    print("len(root history):", len(g.history(root)))


if __name__ == "__main__":
    main()
