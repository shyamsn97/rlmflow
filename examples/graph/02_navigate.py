"""Navigating a graph: each agent's nodes, its children, and their results."""

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
    """One indented line per node of ``agent``, then each child below it."""
    pad = "  " * depth
    print(f"{pad}{agent.path()} [{agent.status()}]")
    for node in agent.nodes():
        print(f"{pad}  {kind(node)}")
    for child in agent.children():
        print_tree(g, child, depth + 1)


def main() -> None:
    g, root = build_graph()

    print_tree(g, root)
    print("\nagents:", [agent.path() for agent in root.subtree()])
    print("children:", [child.path() for child in root.children()])
    for child in root.children():
        print(f"{child.path()} result:", child.final_result())
    print("len(root.nodes()):", len(root.nodes()))


if __name__ == "__main__":
    main()
