"""Navigating a graph: each agent's history, its children, and their results."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow import AgentStart, Graph, reads


def build_graph():
    spec = importlib.util.spec_from_file_location(
        "graph_01_query", Path(__file__).with_name("01_query.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load 01_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


def print_tree(g: Graph, agent: AgentStart, depth: int = 0) -> None:
    """One indented line per node of ``agent``'s history, then each child below it."""
    pad = "  " * depth
    print(f"{pad}{reads.path(g, agent)}")
    for node in reads.history(g, agent)[1:]:
        print(f"{pad}  {node.type}")
    for child in reads.children(g, agent):
        print_tree(g, child, depth + 1)


def main() -> None:
    g, root = build_graph()

    print_tree(g, root)
    print("\nlog:", [item.seq for item in g.items()][:8], "...")
    print("agents:", [reads.path(g, agent) for agent in reads.tree(g, root)])
    print("children:", [reads.path(g, child) for child in reads.children(g, root)])
    for child in reads.children(g, root):
        print(f"{reads.path(g, child)} result:", reads.result(g, child))
    print("len(root history):", len(reads.history(g, root)))


if __name__ == "__main__":
    main()
