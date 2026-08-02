"""Navigating a minimal Node trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow import AgentStart, Node


def build_graph():
    spec = importlib.util.spec_from_file_location(
        "graph_01_query", Path(__file__).with_name("01_query.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load 01_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


def print_tree(node: Node, depth: int = 0) -> None:
    """One indented line per node, from ``node`` down."""
    label = f"{node.type} [{node.config.path}]" if isinstance(node, AgentStart) else node.type
    print(f"{'  ' * depth}{label}")
    for child in node.children:
        print_tree(child, depth + 1)


def main() -> None:
    graph = build_graph()

    print_tree(graph)
    print("\nwalk:", [node.id for node in graph.walk()])
    print("agents:", [node.config.path for node in graph.walk() if isinstance(node, AgentStart)])
    print("children:", [child.config.path for child in graph.sub_agents])
    for child in graph.sub_agents:
        print(f"{child.config.path} result:", child.result())
    print("len(root nodes):", len(graph.transcript()))


if __name__ == "__main__":
    main()
