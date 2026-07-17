"""Mutating a minimal Graph."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rflow import DoneOutput, Graph, UserQuery, render_tree


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
    print("before:\n" + render_tree(graph))

    graph.remove_child("root.test")
    sibling = Graph(
        agent_id="root.docs",
        query="write docs",
        depth=1,
        parent_agent_id="root",
    )
    sibling.commit(UserQuery(content=sibling.query))
    sibling.commit(DoneOutput(result="wrote README"))
    graph.children[sibling.agent_id] = sibling

    graph.replace(graph.current().id, DoneOutput(result="package shipped with docs"))
    print("\nafter:\n" + render_tree(graph))


if __name__ == "__main__":
    main()
