"""Mutating a minimal Node trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rlmflow import (
    DoneOutput,
    Node,
    UserQuery,
    surgery,
)
from rlmflow.view import render_tree


def build_graph() -> Node:
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

    test_root = graph.transcript("root.test")[0]
    assert test_root.parent is not None
    parent = test_root.parent
    surgery.remove(graph, test_root.id)
    docs = parent.attach(UserQuery(agent_id="root.docs", content="write docs"))
    docs.attach(DoneOutput(result="wrote README"))

    surgery.insert(
        graph,
        graph.tail("root").id,
        DoneOutput(result="package shipped with docs"),
        mode="replace",
    )
    print("\nafter:\n" + render_tree(graph))


if __name__ == "__main__":
    main()
