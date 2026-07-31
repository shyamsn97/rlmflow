"""Building minimal Node snapshots over time."""

from __future__ import annotations

from copy import deepcopy

from rlmflow import (
    DoneOutput,
    LLMOutput,
    Node,
    start,
)
from rlmflow.view import render_tree


def main() -> None:
    graph = start(query="write hello.py")
    snapshots: list[Node] = [deepcopy(graph)]

    graph.tail().attach(LLMOutput(content='write_file("hello.py", "print(\\"hello\\")\\n")'))
    snapshots.append(deepcopy(graph))
    graph.tail().attach(DoneOutput(result="hello.py created"))
    snapshots.append(deepcopy(graph))

    for i, snap in enumerate(snapshots):
        print(f"\nstep {i}")
        print(render_tree(snap))


if __name__ == "__main__":
    main()
