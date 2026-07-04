"""Building minimal Graph snapshots over time."""

from __future__ import annotations

from copy import deepcopy

from rflow.minimal import DoneOutput, Graph, LLMOutput, UserQuery, render_tree


def main() -> None:
    graph = Graph(query="write hello.py")
    snapshots: list[Graph] = []

    graph.commit(UserQuery(content=graph.query))
    snapshots.append(deepcopy(graph))
    graph.commit(LLMOutput(content='write_file("hello.py", "print(\\"hello\\")\\n")'))
    snapshots.append(deepcopy(graph))
    graph.commit(DoneOutput(result="hello.py created"))
    snapshots.append(deepcopy(graph))

    for i, snap in enumerate(snapshots):
        print(f"\nstep {i}")
        print(render_tree(snap))


if __name__ == "__main__":
    main()
