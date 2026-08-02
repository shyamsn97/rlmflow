"""Building minimal Node snapshots over time."""

from __future__ import annotations

from copy import deepcopy

from rlmflow import (
    AgentStart,
    DoneOutput,
    LLMOutput,
    Node,
    start,
)


def print_tree(node: Node, depth: int = 0) -> None:
    """One indented line per node, from ``node`` down."""
    label = f"{node.type} [{node.config.path}]" if isinstance(node, AgentStart) else node.type
    print(f"{'  ' * depth}{label}")
    for child in node.children:
        print_tree(child, depth + 1)


def main() -> None:
    graph = start(query="write hello.py")
    snapshots: list[Node] = [deepcopy(graph)]

    graph.frontier.append(LLMOutput(content='write_file("hello.py", "print(\\"hello\\")\\n")'))
    snapshots.append(deepcopy(graph))
    graph.frontier.append(DoneOutput(result="hello.py created"))
    snapshots.append(deepcopy(graph))

    for i, snap in enumerate(snapshots):
        print(f"\nstep {i}")
        print_tree(snap)


if __name__ == "__main__":
    main()
