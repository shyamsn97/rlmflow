"""Forking a minimal Node trajectory."""

from __future__ import annotations

from rlmflow import (
    AgentStart,
    DoneOutput,
    Node,
    start,
)


def seed_graph() -> AgentStart:
    return start(query="choose a release plan")


def print_tree(node: Node, depth: int = 0) -> None:
    """One indented line per node, from ``node`` down."""
    label = f"{node.type} [{node.config.path}]" if isinstance(node, AgentStart) else node.type
    print(f"{'  ' * depth}{label}")
    for child in node.children:
        print_tree(child, depth + 1)


def main() -> None:
    base = seed_graph()
    # Each fork copies the tree and cuts everything after the forked node, so the
    # two branches share this base and nothing else.
    conservative = base.fork()
    bold = base.fork()

    conservative.frontier.append(DoneOutput(result="ship a small patch release"))
    bold.frontier.append(DoneOutput(result="ship a major release candidate"))

    print("base:")
    print_tree(base)
    print("\nconservative:")
    print_tree(conservative)
    print("\nbold:")
    print_tree(bold)


if __name__ == "__main__":
    main()
