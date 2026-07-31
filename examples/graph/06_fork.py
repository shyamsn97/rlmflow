"""Forking a minimal Node trajectory."""

from __future__ import annotations

from rlmflow import (
    DoneOutput,
    Node,
    start,
    surgery,
)
from rlmflow.view import render_tree


def seed_graph() -> Node:
    return start(query="choose a release plan")


def main() -> None:
    base = seed_graph()
    conservative = surgery.fork(base)
    bold = surgery.fork(base)

    conservative.tail().attach(DoneOutput(result="ship a small patch release"))
    bold.tail().attach(DoneOutput(result="ship a major release candidate"))

    print("base:\n" + render_tree(base))
    print("\nconservative:\n" + render_tree(conservative))
    print("\nbold:\n" + render_tree(bold))


if __name__ == "__main__":
    main()
