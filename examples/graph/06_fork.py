"""Forking a graph: branches share their prefix and nothing is copied."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmflow import Graph, Node
from rlmflow.view.text import kind


def print_history(g: Graph, agent: Node) -> None:
    print("  " + " -> ".join(kind(node) for node in g.history(agent)))


def finish(g: Graph, agent: Node, result: str) -> None:
    """Append the turn that ends ``agent`` with ``result``."""
    g.append(agent, Node("assistant", result=result, seen=len(g.history(agent)), terminal=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    g = Graph()
    base = g.add_agent("choose a release plan", name="base")
    # Each fork is a new agent whose history is the base up to the fork point.
    conservative = g.fork(base, name="conservative")
    bold = g.fork(base, name="bold")

    finish(g, conservative, "ship a small patch release")
    finish(g, bold, "ship a major release candidate")

    for agent in (base, conservative, bold):
        print(f"{agent.root_data.name}: {g.result(agent)!r}")
        print_history(g, agent)
    if args.out_dir is not None:
        print("saved:", g.save(args.out_dir))


if __name__ == "__main__":
    main()
