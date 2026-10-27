"""Forking a graph: branches share their prefix and nothing is copied."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmflow import DONE, Graph, Node


def print_history(g: Graph, agent: Node) -> None:
    print("  " + " -> ".join(f"{node.type}" for node in g.history(agent)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    g = Graph()
    base = g.spawn("choose a release plan", name="base")
    # Each fork is a new agent whose history is the base up to the fork point.
    conservative = g.fork(base, name="conservative")
    bold = g.fork(base, name="bold")

    g.append(conservative, Node(DONE, result="ship a small patch release"))
    g.append(bold, Node(DONE, result="ship a major release candidate"))

    for agent in (base, conservative, bold):
        print(f"{agent.name}: {g.result(agent)!r}")
        print_history(g, agent)
    if args.out_dir is not None:
        print("saved:", g.save(args.out_dir))


if __name__ == "__main__":
    main()
