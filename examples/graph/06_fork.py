"""Forking a graph: branches share their prefix and nothing is copied."""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmflow import AgentStart, DoneOutput, Graph, ops, reads


def print_history(g: Graph, agent: AgentStart) -> None:
    print("  " + " -> ".join(f"{node.type}" for node in reads.history(g, agent)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    g = Graph()
    base = ops.spawn(g, "choose a release plan")
    # Each fork is a new agent whose history is the base up to the fork point.
    conservative = ops.fork(g, base, name="conservative")
    bold = ops.fork(g, base, name="bold")

    ops.append(g, conservative, DoneOutput(result="ship a small patch release"))
    ops.append(g, bold, DoneOutput(result="ship a major release candidate"))

    for agent in (base, conservative, bold):
        print(f"{agent.config.name}: {reads.result(g, agent)!r}")
        print_history(g, agent)
    if args.out_dir is not None:
        print("saved:", g.save(args.out_dir))


if __name__ == "__main__":
    main()
