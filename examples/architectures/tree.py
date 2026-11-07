"""Recursive tree: a root launches section agents, and each section launches two of its own.

    python examples/architectures/tree.py --model gpt-5-mini

The run is saved under ``examples/_runs/architectures/tree``; render it with
``rlmflow render gif examples/_runs/architectures/tree tree.gif``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import AgentConfig, Flow, Graph

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir, print_run  # noqa: E402

TASK = """Write a short field guide to three kinds of clouds: cumulus, stratus and cirrus.
Launch one agent per cloud kind. Tell each to launch two agents of its own, one to
describe what the cloud looks like and one to describe the weather it brings, and to
combine their answers into its section. Combine the three sections and submit the guide."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    args = parser.parse_args()

    flow = Flow(configs=[AgentConfig(model=args.model, max_depth=2)])
    g = Graph()
    root = g.add_agent(TASK)
    out_dir = example_run_dir("architectures") / "tree"
    try:
        asyncio.run(print_run(flow, g, out_dir, width=28))
    finally:
        flow.runtime.close_repls()
    print(f"\n{g.result(root)}\n\nsaved {out_dir}")


if __name__ == "__main__":
    main()
