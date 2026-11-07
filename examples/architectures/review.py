"""Review loop: a root critiques a finished writer twice, reopening it to revise each time.

    python examples/architectures/review.py --model gpt-5-mini

The run is saved under ``examples/_runs/architectures/review``.
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

TASK = """Launch a writer agent to write a four-line poem about a lighthouse, and await it.
Then review the poem. Send the writer your critique with `await writer.send(...)`,
which reopens it and returns its revision. Do this twice, then submit the final poem."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    args = parser.parse_args()

    flow = Flow(configs=[AgentConfig(model=args.model)])
    g = Graph()
    root = g.add_agent(TASK)
    out_dir = example_run_dir("architectures") / "review"
    try:
        asyncio.run(print_run(flow, g, out_dir, width=12))
    finally:
        flow.runtime.close_repls()
    print(f"\n{g.result(root)}\n\nsaved {out_dir}")


if __name__ == "__main__":
    main()
