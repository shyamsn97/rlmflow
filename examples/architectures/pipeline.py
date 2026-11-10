"""Pipeline: a planner, a coder and a reviewer, each starting from the results before it.

    python examples/architectures/pipeline.py --model gpt-5-mini

A root runs the stages in order: it awaits each agent and passes its result to the
next one as an argument, where it lands as a REPL variable. The run is saved under
``examples/_runs/architectures/pipeline``.
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

TASK = """Run a three-stage pipeline for a Python function that parses durations like '1h30m'.
Each stage is one agent: await it, then pass its result to the next as an argument.

    spec = await launch_agent("Write a short spec for the function.", config="planner")
    code = await launch_agent(
        "Implement `spec` as a Python function, test it in your REPL, and submit its source.",
        config="coder",
        spec=spec,
    )
    review = await launch_agent(
        "Review `code` against `spec` and list any bugs.",
        config="reviewer",
        spec=spec,
        code=code,
    )

Check that each stage returned something before starting the next, then submit the review."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    args = parser.parse_args()

    flow = Flow(
        configs=[
            AgentConfig(model=args.model, max_depth=1),
            AgentConfig(name="planner", model=args.model, max_depth=0),
            AgentConfig(name="coder", model=args.model, max_depth=0),
            AgentConfig(name="reviewer", model=args.model, max_depth=0),
        ]
    )
    g = Graph()
    root = g.start(TASK, name="pipeline")
    out_dir = example_run_dir("architectures") / "pipeline"
    try:
        asyncio.run(print_run(flow, g, out_dir, width=20))
    finally:
        flow.runtime.close_repls()
    print(f"\n{root.final_result()}\n\nsaved {out_dir}")


if __name__ == "__main__":
    main()
