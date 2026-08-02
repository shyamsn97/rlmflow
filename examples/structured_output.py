"""Structured output for root runs and delegated child agents.

This live example asks a real model to extract facts from a provided trip brief.
The root launches child agents with JSON Schema output contracts, receives their
typed dictionary results, then returns a root result that validates against a
Pydantic model.

Run:
    export OPENAI_API_KEY=...
    python examples/structured_output.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import AgentConfig, Flow, json_schema_for, start
from rlmflow.llm import OpenAIClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir  # noqa: E402


class CityForecast(BaseModel):
    """Forecast facts for one city extracted from the trip brief."""

    city: str
    condition: Literal["rain", "sun", "clouds"]
    high_f: float
    packing_tip: str


class PackingPlan(BaseModel):
    """Packing plan synthesized from the structured city forecasts."""

    destination_count: int
    forecasts: list[CityForecast]
    shared_items: list[str]
    summary: str


TRIP_BRIEF = """\
Trip brief for structured extraction.

Seattle leg:
- City: Seattle
- Forecast condition: rain
- Forecast high: 60.0 F

Austin leg:
- City: Austin
- Forecast condition: sun
- Forecast high: 96.0 F

Denver leg:
- City: Denver
- Forecast condition: clouds
- Forecast high: 72.0 F
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Live structured output example")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default=str(example_run_dir("structured-output")),
        help="save the final run here (default: examples/_runs/structured-output/)",
    )
    args = parser.parse_args()

    flow = Flow(
        OpenAIClient(args.model),
        config=AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
    )

    query = (
        "Build a packing plan for my upcoming trip. The important weather info "
        'is in `INPUTS["trip_brief"]`. Make sure to delegate each city to child '
        "agents."
    )

    root = start(
        query,
        inputs={"trip_brief": TRIP_BRIEF},
        output_schema=json_schema_for(PackingPlan),
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )
    out_dir = Path(args.out_dir)

    async def drive() -> None:
        async for node in flow.run_streaming(root):
            print(f"{node.parent_agent.config.path}  {node.type}")
            root.save(out_dir)

    asyncio.run(drive())

    # `done(value)` parsed the answer against the schema, so the result is a
    # dict; the Pydantic model is the typed view of that same value.
    plan = PackingPlan.model_validate(root.result())

    print("Typed result:")
    print(type(plan).__name__)
    print(plan.model_dump_json(indent=2))

    print("\nRoot result:")
    print(json.dumps(root.result(), indent=2))

    print(f"\nRun saved to {out_dir}")

    flow.runtime.close_repls()


if __name__ == "__main__":
    main()
