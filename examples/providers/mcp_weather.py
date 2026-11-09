"""Use Flow with a real MCP weather server.

Requires MCP and a live LLM client:

    export OPENAI_API_KEY=...
    pip install -e ".[openai,mcp]"
    python examples/providers/mcp_weather.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import AgentConfig, Flow, Graph, LocalRuntime
from rlmflow.integrations.mcp import MCPTools

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import print_run  # noqa: E402

DEFAULT_QUERY = """I will be in Seattle today, Austin 3 days after that, and San Francisco 5 days after that. Check the weather and tell me what to pack for each city.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow + MCP weather example")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "mcp-weather"),
        help="Save the final run here (default: examples/_runs/mcp-weather/).",
    )
    args = parser.parse_args()

    server = Path(__file__).with_name("mcp_weather_server.py")
    # The Flow starts the server when the run begins and stops it when the run ends.
    flow = Flow(
        configs=[AgentConfig(model=args.model, max_depth=args.max_depth, max_iters=args.max_iters)],
        runtime=LocalRuntime(),
        tools=[MCPTools(sys.executable, str(server))],
    )
    g = Graph()
    root = g.start(args.query)
    out_dir = Path(args.out_dir)
    asyncio.run(print_run(flow, g, out_dir))

    print(f"\n{'=' * 60}\nWEATHER PACKING RECOMMENDATION\n{'=' * 60}")
    print(g.result(root))
    print(f"\nRun saved to {out_dir}")
    flow.runtime.close_repls()


if __name__ == "__main__":
    main()
