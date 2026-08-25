"""Run Flow with Tinker inference.

Requires Tinker credentials and optional dependencies:

    export TINKER_API_KEY=...
    pip install -e ".[tinker]"
    python examples/providers/tinker_agent.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import AgentConfig, Flow
from rlmflow.llm import TinkerClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, save_example_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Flow task with Tinker.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Tinker base model for inference.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional saved Tinker weights path, e.g. tinker://run/weights/checkpoint.",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3",
        help="Tinker cookbook renderer name matching the model family.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument(
        "--query",
        default="Use Python to compute 17 * 23, then call done() with the answer.",
    )
    args = parser.parse_args()

    llm = TinkerClient(
        base_model=None if args.model_path else args.base_model,
        model_path=args.model_path,
        renderer=args.renderer,
        max_tokens=args.max_tokens,
    )
    flow = Flow(llm, root_config=AgentConfig(max_iters=args.max_iters))
    print(f"Query: {args.query}\n")
    root = flow.start(args.query)
    out_dir = example_run_dir("tinker-agent")

    async def drive() -> None:
        async for node in flow.run_streaming(root):
            print(f"{node.parent_agent.config.path}  {node.type}")
            root.save(out_dir)

    asyncio.run(drive())
    print(root.result())
    save_example_graph(root, "tinker-agent")
    flow.runtime.close_repls()


if __name__ == "__main__":
    main()
