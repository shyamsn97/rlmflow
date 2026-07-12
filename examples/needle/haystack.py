"""Needle in a massive in-memory input.

Inspired by alexzhang13/rlm-minimal's million-line magic-number demo. This
version passes the haystack as a single REPL input instead of writing many
files, so the agent must chunk `INPUTS["haystack"]` and fan out parallel child
agents.

Usage:
    python examples/needle/haystack.py
    python examples/needle/haystack.py --num-lines 1000000 --no-viz
    python examples/needle/haystack.py --viewer
    python examples/needle/haystack.py --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
from pathlib import Path

from rflow.clients import AnthropicClient, OpenAIClient
from rflow import DockerRuntime, Flow, Graph, LiveTreeRenderer


def generate_massive_context(
    num_lines: int = 1_000_000,
    *,
    answer: str | None = None,
) -> tuple[str, str, int]:
    print(f"Generating massive context with {num_lines:,} lines...")

    words = ["blah", "random", "text", "data", "content", "information", "sample"]
    answer = answer or "".join(random.choices(string.digits, k=7))

    lines = []
    for _ in range(num_lines):
        n = random.randint(3, 8)
        lines.append(" ".join(random.choice(words) for _ in range(n)))

    if num_lines <= 0:
        raise ValueError("--num-lines must be positive")

    low = min(num_lines - 1, max(0, int(num_lines * 0.4)))
    high = min(num_lines - 1, max(low, int(num_lines * 0.6)))
    needle_line = random.randint(low, high)
    lines[needle_line] = f"The magic number is {answer}"

    print(f"Magic number inserted at line {needle_line}")
    return "\n".join(lines), answer, needle_line


def main():
    parser = argparse.ArgumentParser(description="Needle in a massive haystack input")
    parser.add_argument("--num-lines", type=int, default=1_000_000)
    parser.add_argument(
        "--viewer", action="store_true", help="Open the state viewer after finishing"
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "needle-haystack"),
        help="Save the final run here (default: examples/_runs/needle-haystack/).",
    )
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    haystack, answer, needle_line = generate_massive_context(num_lines=args.num_lines)

    runtime = DockerRuntime(args.docker_image) if args.docker_image else None

    llm = (
        AnthropicClient(args.model)
        if args.model.startswith("claude")
        else OpenAIClient(args.model)
    )
    llm_clients = None
    if args.fast_model:
        fast = (
            AnthropicClient(args.fast_model)
            if args.fast_model.startswith("claude")
            else OpenAIClient(args.fast_model)
        )
        llm_clients = {"fast": fast}

    flow = Flow(
        llm,
        llm_clients=llm_clients,
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )

    graph = Graph(
        query=(
            "I'm looking for a magic number buried somewhere in the haystack in "
            "INPUTS['haystack']. What is it? Chunk the string and search the "
            "pieces in parallel."
        )
    )

    async def drive() -> None:
        renderer = LiveTreeRenderer(clear=not args.no_viz)
        async for event in flow.run_streaming(graph, inputs={"haystack": haystack}):
            renderer.handle(event, graph)

    asyncio.run(drive())

    print(f"\n{'=' * 40}")
    print(f"Result:         {graph.result()}")
    print(f"Actual answer:  {answer}")
    print(f"Correct:        {answer in graph.result()}")

    if args.out_dir:
        path = graph.save(Path(args.out_dir))
        print(f"Graph saved to {path}")

    if args.viewer:
        print("Viewer support is not part of the minimal example path.")

    flow.close_repls(graph.graph_id)


if __name__ == "__main__":
    main()
