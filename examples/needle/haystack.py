"""Needle in a massive in-memory input.

Inspired by alexzhang13/rlm-minimal's million-line magic-number demo. This
version passes the haystack as a single REPL input instead of writing many
files, so the agent must chunk `INPUTS["haystack"]` and fan out parallel child
agents.

Usage:
    python examples/needle/haystack.py
    python examples/needle/haystack.py --no-viz
    python examples/needle/haystack.py --num-lines 1000000
    python examples/needle/haystack.py --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import sys
from pathlib import Path

from rlmflow import (
    AgentConfig,
    DockerRuntime,
    Flow,
)
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer, LiveGraphTree

examples_dir = next(path for path in Path(__file__).resolve().parents if path.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client  # noqa: E402, RUF100


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
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true", help="Disable the live agent tree")
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

    haystack, answer, _needle_line = generate_massive_context(num_lines=args.num_lines)

    runtime = DockerRuntime(args.docker_image) if args.docker_image else None

    llm = build_client(args.model)
    llm_clients = None
    if args.fast_model:
        llm_clients = {"fast": build_client(args.fast_model)}

    flow = Flow(
        llm,
        llm_clients=llm_clients,
        runtime=runtime,
        config=AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
    )

    root = flow.start(
        "I'm looking for a magic number buried somewhere in the haystack in "
        "INPUTS['haystack']. What is it? Chunk the string and search the "
        "pieces in parallel.",
        inputs={"haystack": haystack},
    )
    out_dir = Path(args.out_dir)
    consumers = ConsumerGroup()
    if not args.no_viz:
        consumers.append(LiveGraphTree(title="Needle haystack search"))
    if args.out_dir:
        consumers.append(GraphCheckpointer(out_dir))

    async def drive() -> None:
        try:
            async for node in flow.run_streaming(root):
                consumers.handle(node)
        finally:
            consumers.close()

    asyncio.run(drive())

    print(f"\n{'=' * 40}")
    print(f"Result:         {root.result()}")
    print(f"Actual answer:  {answer}")
    print(f"Correct:        {answer in root.result()}")
    print(f"Run saved to    {out_dir}")

    flow.runtime.close_repls()


if __name__ == "__main__":
    main()
