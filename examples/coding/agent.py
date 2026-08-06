"""Interactive coding agent.

A plain REPL: type a query, watch one line stream past per node the run lands,
then read the final answer. Every node checkpoints the tree under
``<workdir>/graph``, and the agent's file edits land in ``<workdir>`` itself.

Usage:
    python examples/coding/agent.py --workdir ./myproject
    python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import (
    FILE_TOOLS,
    AgentStart,
    DockerRuntime,
    Flow,
    LocalRuntime,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client  # noqa: E402


def main():
    examples_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interactive coding agent")
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(examples_root / "_runs" / "coding"),
        help="working directory the agent edits (default: examples/_runs/coding/)",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent tasks to run.",
    )
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    else:
        print(">>> LOCAL RUNTIME")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {workdir}")

    if args.docker_image:
        runtime = DockerRuntime(args.docker_image, working_directory=workdir)
    else:
        runtime = LocalRuntime(working_directory=workdir)

    flow = Flow(
        build_client(args.model),
        llm_clients={"fast": build_client(args.fast_model)},
        runtime=runtime,
        tools=FILE_TOOLS,
        workers=args.max_concurrency,
    )

    try:
        _run_cli(flow, workdir, max_depth=args.max_depth, max_iters=args.max_iters)
    finally:
        flow.runtime.close_repls()


async def _stream(flow: Flow, root: AgentStart, graph_dir: Path) -> None:
    """Drive one query to done, printing each node and checkpointing the tree."""
    async for node in flow.run_streaming(root):
        print(f"{node.parent_agent.config.path:<20} {node.type}")
        root.save(graph_dir)


def _run_cli(flow: Flow, workdir: Path, *, max_depth: int, max_iters: int) -> None:
    graph_dir = workdir / "graph"
    print("Agent ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        root = flow.start(query, max_depth=max_depth, max_iters=max_iters)
        asyncio.run(_stream(flow, root, graph_dir))

        print(f"\n{root.result() or '(no result)'}\n")
        print(f"Run checkpointed to {graph_dir}")
        print(f"Files written under {workdir}")


if __name__ == "__main__":
    main()
