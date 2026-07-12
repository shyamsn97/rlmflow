"""Interactive coding agent.

A REPL interface to a Flow coding agent. Talk to it, give it tasks, it writes
and edits files in your working directory using delegation.

Usage:
    python examples/coding/agent.py --workdir ./myproject
    python examples/coding/agent.py --workdir ./myproject --no-viz
    python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rflow.clients import AnthropicClient, OpenAIClient
from rflow import (
    DockerRuntime,
    FILE_TOOLS,
    Flow,
    Graph,
    LiveTreeRenderer,
    LocalRuntime,
)


def build_llm(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


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
    parser.add_argument("--no-viz", action="store_true")
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

    # The runtime decides where code runs and carries the file tools. Local runs
    # in-process with the cwd switched into `workdir`; Docker runs each agent in a
    # container with `workdir` bind-mounted to /workspace. Same interface.
    if args.docker_image:
        runtime = DockerRuntime(args.docker_image, working_directory=workdir)
    else:
        runtime = LocalRuntime(working_directory=workdir)
    runtime.register_tools(FILE_TOOLS)

    flow = Flow(
        build_llm(args.model),
        llm_clients={"fast": build_llm(args.fast_model)},
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        max_concurrency=args.max_concurrency,
    )

    print("Agent ready. Type a query, or 'quit' to exit.\n")
    try:
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break

            graph = Graph(query=query)

            async def drive() -> None:
                renderer = LiveTreeRenderer(clear=not args.no_viz)
                async for event in flow.run_streaming(graph):
                    renderer.handle(event, graph)

            asyncio.run(drive())

            print(f"\n{graph.result() or '(no result)'}\n")
            path = graph.save(workdir / "graph")
            print(f"Graph saved to {path}")
            print(f"Files written under {workdir}")
    finally:
        flow.close_repls()


if __name__ == "__main__":
    main()
