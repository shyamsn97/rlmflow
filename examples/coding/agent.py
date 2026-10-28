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
    CODING_HANDOFF_SCHEMA,
    FILE_TOOLS,
    AgentConfig,
    DockerRuntime,
    Flow,
    Graph,
    LocalRuntime,
)
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import print_run  # noqa: E402


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
    parser.add_argument("--max-depth", type=int, default=1)
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
        client_for(args.model),
        llm_clients={"fast": client_for(args.fast_model)},
        configs=[
            AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
            AgentConfig(
                name="worker",
                model="fast",
                description="implements one well-scoped change and reports the files it touched",
                max_depth=args.max_depth,
                max_iters=args.max_iters,
                output_schema=CODING_HANDOFF_SCHEMA,
            ),
        ],
        runtime=runtime,
        tools=[FILE_TOOLS],
        workers=args.max_concurrency,
    )

    try:
        _run_cli(flow, workdir)
    finally:
        flow.runtime.close_repls()


def _run_cli(flow: Flow, workdir: Path) -> None:
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

        g = Graph()
        root = g.create_agent(query)
        asyncio.run(print_run(flow, g, graph_dir, width=20))

        print(f"\n{g.result(root) or '(no result)'}\n")
        print(f"Run checkpointed to {graph_dir}")
        print(f"Files written under {workdir}")


if __name__ == "__main__":
    main()
