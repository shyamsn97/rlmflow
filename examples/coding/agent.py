"""Interactive coding agent.

Opens the rich rlmflow TUI by default: query/context inputs, live chat bubbles,
and side tabs for the execution tree, agents, counts, waiting supervisors,
errors, and latest nodes. Pass ``--cli`` for the plain REPL + live tree.

Usage:
    python examples/coding/agent.py --workdir ./myproject
    python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
    python examples/coding/agent.py --workdir ./myproject --cli
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from rlmflow import (
    ConsumerGroup,
    DockerRuntime,
    FILE_TOOLS,
    Flow,
    FlowTUI,
    Graph,
    GraphCheckpointer,
    LiveTreeRenderer,
    LocalRuntime,
)
from rlmflow.graph.events import is_append

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
        "--cli",
        action="store_true",
        help="Use the plain REPL + live tree instead of the full-screen TUI.",
    )
    parser.add_argument("--no-viz", action="store_true", help="CLI only: disable live tree.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent tasks to run.",
    )
    parser.add_argument(
        "--max-steps-per-turn",
        type=int,
        default=240,
        help="TUI only: safety cap per submitted prompt before returning control.",
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
    runtime.register_tools(FILE_TOOLS)

    flow = Flow(
        build_client(args.model),
        llm_clients={"fast": build_client(args.fast_model)},
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        workers=args.max_concurrency,
    )

    try:
        if args.cli:
            _run_cli(flow, workdir, no_viz=args.no_viz)
        else:
            graph = _run_tui(flow, workdir, max_steps_per_turn=args.max_steps_per_turn)
            if graph is not None:
                print(f"\n{graph.result() or '(no result)'}\n")
                print(f"Graph checkpointed under {workdir / 'graph'}")
                print(f"Files written under {workdir}")
    finally:
        flow.close_repls()


def _run_tui(
    flow: Flow, workdir: Path, *, max_steps_per_turn: int | None
) -> Graph | None:
    ui = FlowTUI()
    checkpointer = GraphCheckpointer(workdir / "graph")

    async def drive(
        graph: Graph | None,
        *,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        until: Any = "done",
    ) -> Graph | None:
        # FlowTUI seeds ``graph`` on the first Send; later turns pass ``query``.
        if graph is None:
            return None

        boundary = until
        if until == "done" and max_steps_per_turn is not None:
            steps = 0

            def until_cap(event, current) -> bool:
                nonlocal steps
                if is_append(event):
                    steps += 1
                if current.finished:
                    return True
                return steps >= max_steps_per_turn

            boundary = until_cap

        async for event in flow.run_streaming(
            graph=graph, query=query, inputs=inputs, until=boundary
        ):
            ui.handle(event, graph)
            checkpointer.handle(event, graph)
        return graph

    try:
        return ui.run(drive)
    finally:
        checkpointer.close()
        ui.close()


def _run_cli(flow: Flow, workdir: Path, *, no_viz: bool) -> None:
    print("Agent ready. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        graph = Graph(query=query)
        consumers = ConsumerGroup(
            [
                LiveTreeRenderer(clear=not no_viz),
                GraphCheckpointer(workdir / "graph"),
            ]
        )

        async def drive() -> None:
            try:
                async for event in flow.run_streaming(graph=graph):
                    consumers.handle(event, graph)
            finally:
                consumers.close()

        asyncio.run(drive())

        print(f"\n{graph.result() or '(no result)'}\n")
        print(f"Graph checkpointed to {workdir / 'graph'}")
        print(f"Files written under {workdir}")


if __name__ == "__main__":
    main()
