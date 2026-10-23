"""A DeLM-style shared task board where the board is the graph.

Every task is an agent. Seed tasks are top-level agents, follow-up work is a
child launched with ``launch_subagent``, and the shared context is ``AGENTS``:
each agent can read every other agent's status and result. ``Flow(workers=N)``
is the pool of workers that claims ready tasks, and the saved graph holds the
whole board.

Example:
    python examples/collaboration/shared_board.py \
      --model gpt-5-mini --workers 3 \
      --task "API::Propose the public API and its invariants." \
      --task "Tests::Design concurrency and termination tests."
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from rlmflow import AgentConfig, AgentStart, Flow, Graph, GraphCheckpointer, reads
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir  # noqa: E402

BOARD = """\
You are one task on a shared board. There is no manager: other tasks run in
parallel, and AGENTS shows every task's status and result.

1. Before starting, read AGENTS.print_graph(show_results=True) and reuse any
   finished results that help.
2. Solve your task in your REPL.
3. If you find concrete follow-up work the team still needs, and no task on the
   board covers it, launch_subagent(task=...) for it. Don't await it unless your
   own answer depends on it.
4. submit(...) a compact, reusable result.

Your task ({subject}):
{detail}
"""


def parse_task(value: str) -> tuple[str, str]:
    subject, separator, detail = value.partition("::")
    subject = subject.strip()
    detail = detail.strip() if separator else subject
    if not subject or not detail:
        raise argparse.ArgumentTypeError("tasks must be SUBJECT or SUBJECT::DETAIL")
    return subject, detail


def task_name(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.casefold()).strip("_") or "task"


async def run(flow: Flow, g: Graph, out_dir: str) -> None:
    checkpointer = GraphCheckpointer(out_dir)
    try:
        async for item in flow.run_streaming(g, close_repls=True):
            checkpointer.handle(item, g)
    finally:
        checkpointer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--workers", type=int, default=2, help="Tasks that run at once.")
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        type=parse_task,
        metavar="SUBJECT[::DETAIL]",
        help="Seed task; repeat for independent initial work.",
    )
    parser.add_argument("--max-iters", type=int, default=12)
    parser.add_argument("--out-dir", default=str(example_run_dir("shared-board")))
    args = parser.parse_args()

    g = Graph()
    for subject, detail in args.task:
        g.create_agent(
            BOARD.format(subject=subject, detail=detail),
            config=AgentConfig(name=task_name(subject), max_depth=1, max_iters=args.max_iters),
        )

    flow = Flow(client_for(args.model), workers=args.workers, use_agent_tree=True)
    asyncio.run(run(flow, g, args.out_dir))

    print("board:")
    for agent in g.nodes(AgentStart):
        print(f"  {reads.path(g, agent)}: {reads.result(g, agent)!r}")
    print(f"\nGraph saved to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
