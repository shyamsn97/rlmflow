"""A DeLM-style shared task board, as host functions the workers call as tools.

A fixed pool of worker agents shares one board: tasks, claims and results in a
few dicts behind a lock, in this process. Plain functions passed as ``tools``
run on the host, so every worker sees the same board. ``board_claim`` is atomic,
so exactly one worker gets each task. Workers post follow-up tasks as they find
them and stop when nothing is left to claim. There is no manager. The tools are
``@tool(cache=True)``, so each board call's result is saved on the node whose code
made it: the saved run records what every worker saw, and a rebuilt REPL replays
it instead of calling the board again.

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
import threading
from pathlib import Path

from rlmflow import AgentConfig, Flow, Graph, GraphCheckpointer, tool

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir  # noqa: E402

WORKER = """\
You are one worker on a shared task board; the other workers run in parallel and
there is no manager. Repeat until board_open() is empty:
1. Pick a task id from board_open() and claim it with board_claim(id, AGENTS.self.name).
   False means another worker has it: pick again.
2. Read board_results() for what helps, do the task in your REPL, and
   board_finish(id, result) with a compact, reusable result.
3. If you find concrete follow-up work the team still needs and no task covers,
   post it with board_post(new_id, task).
Then submit the ids you finished."""


class Board:
    """Tasks, claims and results, shared by every worker through the tools below."""

    def __init__(self) -> None:
        self.tasks: dict[str, str] = {}
        self.claims: dict[str, str] = {}
        self.results: dict[str, str] = {}
        self.lock = threading.Lock()

    def tools(self) -> list:
        lock, tasks, claims, results = self.lock, self.tasks, self.claims, self.results

        @tool(cache=True)
        def board_post(id: str, task: str) -> bool:
            """Post a task unless the id is taken."""
            with lock:
                return tasks.setdefault(id, task) == task

        @tool(cache=True)
        def board_claim(id: str, worker: str) -> bool:
            """Claim an open task for `worker`; False if another worker has it."""
            with lock:
                if id in claims or id not in tasks:
                    return False
                claims[id] = worker
                return True

        @tool(cache=True)
        def board_open() -> dict:
            """Tasks nobody has claimed: {id: task}."""
            with lock:
                return {k: v for k, v in tasks.items() if k not in claims}

        @tool(cache=True)
        def board_finish(id: str, result: str) -> None:
            """Record a claimed task's result."""
            with lock:
                results[id] = result

        @tool(cache=True)
        def board_results() -> dict:
            """Every finished task's result: {id: result}."""
            with lock:
                return dict(results)

        return [board_post, board_claim, board_open, board_finish, board_results]


def parse_task(value: str) -> tuple[str, str]:
    subject, separator, detail = value.partition("::")
    subject = subject.strip()
    detail = detail.strip() if separator else subject
    if not subject or not detail:
        raise argparse.ArgumentTypeError("tasks must be SUBJECT or SUBJECT::DETAIL")
    return subject, detail


def task_id(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.casefold()).strip("_") or "task"


async def run(flow: Flow, g: Graph, out_dir: str, config: AgentConfig) -> None:
    checkpointer = GraphCheckpointer(out_dir)
    try:
        async for item in flow.run_streaming(g, root_config=config):
            checkpointer.handle(item, g)
    finally:
        checkpointer.close()
        flow.runtime.close_repls()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--workers", type=int, default=2, help="Worker agents on the board.")
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

    board = Board()
    for subject, detail in args.task:
        board.tasks[task_id(subject)] = detail

    g = Graph()
    for i in range(args.workers):
        g.start(WORKER, name=f"worker_{i}")

    flow = Flow(workers=args.workers, tools=board.tools())
    asyncio.run(run(flow, g, args.out_dir, AgentConfig(model=args.model, max_iters=args.max_iters)))

    print("board:")
    for tid, task in board.tasks.items():
        print(f"  {tid} [{board.claims.get(tid, 'unclaimed')}]: {task}")
        print(f"    -> {board.results.get(tid)!r}")
    for worker in g.agents():
        print(f"{worker.name} finished: {worker.final_result()!r}")
    print(f"\nGraph saved to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
