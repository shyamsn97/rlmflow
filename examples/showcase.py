"""Showcase the Node-only Flow API.

This walks through the pieces that matter in the engine:

1. Node-by-Node execution that advances a caller-owned trajectory.
2. Persisting a run with ``persistence.save()`` / ``persistence.load()``.
3. Latest-state inspection across agents.
4. In-process history by keeping graph snapshots.
5. Node summary helpers (``render_tree(graph)``, ``graph.tokens()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from copy import deepcopy
from pathlib import Path

from rlmflow import (
    FILE_TOOLS,
    Flow,
    LLMUsage,
    LocalRuntime,
    Node,
    persistence,
    start,
)
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer
from rlmflow.view import LiveTreeRenderer, render_tree

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir  # noqa: E402

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class DemoLLM:
    """Deterministic LLM for an offline showcase."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        prompt = messages[-1]["content"].lower()
        if "hello.py" in prompt and "goodbye.py" in prompt:
            return (
                "```repl\n"
                "results = await launch_subagents([\n"
                '    {"name": "hello", "query": "Create hello.py"},\n'
                '    {"name": "goodbye", "query": "Create goodbye.py"},\n'
                "])\n"
                'done("\\n".join(results))\n'
                "```"
            )
        if "hello.py" in prompt:
            return '```repl\nwrite_file("hello.py", "print(\\"hello\\")\\n")\ndone("hello.py")\n```'
        if "goodbye.py" in prompt:
            return '```repl\nwrite_file("goodbye.py", "print(\\"goodbye\\")\\n")\ndone("goodbye.py")\n```'
        if "haiku" in prompt:
            return '```repl\nwrite_file("haiku.txt", "Calls fold into calls\\nNodes branch, wait, and then resume\\nFlow returns a leaf\\n")\ndone("wrote haiku.txt")\n```'
        return '```repl\ndone("ok")\n```'


def file_flow(workdir: Path, **kwargs) -> Flow:
    """A Flow whose agents get the filesystem tools, running inside ``workdir``."""
    runtime = LocalRuntime(working_directory=workdir)
    runtime.register_tools(FILE_TOOLS)
    return Flow(DemoLLM(), runtime=runtime, **kwargs)


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def node_count(graph: Node) -> int:
    return sum(1 for _node in graph.walk())


async def run(flow: Flow, graph: Node, no_viz: bool, out_dir: Path) -> list[Node]:
    history = [deepcopy(graph)]
    consumers = ConsumerGroup(
        [
            LiveTreeRenderer(clear=not no_viz),
            GraphCheckpointer(out_dir),
        ]
    )
    step = 0
    try:
        async for event in flow.run_streaming(graph):
            step += 1
            history.append(deepcopy(graph))
            if no_viz:
                print(f"-- event {step}: {event.type} --")
            consumers.handle(event)
    finally:
        consumers.close()
    return history


async def gym_loop(flow: Flow, graph: Node, out_dir: Path) -> list[float]:
    rewards: list[float] = []
    checkpointer = GraphCheckpointer(out_dir)
    step = 0
    while not graph.finished():
        async for _event in flow.run_streaming(graph, until="next"):
            pass
        checkpointer.save(graph)
        step += 1
        current = graph.tail()
        reward = 1.0 if graph.finished() else 0.0
        rewards.append(reward)
        kind = current.type if current else "empty"
        print(f"step {step}: state={kind} reward={reward}")
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(example_run_dir("showcase")),
        help="working dir + saved run (default: examples/_runs/showcase/)",
    )
    args = parser.parse_args()

    workdir = Path(args.out_dir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    flow = file_flow(workdir, max_depth=args.max_depth, max_iters=args.max_iters)

    banner("1. Step-by-step execution")
    graph = start(query="Create hello.py and goodbye.py. Delegate each file.")
    history = asyncio.run(run(flow, graph, args.no_viz, workdir / "run"))
    final = history[-1]
    print(f"\n{GREEN}Result:{RESET} {final.agent_result()}")

    banner("2. Persistence — persistence.save() / persistence.load()")
    path = persistence.save(final, workdir / "run")
    loaded = persistence.load(path)
    print(
        f"Saved + reloaded {len(loaded.agent_ids())} agents and "
        f"{node_count(loaded)} states from {path}"
    )
    print(render_tree(loaded))

    banner("3. Latest state per agent")
    for agent_id in loaded.agent_ids():
        print(f"  {agent_id}: {loaded.tail(agent_id).type}")

    banner("4. Time travel — kept snapshots")
    for idx, snapshot in enumerate(history):
        current = snapshot.tail()
        kind = current.type if current else "empty"
        print(f"{CYAN}step {idx}{RESET}: root [{kind}]  agents={len(snapshot.agent_ids())}")

    banner("5. Node summary")
    inp, out = final.tokens()
    print(f"Agents:  {len(final.agent_ids())}")
    print(f"States:  {node_count(final)}")
    print(f"Tokens:  {inp + out:,} ({inp:,} in / {out:,} out)")
    print(f"Final:   {final.tail().type}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir, max_depth=0, max_iters=args.max_iters)
    graph3 = start(query="Write a haiku about recursion to haiku.txt")
    rewards = asyncio.run(gym_loop(flow3, graph3, workdir / "gym-run"))
    print(f"{GREEN}Result:{RESET} {graph3.agent_result()}")
    print(f"Total reward: {sum(rewards):.1f}")
    gym_path = persistence.save(graph3, workdir / "gym-run")
    print(f"Gym run saved to {gym_path}")

    flow.runtime.close_repls(final.trajectory_id)
    flow3.runtime.close_repls(graph3.trajectory_id)

    banner("Done")


if __name__ == "__main__":
    main()
