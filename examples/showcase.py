"""Showcase the Graph-centric Flow API.

This walks through the pieces that matter in the engine:

1. Event-by-event execution that advances a caller-owned ``Graph``.
2. Persisting a run with ``graph.save()`` / minimal ``Graph.load()``.
3. Latest-state inspection across agents.
4. In-process history by keeping graph snapshots.
5. Graph summary helpers (``render_tree(graph)``, ``graph.tokens()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
    python examples/showcase.py --no-viz
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from copy import deepcopy
from pathlib import Path

from rflow.minimal import (
    FILE_TOOLS,
    Flow,
    Graph,
    LLMUsage,
    LiveTreeRenderer,
    LocalRuntime,
    render_tree,
)

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


def node_count(graph: Graph) -> int:
    return sum(len(agent.nodes) for agent in graph.walk())


async def run(flow: Flow, graph: Graph, no_viz: bool) -> list[Graph]:
    history = [deepcopy(graph)]
    renderer = LiveTreeRenderer(clear=not no_viz)
    step = 0
    async for event in flow.run_streaming(graph):
        step += 1
        history.append(deepcopy(graph))
        if no_viz:
            print(f"-- event {step}: {event.type} --")
        renderer.handle(event, graph)
    return history


async def gym_loop(flow: Flow, graph: Graph) -> list[float]:
    rewards: list[float] = []
    step = 0
    while not graph.finished:
        await flow.step(graph if step == 0 else None)
        step += 1
        current = graph.current()
        reward = 1.0 if graph.finished else 0.0
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
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "showcase"),
        help="working dir + saved run (default: examples/_runs/showcase/)",
    )
    args = parser.parse_args()

    workdir = Path(args.out_dir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    flow = file_flow(workdir, max_depth=args.max_depth, max_iters=args.max_iters)

    banner("1. Step-by-step execution")
    graph = Graph(query="Create hello.py and goodbye.py. Delegate each file.")
    history = asyncio.run(run(flow, graph, args.no_viz))
    final = history[-1]
    print(f"\n{GREEN}Result:{RESET} {final.result()}")

    banner("2. Persistence — graph.save() / Graph.load()")
    path = final.save(workdir / "run")
    loaded = Graph.load(path)
    print(
        f"Saved + reloaded {len(loaded.agents)} agents and "
        f"{node_count(loaded)} states from {path}"
    )
    print(render_tree(loaded))

    banner("3. Latest state per agent")
    for aid, sub in loaded.agents.items():
        current = sub.current()
        label = current.type if current else "(empty)"
        print(f"  {aid}: {label}")

    banner("4. Time travel — kept snapshots")
    for idx, snapshot in enumerate(history):
        current = snapshot.current()
        kind = current.type if current else "empty"
        print(
            f"{CYAN}step {idx}{RESET}: root [{kind}]  "
            f"agents={len(snapshot.agents)}"
        )

    banner("5. Graph summary")
    inp, out = final.tokens()
    print(f"Agents:  {len(final.agents)}")
    print(f"States:  {node_count(final)}")
    print(f"Tokens:  {inp + out:,} ({inp:,} in / {out:,} out)")
    print(f"Final:   {final.current().type if final.current() else '(empty)'}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir, max_depth=0, max_iters=args.max_iters)
    graph3 = Graph(query="Write a haiku about recursion to haiku.txt")
    rewards = asyncio.run(gym_loop(flow3, graph3))
    print(f"{GREEN}Result:{RESET} {graph3.result()}")
    print(f"Total reward: {sum(rewards):.1f}")
    gym_path = graph3.save(workdir / "gym-run")
    print(f"Gym run saved to {gym_path}")

    flow.close_repls(final.graph_id)
    flow3.close_repls(graph3.graph_id)

    banner("Done")


if __name__ == "__main__":
    main()
