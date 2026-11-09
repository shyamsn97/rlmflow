"""Showcase the graph-backed Flow API.

This walks through the pieces that matter in the engine:

1. Node-by-node execution that advances a caller-owned graph.
2. Persisting a run with ``g.save()`` / ``Graph.load()``.
3. Latest-state inspection across agents.
4. Time travel: the graph after each agent and node is an earlier graph.
5. Graph summaries (``g.nodes()``, ``g.usage()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from rlmflow import (
    FILE_TOOLS,
    AgentConfig,
    Flow,
    Graph,
    GraphCheckpointer,
    LLMUsage,
    LocalRuntime,
    Node,
)
from rlmflow.view import replay
from rlmflow.view.text import kind

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream, example_run_dir  # noqa: E402

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
                'hello = launch_agent("Create hello.py")\n'
                'goodbye = launch_agent("Create goodbye.py")\n'
                "results = [\n"
                "    await hello,\n"
                "    await goodbye,\n"
                "]\n"
                'submit("\\n".join(results))\n'
                "```"
            )
        if "hello.py" in prompt:
            return (
                '```repl\nwrite_file("hello.py", "print(\\"hello\\")\\n")\nsubmit("hello.py")\n```'
            )
        if "goodbye.py" in prompt:
            return '```repl\nwrite_file("goodbye.py", "print(\\"goodbye\\")\\n")\nsubmit("goodbye.py")\n```'
        if "haiku" in prompt:
            return '```repl\nwrite_file("haiku.txt", "Calls fold into calls\\nNodes branch, wait, and then resume\\nFlow returns a leaf\\n")\nsubmit("wrote haiku.txt")\n```'
        return '```repl\nsubmit("ok")\n```'


def file_flow(workdir: Path, config: AgentConfig) -> Flow:
    """A Flow whose agents get the filesystem tools, running inside ``workdir``."""
    runtime = LocalRuntime(working_directory=workdir)
    return Flow(
        llm_clients={"demo": DemoLLM()},
        configs=[config],
        runtime=runtime,
        tools=[FILE_TOOLS],
    )


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


async def run(flow: Flow, g: Graph, out_dir: Path) -> None:
    """Stream the run, checkpointing periodically and printing every node as it lands."""
    async for step in checkpoint_stream(flow.run_streaming(g), g, out_dir):
        print(f"{g.path(step.agent)}  {kind(step.node)}")


async def gym_loop(flow: Flow, g: Graph, root: Node, out_dir: Path) -> list[float]:
    rewards: list[float] = []
    step = 0
    checkpoint = GraphCheckpointer(out_dir)
    try:
        while not g.status(root) == "done":
            async for item in flow.run_streaming(g, until="next"):
                checkpoint.handle(item, g)
            step += 1
            reward = 1.0 if g.status(root) == "done" else 0.0
            rewards.append(reward)
            print(f"step {step}: node={kind(g.tail(root))} reward={reward}")
    finally:
        checkpoint.close()
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=8)
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

    flow = file_flow(
        workdir, AgentConfig(model="demo", max_depth=args.max_depth, max_iters=args.max_iters)
    )

    banner("1. Step-by-step execution")
    g = Graph()
    root = g.start("Create hello.py and goodbye.py. Delegate each file.")
    asyncio.run(run(flow, g, workdir / "run"))
    print(f"\n{GREEN}Result:{RESET} {g.result(root)}")

    banner("2. Persistence — g.save() / Graph.load()")
    path = g.save(workdir / "run")
    loaded = Graph.load(path)
    print(
        f"Saved + reloaded {len(loaded.agents())} agents and "
        f"{len(loaded.nodes())} nodes from {path}"
    )

    banner("3. Latest state per agent")
    for agent in loaded.agents():
        print(f"  {loaded.path(agent)}: {loaded.status(agent)} at {kind(loaded.tail(agent))}")

    banner("4. Time travel — the graph after each step is an earlier graph")
    for idx, past in enumerate(replay(g)):
        latest = kind(past.tail(past.agent(root.id)))
        print(f"{CYAN}step {idx}{RESET}: root [{latest}]  agents={len(past.agents())}")

    banner("5. Graph summary")
    usage = g.usage()
    print(f"Agents:  {len(g.agents())}")
    print(f"Nodes:   {len(g.nodes())}")
    print(f"Tokens:  {usage.total:,} ({usage.input_tokens:,} in / {usage.output_tokens:,} out)")
    print(f"Final:   {kind(g.tail(root))}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir, AgentConfig(model="demo", max_depth=0, max_iters=args.max_iters))
    g3 = Graph()
    root3 = g3.start("Write a haiku about recursion to haiku.txt")
    rewards = asyncio.run(gym_loop(flow3, g3, root3, workdir / "gym-run"))
    print(f"{GREEN}Result:{RESET} {g3.result(root3)}")
    print(f"Total reward: {sum(rewards):.1f}")
    gym_path = g3.save(workdir / "gym-run")
    print(f"Gym run saved to {gym_path}")

    flow.runtime.close_repls()
    flow3.runtime.close_repls()

    banner("Done")


if __name__ == "__main__":
    main()
