"""Showcase the graph-backed Flow API.

This walks through the pieces that matter in the engine:

1. Node-by-node execution that advances a caller-owned graph.
2. Persisting a run with ``g.save()`` / ``Graph.load()``.
3. Latest-state inspection across agents.
4. Time travel: every log position is an earlier graph.
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
    AGENT_START,
    FILE_TOOLS,
    AgentConfig,
    Flow,
    Graph,
    GraphCheckpointer,
    LLMUsage,
    LocalRuntime,
    Node,
)

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


def file_flow(workdir: Path) -> Flow:
    """A Flow whose agents get the filesystem tools, running inside ``workdir``."""
    runtime = LocalRuntime(working_directory=workdir)
    return Flow(
        DemoLLM(),
        runtime=runtime,
        tools=[FILE_TOOLS],
    )


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


async def run(flow: Flow, g: Graph, out_dir: Path) -> list[int]:
    """Stream the run, checkpointing periodically and noting the log position of every node."""
    positions = [g.items()[-1].seq]
    async for step in checkpoint_stream(flow.run_streaming(g), g, out_dir):
        print(f"{g.path(step.agent)}  {step.node.type}")
        positions.append(step.node.seq)
    return positions


def as_of(g: Graph, seq: int) -> Graph:
    """The graph as it stood once item ``seq`` (and the edges that landed with it) was written."""
    items = g.items()
    later = [item.seq for item in items if isinstance(item, Node) and item.seq > seq]
    return g.as_of(later[0] - 1 if later else items[-1].seq)


async def gym_loop(flow: Flow, g: Graph, root: Node, out_dir: Path) -> list[float]:
    rewards: list[float] = []
    step = 0
    checkpoint = GraphCheckpointer(out_dir)
    try:
        while not g.terminal(root):
            async for item in flow.run_streaming(g, until="next"):
                checkpoint.handle(item, g)
            step += 1
            reward = 1.0 if g.terminal(root) else 0.0
            rewards.append(reward)
            print(f"step {step}: state={g.frontier(root).type} reward={reward}")
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

    flow = file_flow(workdir)
    config = AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters)

    banner("1. Step-by-step execution")
    g = Graph()
    root = g.create_agent("Create hello.py and goodbye.py. Delegate each file.", config=config)
    positions = asyncio.run(run(flow, g, workdir / "run"))
    print(f"\n{GREEN}Result:{RESET} {g.result(root)}")

    banner("2. Persistence — g.save() / Graph.load()")
    path = g.save(workdir / "run")
    loaded = Graph.load(path)
    print(
        f"Saved + reloaded {len(loaded.nodes(AGENT_START))} agents and "
        f"{len(loaded.nodes())} states from {path}"
    )

    banner("3. Latest state per agent")
    for agent in loaded.nodes(AGENT_START):
        print(f"  {loaded.path(agent)}: {loaded.frontier(agent).type}")

    banner("4. Time travel — any log position is an earlier graph")
    for idx, seq in enumerate(positions):
        past = as_of(g, seq)
        print(
            f"{CYAN}step {idx}{RESET}: root [{past.frontier(past.node(root.id)).type}]  "
            f"agents={len(past.nodes(AGENT_START))}"
        )

    banner("5. Graph summary")
    usage = g.usage()
    print(f"Agents:  {len(g.nodes(AGENT_START))}")
    print(f"States:  {len(g.nodes())}")
    print(f"Tokens:  {usage.total:,} ({usage.input_tokens:,} in / {usage.output_tokens:,} out)")
    print(f"Final:   {g.frontier(root).type}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir)
    g3 = Graph()
    root3 = g3.create_agent(
        "Write a haiku about recursion to haiku.txt",
        config=AgentConfig(max_depth=0, max_iters=args.max_iters),
    )
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
