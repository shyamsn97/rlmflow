"""Showcase the Node-only Flow API.

This walks through the pieces that matter in the engine:

1. Node-by-Node execution that advances a caller-owned trajectory.
2. Persisting a run with ``persistence.save()`` / ``persistence.load()``.
3. Latest-state inspection across agents.
4. In-process history by keeping tree snapshots.
5. Node summary helpers (``root.walk()``, ``root.tokens()``).
6. Gym-style stepping with a scalar reward.

Usage:
    python examples/showcase.py
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
    AgentConfig,
    AgentStart,
    Flow,
    LLMUsage,
    LocalRuntime,
    Node,
    persistence,
)

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
                "hello = await launch_subagent("
                '"Create hello.py", model="default", name="hello")\n'
                "goodbye = await launch_subagent("
                '"Create goodbye.py", model="default", name="goodbye")\n'
                "results = [\n"
                "    await hello.wait_for_result(),\n"
                "    await goodbye.wait_for_result(),\n"
                "]\n"
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


def file_flow(workdir: Path, config: AgentConfig) -> Flow:
    """A Flow whose agents get the filesystem tools, running inside ``workdir``."""
    runtime = LocalRuntime(working_directory=workdir)
    return Flow(DemoLLM(), runtime=runtime, tools=FILE_TOOLS, root_config=config)


def banner(msg: str) -> None:
    print(f"\n{BOLD}{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}{RESET}\n")


def node_count(root: Node) -> int:
    return sum(1 for _node in root.walk())


def agents(root: Node) -> list[AgentStart]:
    return [node for node in root.walk() if isinstance(node, AgentStart)]


async def run(flow: Flow, root: AgentStart, out_dir: Path) -> list[AgentStart]:
    """Stream the run, saving and snapshotting the tree after every node."""
    history = [deepcopy(root)]
    async for node in flow.run_streaming(root):
        print(f"{node.parent_agent.config.path}  {node.type}")
        root.save(out_dir)
        history.append(deepcopy(root))
    return history


async def gym_loop(flow: Flow, root: AgentStart, out_dir: Path) -> list[float]:
    rewards: list[float] = []
    step = 0
    while not root.terminal:
        async for _node in flow.run_streaming(root, until="next"):
            pass
        root.save(out_dir)
        step += 1
        reward = 1.0 if root.terminal else 0.0
        rewards.append(reward)
        print(f"step {step}: state={root.frontier.type} reward={reward}")
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-depth", type=int, default=2)
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

    flow = file_flow(workdir, AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters))

    banner("1. Step-by-step execution")
    root = flow.start("Create hello.py and goodbye.py. Delegate each file.")
    history = asyncio.run(run(flow, root, workdir / "run"))
    final = history[-1]
    print(f"\n{GREEN}Result:{RESET} {final.result()}")

    banner("2. Persistence — persistence.save() / persistence.load()")
    path = persistence.save(final, workdir / "run")
    loaded = persistence.load(path)
    print(
        f"Saved + reloaded {len(agents(loaded))} agents and {node_count(loaded)} states from {path}"
    )

    banner("3. Latest state per agent")
    for agent in agents(loaded):
        print(f"  {agent.config.path}: {agent.frontier.type}")

    banner("4. Time travel — kept snapshots")
    for idx, snapshot in enumerate(history):
        print(
            f"{CYAN}step {idx}{RESET}: root [{snapshot.frontier.type}]  "
            f"agents={len(agents(snapshot))}"
        )

    banner("5. Node summary")
    usage = final.tokens()
    print(f"Agents:  {len(agents(final))}")
    print(f"States:  {node_count(final)}")
    print(f"Tokens:  {usage.total:,} ({usage.input_tokens:,} in / {usage.output_tokens:,} out)")
    print(f"Final:   {final.frontier.type}")

    banner("6. Gym-style loop")
    flow3 = file_flow(workdir, AgentConfig(max_depth=0, max_iters=args.max_iters))
    root3 = flow3.start("Write a haiku about recursion to haiku.txt")
    rewards = asyncio.run(gym_loop(flow3, root3, workdir / "gym-run"))
    print(f"{GREEN}Result:{RESET} {root3.result()}")
    print(f"Total reward: {sum(rewards):.1f}")
    gym_path = persistence.save(root3, workdir / "gym-run")
    print(f"Gym run saved to {gym_path}")

    flow.runtime.close_repls()
    flow3.runtime.close_repls()

    banner("Done")


if __name__ == "__main__":
    main()
