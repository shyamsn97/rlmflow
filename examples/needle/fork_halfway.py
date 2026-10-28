"""Fork a Claude Code root halfway through the filesystem needle.

The root runs until a cell has launched its search workers. The run stops
there and the root is forked at that cell. The fork starts a fresh Claude Code
session on the rebuilt REPL, so it has to pick the work up from ``notes`` and
the worker handles already in its REPL instead of launching the search again.

Usage:
    python examples/needle/fork_halfway.py
"""

from __future__ import annotations

import argparse
import asyncio
import random
import tempfile
from pathlib import Path

from filesystem import NEEDLE_FILE_TOOLS, generate_haystack

from rlmflow import CODE, DONE, AgentConfig, Flow, Graph, Node, SubprocessRuntime
from rlmflow.consumers import GraphCheckpointer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-files", type=int, default=200)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "needle-fork-halfway"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "haystack").mkdir()
        answer = generate_haystack(workdir / "haystack", num_files=args.num_files)
        flow = Flow(
            configs=[
                AgentConfig(
                    name="searcher",
                    model=args.fast_model,
                    description="cheap model for searching one batch of files",
                    max_depth=0,
                    max_iters=15,
                )
            ],
            runtime=SubprocessRuntime(working_directory=workdir),
            tools=NEEDLE_FILE_TOOLS,
        )
        config = AgentConfig(harness="claude_code")
        g = Graph()
        root = g.create_agent(
            f"There are {args.num_files} text files in haystack/. Exactly one line "
            "in one file matches `The magic number is <number>`. Find and return "
            "the number. Split the files into batches and give each batch to a "
            "searcher.",
        )

        def launched(step) -> bool:
            return step.agent.id == root.id and step.node.type == CODE and bool(g.children(root))

        async def drive() -> None:
            checkpointer = GraphCheckpointer(args.out_dir)
            try:
                async for step in flow.run_streaming(g, until=launched, root_config=config):
                    checkpointer.handle(step, g)
                    print(f"{g.path(step.agent):<20} {step.node.type}")
                cell = g.frontier(root)
                workers = g.children(root)
                print(f"\n--- forking at {cell.id} after {len(workers)} workers launched ---\n")
                fork = g.fork(cell, name="fork")
                g.append(root, Node(DONE, result="[superseded by fork]"))
                async for step in flow.run_streaming(g, root_config=config):
                    checkpointer.handle(step, g)
                    print(f"{g.path(step.agent):<20} {step.node.type}")
            finally:
                checkpointer.close()
                await flow.aclose()
            report(g, fork, workers, answer)

        asyncio.run(drive())


def report(g: Graph, fork: Node, workers: list[Node], answer: str) -> None:
    cells = [node["code"] for node in g.nodes(CODE, agent=fork.id) if node.get("code")]
    relaunched = g.children(fork)
    result = g.result(fork)
    print(f"\n{'=' * 40}")
    print(f"fork cells:       {len(cells)}")
    print(f"read notes:       {any('notes' in code for code in cells)}")
    print(f"workers reused:   {len(workers)}, new launches: {len(relaunched)}")
    print(f"result:           {result}")
    print(f"actual answer:    {answer}")
    print(f"correct:          {answer in str(result)}")


if __name__ == "__main__":
    main()
