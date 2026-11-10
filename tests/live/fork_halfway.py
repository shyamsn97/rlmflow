"""Fork a Claude Code root halfway through the filesystem needle.

The root runs until a node's code has launched its search workers. The run stops
there and the graph is forked at that node: the copy drops the workers and sets
the node to run again. Running the copy reruns that code in a rebuilt REPL,
which launches its own workers, then a fresh Claude Code session picks the work
up from the REPL. The original keeps its workers.

``test_live_modes.py`` runs it; by hand:
    python tests/live/fork_halfway.py
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "needle"))
from filesystem import NEEDLE_FILE_TOOLS, generate_haystack

from rlmflow import AgentConfig, Flow, Graph, Node, SubprocessRuntime
from rlmflow.consumers import GraphCheckpointer
from rlmflow.view.text import kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-files", type=int, default=200)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "_runs" / "mode2-fork-halfway"),
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
        root = g.start(
            f"There are {args.num_files} text files in haystack/. Exactly one line "
            "in one file matches `The magic number is <number>`. Find and return "
            "the number. Split the files into batches and give each batch to a "
            "searcher.",
        )

        def launched(step) -> bool:
            return step.agent.id == root.id and step.node.runs_code and bool(root.children())

        async def drive() -> None:
            checkpointer = GraphCheckpointer(args.out_dir)
            try:
                async for step in flow.run_streaming(g, until=launched, root_config=config):
                    checkpointer.handle(step, g)
                    print(f"{step.agent.path():<20} {kind(step.node)}")
                checkpointer.close()
                launch = root.nodes()[-1]
                workers = root.children()
                print(f"\n--- forking at {launch.id} after {len(workers)} workers launched ---\n")
                branch = g.fork(launch)
                checkpointer = GraphCheckpointer(f"{args.out_dir}-fork")
                async for step in flow.run_streaming(branch, root_config=config):
                    checkpointer.handle(step, branch)
                    print(f"{step.agent.path():<20} {kind(step.node)}")
            finally:
                checkpointer.close()
                await flow.aclose()
            report(g, branch, root, workers, answer)

        asyncio.run(drive())


def report(g: Graph, branch: Graph, root: Node, workers: list[Node], answer: str) -> None:
    copy = branch.agent(root.id)
    code = [node.code for node in copy.nodes() if node.runs_code]
    relaunched = copy.children()
    result = copy.final_result()
    kept = [worker.id in branch for worker in workers]
    print(f"\n{'=' * 40}")
    print(f"fork code runs:   {len(code)}")
    print(f"original workers: {len(workers)} (in the copy: {sum(kept)})")
    print(f"copy's launches:  {len(relaunched)}")
    print(f"original intact:  {root.children() == workers}")
    print(f"result:           {result}")
    print(f"actual answer:    {answer}")
    print(f"correct:          {answer in str(result)}")
    if answer not in str(result) or any(kept) or root.children() != workers:
        sys.exit(1)


if __name__ == "__main__":
    main()
