"""The filesystem needle, with any harness as the root and any harness as the workers.

One Python-API script for checking every harness end to end: the root and its
``searcher`` workers can each be rlmflow's REPL loop, a Claude Code session, or
a pi session. ``--all`` runs every combination your machine can run and prints
a table.

Usage:
    python examples/needle/harnesses/run.py                              # repl root, repl workers
    python examples/needle/harnesses/run.py --root claude_code
    python examples/needle/harnesses/run.py --root repl --workers claude_code
    python examples/needle/harnesses/run.py --root pi --workers pi --pi-provider anthropic
    python examples/needle/harnesses/run.py --all --num-files 60
    python examples/needle/harnesses/run.py --root claude_code --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from rlmflow import AgentConfig, DockerRuntime, Flow, Graph, SubprocessRuntime
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer, LiveGraphTree
from rlmflow.llm import client_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filesystem import NEEDLE_FILE_TOOLS, generate_haystack

HARNESSES = ("repl", "claude_code", "pi")
RUNS = Path(__file__).resolve().parents[2] / "_runs" / "needle-harnesses"
TASK = (
    "There are {n} text files in haystack/. Exactly one line in one file matches "
    "`The magic number is <number>`. Find and return the number. Split the files "
    "into batches and give each batch to a `searcher` agent; run them in parallel."
)


@dataclass
class Outcome:
    root: str
    workers: str
    correct: bool
    result: object
    seconds: float
    agents: int
    saved: Path


def missing(harness: str, args: argparse.Namespace) -> str | None:
    """Why this machine can't run ``harness``, or ``None``."""
    if (
        harness == "repl"
        and not os.environ.get("OPENAI_API_KEY")
        and not args.model.startswith("claude")
    ):
        return "OPENAI_API_KEY is not set"
    if harness == "claude_code" and importlib.util.find_spec("claude_agent_sdk") is None:
        return "pip install 'rlmflow[claude]'"
    if harness == "pi" and shutil.which(args.pi_command) is None:
        return f"`{args.pi_command}` is not on PATH (npm i -g @mariozechner/pi-coding-agent)"
    return None


def config(harness: str, args: argparse.Namespace, **fields) -> AgentConfig:
    """An ``AgentConfig`` for ``harness``, with the model and options that harness takes."""
    if harness == "claude_code":
        return AgentConfig(harness=harness, model=args.claude_model, **fields)
    if harness == "pi":
        options = {
            "command": args.pi_command,
            **({"provider": args.pi_provider} if args.pi_provider else {}),
        }
        return AgentConfig(harness=harness, model=args.pi_model, harness_options=options, **fields)
    return AgentConfig(harness=harness, **fields)


def run_one(root_harness: str, worker_harness: str, args: argparse.Namespace) -> Outcome:
    saved = Path(args.out_dir) / f"{root_harness}-{worker_harness}"
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "haystack").mkdir()
        random.seed(args.seed)
        answer = generate_haystack(workdir / "haystack", num_files=args.num_files)

        searcher = config(
            worker_harness,
            args,
            name="searcher",
            description="searches one batch of files and returns the magic number, or None",
            max_depth=0,
            max_iters=args.max_iters,
        )
        if worker_harness == "repl":
            searcher = replace(searcher, model=args.worker_model)
        flow = Flow(
            client_for(args.model),
            configs=[searcher],
            runtime=DockerRuntime(args.docker_image, working_directory=workdir)
            if args.docker_image
            else SubprocessRuntime(working_directory=workdir),
            tools=NEEDLE_FILE_TOOLS,
        )
        root_config = config(root_harness, args, max_depth=1, max_iters=args.max_iters)

        g = Graph()
        root = g.create_agent(TASK.format(n=args.num_files))
        consumers = ConsumerGroup([GraphCheckpointer(saved)])
        if not args.no_viz:
            consumers.append(
                LiveGraphTree(title=f"needle: {root_harness} root, {worker_harness} workers")
            )

        async def drive() -> None:
            try:
                async for step in flow.run_streaming(g, root_config=root_config):
                    consumers.handle(step, g)
            finally:
                consumers.close()
                await flow.aclose()

        started = time.monotonic()
        asyncio.run(drive())
        result = g.result(root)
        return Outcome(
            root_harness,
            worker_harness,
            answer in str(result),
            result,
            time.monotonic() - started,
            len(g.agents()),
            saved,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", choices=HARNESSES, default="repl", help="the root agent's harness"
    )
    parser.add_argument(
        "--workers", choices=HARNESSES, default="repl", help="the searchers' harness"
    )
    parser.add_argument(
        "--all", action="store_true", help="every root x workers combination that can run here"
    )
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument(
        "--model", default="gpt-5-mini", help="a repl root's model (the Flow's default)"
    )
    parser.add_argument("--worker-model", default="gpt-5-nano", help="repl searchers' model")
    parser.add_argument(
        "--claude-model",
        default="default",
        help="Claude Code's model; default: your Claude setting",
    )
    parser.add_argument("--pi-model", default="default", help="pi's --model; default: pi's own")
    parser.add_argument("--pi-provider", default=None, help="pi's --provider")
    parser.add_argument("--pi-command", default="pi")
    parser.add_argument(
        "--docker-image", default=None, help="run every REPL in this image, e.g. rlmflow:local"
    )
    parser.add_argument("--no-viz", action="store_true", help="no live agent tree")
    parser.add_argument(
        "--out-dir", default=str(RUNS), help="each run is saved in <out-dir>/<root>-<workers>/"
    )
    args = parser.parse_args()

    pairs = (
        [(r, w) for r in HARNESSES for w in HARNESSES] if args.all else [(args.root, args.workers)]
    )
    outcomes: list[Outcome] = []
    for root_harness, worker_harness in pairs:
        reason = missing(root_harness, args) or missing(worker_harness, args)
        if reason:
            print(f"skip {root_harness} root, {worker_harness} workers: {reason}")
            continue
        print(f"\n>>> {root_harness} root, {worker_harness} workers")
        outcome = run_one(root_harness, worker_harness, args)
        outcomes.append(outcome)
        print(f"result {outcome.result!r}, correct {outcome.correct}, saved to {outcome.saved}")

    if not outcomes:
        sys.exit("nothing ran")
    print(f"\n{'root':<12} {'workers':<12} {'correct':<8} {'agents':<7} seconds")
    for o in outcomes:
        print(f"{o.root:<12} {o.workers:<12} {o.correct!s:<8} {o.agents:<7} {o.seconds:.0f}")
    sys.exit(0 if all(o.correct for o in outcomes) else 1)


if __name__ == "__main__":
    main()
