"""Needle in a haystack across many files.

Generates many files of random noise. One file contains a magic string. The
agent uses the standard file tools to find it, delegating the search in
parallel across batches.

Usage:
    python examples/needle/filesystem.py
    python examples/needle/filesystem.py --no-viz
    python examples/needle/filesystem.py --in-process-local
    python examples/needle/filesystem.py --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import sys
import tempfile
from pathlib import Path

from rlmflow import (
    FILE_TOOLS,
    AgentConfig,
    DockerRuntime,
    Flow,
    LocalRuntime,
    SubprocessRuntime,
)
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer, LiveGraphTree
from rlmflow.llm import client_for

examples_dir = next(path for path in Path(__file__).resolve().parents if path.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

def generate_haystack(directory: Path, num_files: int = 500, lines_per_file: int = 200) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    answer = "".join(random.choices(string.digits, k=7))
    needle_file = random.randint(0, num_files - 1)
    needle_line = random.randint(0, lines_per_file - 1)

    for i in range(num_files):
        lines = []
        for j in range(lines_per_file):
            if i == needle_file and j == needle_line:
                lines.append(f"The magic number is {answer}")
            else:
                n = random.randint(3, 8)
                lines.append(" ".join(random.choice(words) for _ in range(n)))
        (directory / f"file_{i:04d}.txt").write_text("\n".join(lines))

    print(f"Needle in file_{needle_file:04d}.txt line {needle_line}")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Needle in a haystack across many files")
    parser.add_argument("--num-files", type=int, default=500)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument(
        "--in-process-local",
        action="store_true",
        help=(
            "Use in-process LocalRuntime for debugging. This serializes REPL "
            "blocks that need cwd/env isolation, so it is not true parallel "
            "filesystem execution."
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to hold haystack/ and run in (default: a temp dir).",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true", help="Disable the live agent tree")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "needle-filesystem"),
        help="Save the final run here (default: examples/_runs/needle-filesystem/).",
    )
    args = parser.parse_args()

    if args.docker_image:
        print(f">>> DOCKER RUNTIME  image={args.docker_image}")
    elif args.in_process_local:
        print(">>> LOCAL RUNTIME (in-process; REPL blocks may serialize)")
    else:
        print(">>> SUBPROCESS RUNTIME")

    tmp = None
    if args.workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = Path(tmp.name)
    else:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        haystack_path = workdir / "haystack"
        haystack_path.mkdir(parents=True, exist_ok=True)
        for stale in haystack_path.glob("*.txt"):
            stale.unlink()
        answer = generate_haystack(haystack_path, num_files=args.num_files)
        print(f"Generated {args.num_files} files in {haystack_path}")

        if args.docker_image:
            runtime = DockerRuntime(args.docker_image, working_directory=workdir)
        elif args.in_process_local:
            runtime = LocalRuntime(working_directory=workdir)
        else:
            runtime = SubprocessRuntime(working_directory=workdir)

        llm_clients = None
        if args.fast_model:
            llm_clients = {"fast": client_for(args.fast_model)}

        flow = Flow(
            client_for(args.model),
            llm_clients=llm_clients,
            runtime=runtime,
            tools=[FILE_TOOLS],
            root_config=AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
        )

        root = flow.start(
            f"There are {args.num_files} text files in haystack/. "
            "Exactly one line in one file matches the pattern "
            "`The magic number is <number>`. Find and return the number. "
            "There are too many files to search manually, so split the work "
            "into batches."
        )
        out_dir = Path(args.out_dir)
        consumers = ConsumerGroup()
        if not args.no_viz:
            consumers.append(LiveGraphTree(title="Needle filesystem search"))
        if args.out_dir:
            consumers.append(GraphCheckpointer(out_dir))

        async def drive() -> None:
            try:
                async for node in flow.run_streaming(root):
                    consumers.handle(node)
            finally:
                consumers.close()
                await flow.aclose()

        asyncio.run(drive())

        print(f"\n{'=' * 40}")
        print(f"Result:         {root.result()}")
        print(f"Actual answer:  {answer}")
        print(f"Correct:        {answer in root.result()}")
        print(f"Run saved to    {out_dir}")

    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
