"""Minimal needle-in-a-haystack search across many files.

This is the tiny `rflow` version of `examples/needle/filesystem.py`.
It keeps the graph caller-owned, runs filesystem tools through LocalRuntime, and
streams graph events while the agent searches.

Usage:
    python examples/needle/minimal_filesystem.py
    python examples/needle/minimal_filesystem.py --num-files 50
    python examples/needle/minimal_filesystem.py --out-dir examples/_runs/needle-minimal-filesystem
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import tempfile
from pathlib import Path

from rflow.clients import AnthropicClient, OpenAIClient
from rflow import FILE_TOOLS, Flow, Graph, LiveTreeRenderer, LocalRuntime


def generate_haystack(
    directory: Path, num_files: int = 100, lines_per_file: int = 80
) -> str:
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


def build_llm(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


async def run_example(args: argparse.Namespace) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(args.workdir) if args.workdir else Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        haystack_path = workdir / "haystack"
        haystack_path.mkdir(parents=True, exist_ok=True)
        for stale in haystack_path.glob("*.txt"):
            stale.unlink()

        answer = generate_haystack(
            haystack_path,
            num_files=args.num_files,
            lines_per_file=args.lines_per_file,
        )
        print(f"Generated {args.num_files} files in {haystack_path}")
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Live graph save: {out_dir}")

        runtime = LocalRuntime(working_directory=workdir)
        runtime.register_tools(FILE_TOOLS)
        flow = Flow(
            build_llm(args.model),
            max_depth=args.max_depth,
            max_iters=args.max_iters,
            runtime=runtime,
        )
        graph = Graph(
            query=(
                f"There are {args.num_files} text files in haystack/. Exactly one "
                "line in one file matches `The magic number is <number>`. Find "
                "and return the number. Use the filesystem tools. Split the file "
                "list into batches and delegate batches to subagents."
            )
        )
        renderer = LiveTreeRenderer(clear=not args.no_clear)

        async for event in flow.run_streaming(graph):
            renderer.handle(event, graph)
            if out_dir is not None:
                metadata = {
                    "example": "needle-minimal-filesystem",
                    "answer": answer,
                    "workdir": str(workdir),
                    "last_event_type": event.type,
                }
                if event.type == "append_node":
                    metadata["last_agent_id"] = event.agent_id
                    metadata["last_node_type"] = event.node_type
                elif event.type == "add_child":
                    metadata["last_agent_id"] = event.parent_agent_id
                    metadata["last_child_agent_id"] = event.child.agent_id
                elif event.type == "remove_child":
                    metadata["last_agent_id"] = event.parent_agent_id
                    metadata["last_child_agent_id"] = event.child_agent_id
                graph.save(out_dir, metadata=metadata)

        result = graph.result()
        if out_dir is not None:
            graph.save(
                out_dir,
                metadata={
                    "example": "needle-minimal-filesystem",
                    "answer": answer,
                    "workdir": str(workdir),
                    "result": result,
                    "correct": answer in result,
                },
            )
        return result, answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal rflow filesystem needle-in-a-haystack example"
    )
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--lines-per-file", type=int, default=80)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Print each tree update instead of redrawing the terminal.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to hold haystack/ (default: a temp dir).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "needle-minimal-filesystem"),
        help="Live-save the minimal run layout here.",
    )
    args = parser.parse_args()

    result, answer = asyncio.run(run_example(args))
    print(f"\n{'=' * 40}")
    print(f"Result:         {result}")
    print(f"Actual answer:  {answer}")
    print(f"Correct:        {answer in result}")


if __name__ == "__main__":
    main()
