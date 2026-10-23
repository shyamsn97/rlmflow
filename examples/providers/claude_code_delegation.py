"""Use Claude Code as the host-configured backend for delegated children.

The root remains an rlmflow agent. Its ordinary ``launch_subagent(...)`` calls
run Claude Code jobs; no backend name is exposed to the model.

Prerequisites:
    pip install "rlmflow[openai]"
    export OPENAI_API_KEY=...
    claude  # authenticate once if needed

Run:
    python examples/providers/claude_code_delegation.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rlmflow import AgentConfig, ClaudeCodeBackend, Flow, Graph, LocalRuntime, reads
from rlmflow.llm.openai import OpenAIClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, print_run  # noqa: E402

CHILD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "files_changed", "concerns"],
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {
            "type": "array",
            "items": {"type": "string"},
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ROOT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "worker_results"],
    "properties": {
        "summary": {"type": "string"},
        "worker_results": {
            "type": "array",
            "items": CHILD_SCHEMA,
        },
    },
}


def prepare_workspace(path: Path) -> None:
    """Create a harmless tiny project when the target directory is empty."""
    path.mkdir(parents=True, exist_ok=True)
    calculator = path / "calculator.py"
    if not calculator.exists():
        calculator.write_text(
            "def add(left, right):\n"
            "    return left + right\n\n"
            "def divide(left, right):\n"
            "    return left / right\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delegate child work to Claude Code without changing launch_subagent"
    )
    parser.add_argument("--model", default="gpt-5-mini", help="Root rlmflow model")
    parser.add_argument(
        "--workdir",
        default=str(example_run_dir("claude-code-workspace")),
        help="Shared directory Claude Code children may inspect and edit",
    )
    parser.add_argument(
        "--out-dir",
        default=str(example_run_dir("claude-code-delegation")),
        help="Saved rlmflow graph",
    )
    parser.add_argument("--claude-command", default="claude")
    parser.add_argument("--max-iters", type=int, default=12)
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    prepare_workspace(workdir)

    flow = Flow(
        OpenAIClient(args.model),
        runtime=LocalRuntime(working_directory=workdir),
        backends={
            "claude-code": ClaudeCodeBackend(
                command=args.claude_command,
                permission_mode="acceptEdits",
                allowed_tools=("Read", "Glob", "Grep", "Write", "Edit"),
            )
        },
    )
    g = Graph()
    root = g.create_agent(
        "Launch two independent child agents before awaiting either one. "
        "Child one must review calculator.py and add useful docstrings without "
        "changing behavior. Child two must inspect calculator.py and create "
        "test_calculator.py using only the Python standard library. Pass clear, "
        "non-overlapping tasks to launch_subagent, await both handles together, "
        "then submit a concise summary and both structured worker results.",
        config=AgentConfig(
            max_depth=1,
            max_iters=args.max_iters,
            output_schema=ROOT_SCHEMA,
            child_config={"backend": "claude-code", "output_schema": CHILD_SCHEMA},
        ),
    )
    out_dir = Path(args.out_dir)
    asyncio.run(print_run(flow, g, out_dir))

    print("\nResult:")
    print(json.dumps(reads.result(g, root), indent=2))
    print("\nWorkspace:")
    for path in sorted(workdir.iterdir()):
        if path.is_file():
            print(f"- {path.name}")
    print(f"\nRun saved to {out_dir}")


if __name__ == "__main__":
    main()
