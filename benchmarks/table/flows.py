"""The Flows the two rlmflow systems run: one model, workers of the root's own kind.

``claude_code_workers`` is what the plugin's serve loads (``RLMFLOW_FLOW``), so a
Claude Code root launches Claude Code workers. Claude Code keeps its login token
(``CLAUDE_CODE_OAUTH_TOKEN``) out of the hooks that start the serve, so the
runner passes it as ``RLMFLOW_BENCH_CLAUDE_TOKEN`` for the workers.
``repl_workers`` is the REPL harness all the way down.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rlmflow import FILE_TOOLS, AgentConfig, Flow, LocalRuntime


def model() -> str:
    from benchmarks.table.systems import MODEL

    return os.environ.get("RLMFLOW_BENCH_MODEL", MODEL)


def claude_code_workers() -> Flow:
    token = os.environ.get("RLMFLOW_BENCH_CLAUDE_TOKEN")
    if token and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return Flow(
        configs=[
            AgentConfig(
                name="default",
                harness="claude_code",
                model=model(),
                description="A Claude Code session with its own tools, in the same directory.",
                harness_options={
                    "command": shutil.which("claude"),
                    "sdk": {"permission_mode": "bypassPermissions"},
                },
            )
        ],
        runtime=LocalRuntime(working_directory=Path.cwd()),
    )


def repl_workers(workdir: Path, *, max_budget: int | None = None) -> Flow:
    return Flow(
        configs=[
            AgentConfig(
                name="default",
                model=model(),
                description="An agent with a Python REPL and file tools, in the same directory.",
                max_iters=40,
                max_budget=max_budget,
            )
        ],
        tools=[FILE_TOOLS],
        runtime=LocalRuntime(working_directory=workdir),
    )
