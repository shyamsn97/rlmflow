"""The three systems the table compares, behind one call.

Each system works on a task in a directory and leaves its answer there, and
each task scores what's in the directory, so every system is scored the same
way. All three get the same task text and the same model:

- ``claude_code``: ``claude -p TASK``.
- ``claude_code_rlmflow``: ``claude -p "/rlmflow:run TASK"`` with the rlmflow
  plugin; its workers are Claude Code sessions (``flows.claude_code_workers``).
- ``rlmflow``: rlmflow's REPL harness as the root, and its workers
  (``flows.repl_workers``).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-5-5"
SYSTEMS = ("claude_code", "claude_code_rlmflow", "rlmflow")

#: Dollars per million tokens: Sonnet 5.5 list prices, from ALE-Bench's ``calc_cost.py``.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.2, "cache_write": 2.5},
}


@dataclass(frozen=True)
class Limits:
    """The same for every system: wall-clock minutes, and a dollar cap as a safety net."""

    minutes: float
    budget_usd: float


@dataclass
class Outcome:
    system: str
    seconds: float
    cost_usd: float
    agents: int = 1
    text: str = ""
    error: str | None = None
    run_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run(system: str, task: str, workdir: Path, limits: Limits, model: str = MODEL) -> Outcome:
    if system == "claude_code":
        return claude(system, task, workdir, limits, model)
    if system == "claude_code_rlmflow":
        return claude_rlmflow(task, workdir, limits, model)
    if system == "rlmflow":
        return rlmflow(task, workdir, limits, model)
    raise ValueError(f"unknown system {system!r}; known: {', '.join(SYSTEMS)}")


def claude(
    system: str,
    prompt: str,
    workdir: Path,
    limits: Limits,
    model: str,
    args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> Outcome:
    """One headless Claude Code session, billed to the Claude Code login."""
    environ = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    environ.update(env or {})
    argv = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--max-budget-usd",
        str(limits.budget_usd),
        *args,
    ]
    start = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=workdir,
        env=environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=limits.minutes * 60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        return Outcome(system, time.monotonic() - start, 0.0, error="timeout; cost unknown")
    seconds = time.monotonic() - start
    try:
        reply = json.loads(stdout)
    except json.JSONDecodeError:
        return Outcome(system, seconds, 0.0, error=(stderr or stdout)[-2000:] or "no output")
    return Outcome(
        system,
        seconds,
        float(reply.get("total_cost_usd") or 0.0),
        text=str(reply.get("result") or ""),
        error=str(reply.get("subtype")) if reply.get("is_error") else None,
    )


def claude_rlmflow(task: str, workdir: Path, limits: Limits, model: str) -> Outcome:
    """Claude Code as the root of an rlmflow run; the cost adds the workers' sessions."""
    from rlmflow import Graph

    plugin = REPO / "rlmflow" / "plugin"
    outcome = claude(
        "claude_code_rlmflow",
        f"/rlmflow:run {task}",
        workdir,
        limits,
        model,
        args=("--plugin-dir", str(plugin)),
        env={
            "RLMFLOW_FLOW": "benchmarks.table.flows:claude_code_workers",
            "RLMFLOW_BENCH_MODEL": model,
            "RLMFLOW_BENCH_CLAUDE_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
            "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
            "PATH": os.pathsep.join([str(REPO / ".venv" / "bin"), os.environ["PATH"]]),
        },
    )
    runs = sorted((workdir / ".rlmflow" / "runs").glob("*"))
    if runs:
        g = Graph.load(runs[-1])
        workers = [agent for agent in g.agents() if agent.config_name is not None]
        outcome.cost_usd += g.usage(workers).cost_usd if workers else 0.0
        outcome.agents = len(g.agents())
        outcome.run_dir = str(runs[-1])
    return outcome


def rlmflow(task: str, workdir: Path, limits: Limits, model: str) -> Outcome:
    """rlmflow's REPL harness as the root, billed to ``ANTHROPIC_API_KEY``."""
    from benchmarks.table.flows import repl_workers
    from rlmflow import Graph, GraphCheckpointer

    os.environ["RLMFLOW_BENCH_MODEL"] = model
    flow = repl_workers(workdir, max_budget=tokens_for(limits.budget_usd, model))
    g = Graph()
    root = g.start(task)
    run_dir = workdir / ".rlmflow" / "runs" / "repl"
    checkpointer = GraphCheckpointer(str(run_dir))
    error = None

    async def drive() -> None:
        async for item in flow.run_streaming(g):
            checkpointer.handle(item, g)

    start = time.monotonic()
    try:
        asyncio.run(asyncio.wait_for(drive(), timeout=limits.minutes * 60))
    except TimeoutError:
        error = "timeout"
    finally:
        checkpointer.close()
        flow.runtime.close_repls()
    result = g.result(root)
    return Outcome(
        "rlmflow",
        time.monotonic() - start,
        cost(g.usage(), model),
        agents=len(g.agents()),
        text=result if isinstance(result, str) else json.dumps(result, default=str),
        error=error,
        run_dir=str(run_dir),
    )


def cost(usage, model: str) -> float:
    price = PRICES[model]
    return (usage.input_tokens * price["input"] + usage.output_tokens * price["output"]) / 1e6


def tokens_for(budget_usd: float, model: str) -> int:
    """A token budget worth ``budget_usd``, priced as input; rlmflow's budget is soft."""
    return int(budget_usd / PRICES[model]["input"] * 1e6)


__all__ = ["MODEL", "SYSTEMS", "Limits", "Outcome", "run"]
