"""Needle in a haystack with your Claude Code session as the root, through the plugin.

The scripted version of the plugin run in the main README: it builds the
haystack in a fresh directory, runs ``claude -p "/rlmflow:run TASK"`` there with
the rlmflow plugin loaded, then loads the run the plugin saved and checks the
answer. The searchers are rlmflow REPL agents by default, or Claude Code
sessions with ``--workers claude_code``.

Usage:
    python examples/needle/claude_plugin.py
    python examples/needle/claude_plugin.py --workers claude_code --num-files 40
    python examples/needle/claude_plugin.py --gif   # also render the run (rlmflow[image])

Needs the ``claude`` CLI, signed in. REPL searchers need ``OPENAI_API_KEY``
(or a ``claude-*`` ``--worker-model`` and ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from filesystem import NEEDLE_FILE_TOOLS, TASK, generate_haystack

from rlmflow import AgentConfig, Flow, Graph, LocalRuntime

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parents[1] / "rlmflow" / "plugin"
RUNS = HERE.parent / "_runs" / "needle-plugin"


def build_flow() -> Flow:
    """The Flow the plugin's ``rlmflow serve`` loads (``RLMFLOW_FLOW``): one ``searcher`` config."""
    harness = os.environ["NEEDLE_WORKER_HARNESS"]
    model = os.environ.get("NEEDLE_WORKER_MODEL") or None
    if harness == "claude_code":
        # Claude Code keeps its login token out of the hooks that start the serve.
        token = os.environ.get("NEEDLE_CLAUDE_TOKEN")
        if token and not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = token
        searcher = AgentConfig(
            name="searcher",
            harness="claude_code",
            model=model,
            description="searches one batch of files and returns the magic number, or None",
            harness_options={
                "command": shutil.which("claude"),
                "sdk": {"permission_mode": "bypassPermissions"},
            },
        )
        tools = None
    else:
        searcher = AgentConfig(
            name="searcher",
            model=model,
            description="searches one batch of files and returns the magic number, or None",
            max_depth=0,
            max_iters=15,
        )
        tools = NEEDLE_FILE_TOOLS
    return Flow(configs=[searcher], tools=tools, runtime=LocalRuntime(working_directory=Path.cwd()))


def run_claude(task: str, workdir: Path, args: argparse.Namespace) -> dict:
    """One headless Claude Code session running ``/rlmflow:run``; its JSON reply."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY" or args.keep_api_key}
    env.update(
        RLMFLOW_FLOW="claude_plugin:build_flow",
        NEEDLE_WORKER_HARNESS=args.workers,
        NEEDLE_WORKER_MODEL=args.worker_model
        if args.workers == "repl"
        else args.claude_model or "",
        NEEDLE_CLAUDE_TOKEN=os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
        PYTHONPATH=os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")])),
        # The plugin's hooks call `rlmflow`; use this interpreter's.
        PATH=os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]),
    )
    argv = [
        "claude",
        "-p",
        f"/rlmflow:run {task}",
        "--plugin-dir",
        str(PLUGIN),
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--max-budget-usd",
        str(args.budget),
        *(["--model", args.claude_model] if args.claude_model else []),
    ]
    done = subprocess.run(
        argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=args.minutes * 60
    )
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        sys.exit(f"claude gave no JSON reply:\n{(done.stderr or done.stdout)[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", choices=("repl", "claude_code"), default="repl")
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--worker-model", default="gpt-5-mini", help="REPL searchers' model")
    parser.add_argument(
        "--claude-model", default=None, help="Claude Code's model; default: your Claude setting"
    )
    parser.add_argument("--budget", type=float, default=2.0, help="the root session's dollar cap")
    parser.add_argument("--minutes", type=float, default=15)
    parser.add_argument(
        "--keep-api-key",
        action="store_true",
        help="let claude bill ANTHROPIC_API_KEY instead of your Claude Code login",
    )
    parser.add_argument("--gif", action="store_true", help="render the run to <out-dir>/run.gif")
    parser.add_argument("--out-dir", default=str(RUNS), help="the run is saved in <out-dir>/run/")
    args = parser.parse_args()

    if shutil.which("claude") is None:
        sys.exit("the `claude` CLI is not on PATH")
    if args.workers == "repl" and not (
        os.environ.get("OPENAI_API_KEY") or args.worker_model.startswith("claude")
    ):
        sys.exit("REPL searchers need OPENAI_API_KEY (or a claude-* --worker-model)")

    out = Path(args.out_dir)
    task = TASK.format(n=args.num_files)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "haystack").mkdir()
        random.seed(args.seed)
        answer = generate_haystack(workdir / "haystack", num_files=args.num_files)

        print(f">>> Claude Code root (plugin), {args.workers} searchers")
        started = time.monotonic()
        reply = run_claude(task, workdir, args)
        seconds = time.monotonic() - started

        runs = sorted((workdir / ".rlmflow" / "runs").glob("*"))
        if not runs:
            sys.exit(f"the plugin saved no run; claude said: {reply.get('result')!r}")
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(runs[-1], out / "run")

    g = Graph.load(out / "run")
    root = g.roots()[0]
    result = root.final_result()
    correct = answer in str(result) or answer in str(reply.get("result", ""))
    workers = [a for a in g.agents() if a.parent() is not None]
    failed = [a for a in workers if str(a.final_result()).startswith("[child failed")]
    print(f"\nresult   {result!r}")
    print(f"correct  {correct} (needle {answer})")
    print(f"agents   {len(g.agents())} ({len(workers)} searchers, {len(failed)} failed)")
    for agent in failed:
        print(f"  {agent.name}: {str(agent.final_result())[:200]}")
    print(f"seconds  {seconds:.0f}")
    print(f"cost     root ${float(reply.get('total_cost_usd') or 0):.2f}", end="")
    print(f", searchers ${g.usage(workers).cost_usd:.2f}" if workers else "")
    print(f"saved to {out / 'run'}  (rlmflow view show {out / 'run'})")

    if args.gif:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rlmflow",
                "render",
                "gif",
                str(out / "run"),
                str(out / "run.gif"),
            ],
            check=True,
        )
    # The root can grep the haystack itself, so a right answer alone doesn't show the searchers worked.
    sys.exit(0 if correct and workers and len(failed) < len(workers) else 1)


if __name__ == "__main__":
    main()
