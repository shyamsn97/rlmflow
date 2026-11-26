"""Autoresearch on rlmflow: a planner launches implementers that each train a variant on Modal.

Each implementer works in its own copy of the task files (rlmflow's default
workspaces), edits ``train.py`` and calls ``submit_trial``. The trial's row is
the tool's saved return value, so the graph is the ledger: no trial
directories, no JSONL to keep in sync. The host loop checkpoints the graph and
stops the planner once the submissions are spent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from rlmflow import (
    Agent,
    Budget,
    Checkpoint,
    ClaudeCode,
    Flow,
    Graph,
    LocalRuntime,
    current_repl,
    tool,
)
from rlmflow.view.text import kind as node_kind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modal_runner import ModalConfig, preflight, submit  # noqa: E402

HARNESSES = ("claude_code", "repl")
TASK_FILES = ("README.md", "prepare.py", "train.py", "program.md", "pyproject.toml", "uv.lock")
BASELINES = HERE / "baselines"


def baseline(gpu: str) -> dict[str, Any]:
    """The unmodified train.py's result on ``gpu``; training is fixed-time, so it depends on the GPU."""
    path = BASELINES / f"{gpu}.json"
    if not path.exists():
        raise SystemExit(f"no baseline for {gpu}: score the seed on it into {path}")
    return json.loads(path.read_text())


PLANNER = """\
You plan autoresearch trials. `program.md` (your `task_instructions` input) is the task;
`baseline` is the unmodified train.py's result. Lower `val_bpb` is better.

Each round: read `trials()` (best first, with log tails), form hypotheses that cite a number
from a log (loss still falling? lr? steps? VRAM headroom?), then launch several implementers
at once and wait for them all:

    launch_agent("Implement: ...", agent="implementer", hypothesis="...",
                 workspace=row["commit"])  # build on a trial; omit to start from baseline

A plateau means change direction, not stop. Keep going until `trials()` says no
submissions are left, then `submit` a short summary of what worked.
"""

IMPLEMENTER = """\
You implement one autoresearch trial. Your `hypothesis` input says what to try. Your
directory is your own copy of the task: edit only `train.py`, changing just what the
hypothesis needs. Then `row = submit_trial(hypothesis, change="lr 3e-4 -> 1e-3")`, where
`change` names your edit in a few words for the progress plot; it trains on a GPU for several
minutes. If `row["status"]` is "crashed", read `row["stdout_tail"]`, fix train.py and submit
once more. Finish with `submit(row)`.
"""


def trial_rows(g: Graph) -> list[dict[str, Any]]:
    """Every submitted trial, oldest first: the saved return values of ``submit_trial``."""
    return [
        {"n": i, "agent": node.root.name, **call["value"]}
        for i, (node, call) in enumerate(
            (node, call)
            for node in g.nodes()
            for call in node.calls
            if call.get("tool") == "submit_trial" and isinstance(call.get("value"), dict)
        )
    ]


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (r.get("val_bpb") is None, r.get("val_bpb") or 0))


def research_tools(modal: ModalConfig, max_submissions: int, run_id: str) -> list[Any]:
    @tool(cache=True)
    def submit_trial(hypothesis: str, change: str = "") -> dict[str, Any]:
        """Train your directory's train.py on Modal; returns the row (val_bpb, status, commit)."""
        repl = current_repl()
        if len(trial_rows(repl.graph)) >= max_submissions:
            raise RuntimeError(f"all {max_submissions} submissions are used")
        commit = repl.commit()
        started = time.time()
        result = submit(modal, path=repl.path, slug=repl.root.name, n=0, run_id=run_id)
        keep = ("val_bpb", "status", "num_steps", "total_tokens_M", "peak_vram_mb", "num_params_M")
        return {
            **{key: result.get(key) for key in keep},
            "hypothesis": hypothesis,
            "change": change,
            "commit": commit,
            "stdout_tail": str(result.get("stdout_tail", ""))[-2000:],
            "seconds": round(time.time() - started),
            "ts": time.time(),
        }

    @tool()
    def trials() -> dict[str, Any]:
        """Submitted trials best first, and how many submissions are left."""
        rows = trial_rows(current_repl().graph)
        return {"left": max_submissions - len(rows), "trials": ranked(rows)}

    return [submit_trial, trials]


def run(args: argparse.Namespace) -> None:
    seed = baseline(args.gpu)
    modal = ModalConfig(gpu=args.gpu, parallel=args.parallel, timeout_s=args.modal_timeout_s)
    print("[autoresearch] checking Modal...", preflight(modal), flush=True)

    files = args.out / "files"
    files.mkdir(parents=True, exist_ok=True)
    for name in TASK_FILES:
        if (HERE / name).exists():
            shutil.copyfile(HERE / name, files / name)

    # Claude Code's shell stops commands after 2 minutes; a trial blocks for the Modal job.
    timeout_ms = str((args.modal_timeout_s + 300) * 1000)
    os.environ.setdefault("BASH_DEFAULT_TIMEOUT_MS", timeout_ms)
    os.environ.setdefault("BASH_MAX_TIMEOUT_MS", timeout_ms)

    def make_agent(harness: str, **fields: Any) -> Agent:
        if harness == "claude_code":
            return ClaudeCode(
                args.claude_model, options={"permission_mode": "acceptEdits"}, **fields
            )
        return Agent(args.model, **fields)

    planner = make_agent(args.planner, max_depth=1, prompt={"autoresearch": PLANNER})
    implementer = make_agent(
        args.implementer,
        max_depth=0,
        description="edits train.py in its own copy and trains it once",
        prompt={"autoresearch": IMPLEMENTER},
    )

    runtime = LocalRuntime(working_directory=files, repl_timeout=args.modal_timeout_s + 300)
    flow = Flow(
        agents={"planner": planner, "implementer": implementer},
        runtime=runtime,
        tools=research_tools(modal, args.max_submissions, args.out.name),
        workers=args.parallel,
    )

    g = flow.start(
        "planner",
        f"Run autoresearch: {args.max_submissions} trials, {args.parallel} at a time.",
        task_instructions=(HERE / "program.md").read_text(),
        baseline=seed,
    )
    budget = Budget(seconds=args.minutes * 60 if args.minutes else None, usd=args.max_usd)

    async def drive() -> None:
        try:
            async for node in flow.stream(g, hooks=[budget, Checkpoint(args.out / "graph")]):
                spent = g.usage().cost_usd
                print(f"{node.root.path():<24} {node_kind(node):<12} ${spent:.2f}", flush=True)
                rows = trial_rows(g)
                # The planner took a whole turn with nothing left to submit and didn't finish.
                if (
                    node.root_id == g.root.id
                    and g.root.status() != "done"
                    and len(rows) >= args.max_submissions
                    and (node.started_at or 0) > max(row["ts"] for row in rows)
                ):
                    await flow.stop(g.root, reason="all submissions used")
                write_ledger(rows, args.out)
        finally:
            await flow.aclose()

    asyncio.run(drive())
    rows = ranked(trial_rows(g))
    print(g.root.final_result())
    print(f"Claude Code spend: ${g.usage().cost_usd:.2f}")
    if rows:
        print(f"best val_bpb {rows[0]['val_bpb']} ({rows[0]['hypothesis']})")
    print(f"graph: {args.out / 'graph'}   plot: python {HERE / 'plot_progress.py'} {args.out}")


def write_ledger(rows: list[dict[str, Any]], out: Path) -> None:
    """An export for ``plot_progress.py``; the graph is the record."""
    lines = [json.dumps({**row, "slug": row["agent"]}) for row in rows]
    (out / "ledger.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planner", choices=HARNESSES, default="claude_code")
    p.add_argument("--implementer", choices=HARNESSES, default="claude_code")
    p.add_argument("--model", default="gpt-6-sol", help="for repl agents")
    p.add_argument("--claude-model", default=None, help="for Claude Code agents; default: its own")
    p.add_argument("--gpu", default="A10G", help="one with a baseline in baselines/")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--max-submissions", type=int, default=16)
    p.add_argument("--minutes", type=float, default=None, help="stop everything after this long")
    p.add_argument("--max-usd", type=float, default=None, help="Claude Code dollars for the run")
    p.add_argument("--modal-timeout-s", type=int, default=1200)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.out is None:
        args.out = HERE.parent / "_runs" / "autoresearch" / time.strftime("%Y%m%d-%H%M%S")
    args.out = args.out.resolve()
    if args.parallel < 1 or args.max_submissions < 1:
        raise SystemExit("--parallel and --max-submissions must be at least 1")
    return args


if __name__ == "__main__":
    run(parse_args())
