"""Run one cell of the table: a task, a system, and some or all of its problems.

    python -m benchmarks.table.run oolong rlmflow --ids 2
    python -m benchmarks.table.run ale claude_code --minutes 30 --budget 5

Each problem gets a fresh work directory under ``--out``; its result lands in
``results.jsonl`` there, one line per problem, and a finished problem is skipped
on the next run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path

from benchmarks.table.systems import MODEL, SYSTEMS, Limits, run

TASKS = {"oolong": "benchmarks.table.oolong", "ale": "benchmarks.table.ale"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("system", choices=SYSTEMS)
    parser.add_argument("--ids", default=None, help="A count (first N) or comma-separated ids.")
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--budget", type=float, default=3.0, help="Dollar cap per problem.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="benchmarks/table/runs")
    args = parser.parse_args()

    task = importlib.import_module(TASKS[args.task])
    ids = task.ids()
    if args.ids:
        ids = ids[: int(args.ids)] if args.ids.isdigit() else args.ids.split(",")
    out = Path(args.out) / args.task / args.system
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {json.loads(line)["id"] for line in results.open()} if results.exists() else set()
    limits = Limits(minutes=args.minutes, budget_usd=args.budget)

    for task_id in ids:
        if task_id in done:
            continue
        workdir = Path(tempfile.mkdtemp(prefix=f"{task_id}-", dir=out)).resolve()
        prompt = task.prepare(task_id, workdir, limits)
        outcome = run(args.system, prompt, workdir, limits, args.model)
        record = {
            "id": task_id,
            "model": args.model,
            "limits": vars(limits),
            "workdir": str(workdir),
            "prompt": prompt,
            **outcome.to_dict(),
            **task.score(task_id, workdir, outcome),
        }
        with results.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        print(
            f"{task_id}: score {record['score']}  ${outcome.cost_usd:.2f}  "
            f"{outcome.seconds / 60:.1f} min  agents {outcome.agents}"
            + (f"  error {outcome.error}" if outcome.error else "")
        )


if __name__ == "__main__":
    main()
