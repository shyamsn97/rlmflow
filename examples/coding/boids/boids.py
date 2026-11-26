"""Boids, built by three peer agents with no manager.

``peer1``, ``peer2`` and ``peer3`` are top-level agents in one graph, writing
into one directory. Each gets the same product spec and is told only that the
other two exist. They coordinate on a shared ``TaskBoard``: they post tasks,
claim them (the first claim wins), publish the interfaces they define, and
finish what they claimed; they report bugs to each other with
``AGENTS[name].send(...)``. Each board call is saved on the node that made it,
so the board is part of the graph and its figure draws it as a lane.
Their other tools are generic: the file tools, their own Python REPL, and
``open_page``, which opens a page in headless Chrome, shows them screenshots of
it over a few seconds, and returns its console output. Nothing checks the app
for them.

After the run, the app is validated (local assets, JavaScript syntax, a clean
headless-browser load) and can be recorded:

    python examples/coding/boids/boids.py
    python examples/coding/boids/boids.py --harness claude_code
    python examples/coding/boids/render_simulation.py --run-dir examples/_runs/coding/boids
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

from browser import look
from validate import validate_project, write_report

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import print_run  # noqa: E402

from rlmflow import Agent, ClaudeCode, Flow, Graph, LocalRuntime, attach_image  # noqa: E402
from rlmflow.tools.board import TaskBoard  # noqa: E402
from rlmflow.tools.files import FILE_TOOLS  # noqa: E402

PEERS = ("peer1", "peer2", "peer3")

SPEC = """\
Create a polished browser-based boids simulation in plain HTML, CSS, and
JavaScript, written into the current directory. Use no build tools or external
libraries. Render 2000 boids on a full-viewport 2D canvas, at 60fps. Use
spatial partitioning rather than all-pairs scanning, four species, small
orientation-aware shapes, and an on-canvas FPS readout.

Keep it calm and easy on the eyes: clear the canvas every frame, with no trails,
glow, additive blending or flashing. Use a soft dark background (a deep slate,
not pure black) and four muted, harmonious colors of similar brightness, so the
species read apart without any of them glaring. Motion should be smooth and
unhurried."""

TEAM = """\
You are {name}, one of three peers ({peers}) building this together in one
shared directory. There is no manager: you coordinate through a shared task
board, from your REPL.

- `board_read()` shows every task (status, owner, result) and every finding.
  Read it first, and again before you decide what to do next.
- One peer plans. Before you post any other task, call
  `board_post("plan", "split the app into tasks and publish the interface")`
  and `board_claim("plan")`. If either returns `ok: False`, someone else is
  planning: read the board until `plan` is done, then claim tasks from it.
  If you won `plan`, publish the interface, post the tasks, and finish `plan`.
- `board_post(id, task)` adds a task anyone can take; post what the app needs
  that nobody has posted yet, one task per file.
- `board_claim(id)` takes an open task. Its `ok` is False if another peer got
  it first; pick another. Write only the files of tasks you claimed.
- `board_publish(topic, text)` shares an interface the others build against:
  globals, function signatures, data layout. Publish before you rely on it.
- `board_finish(id, result)` marks your task done, with a one-line result.

You can also message a peer directly with `AGENTS[name].send(...)`, for a bug
in their file. Finish when every task on the board is done and the app runs.

{spec}"""


def open_page_tool(run_dir: Path):
    screenshots_dir = run_dir / ".screenshots"

    def open_page(path: str = "index.html", seconds: float = 3, screenshots: int = 2) -> dict:
        """Open an HTML file in headless Chrome and watch it for `seconds` (at most 10).

        You are shown `screenshots` (at most 4) evenly spaced screenshots of the page.
        Returns its console output and the screenshot paths.
        """
        console, shots = look(
            run_dir / path,
            screenshots_dir,
            seconds=min(float(seconds), 10.0),
            screenshots=max(0, min(int(screenshots), 4)),
        )
        for shot in shots:
            attach_image(shot)
        return {"console": console, "screenshots": [str(shot) for shot in shots]}

    return open_page


def peer_agent(args: argparse.Namespace) -> Agent:
    if args.harness == "claude_code":
        return ClaudeCode(
            args.claude_model,
            max_depth=0,
            max_iters=args.max_iters,
            options={"permission_mode": "acceptEdits"},
        )
    return Agent(
        args.model,
        max_depth=0,
        max_iters=args.max_iters,
        reasoning_effort=args.effort,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", choices=("repl", "claude_code"), default="repl")
    parser.add_argument("--model", default="gpt-6-sol", help="the peers' model with --harness repl")
    parser.add_argument("--effort", default="low", help="repl peers' reasoning effort")
    parser.add_argument("--claude-model", default=None, help="Claude Code's model")
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=examples_dir / "_runs" / "coding" / "boids")
    args = parser.parse_args()

    run_dir = args.out_dir.resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    flow = Flow(
        runtime=LocalRuntime(working_directory=run_dir),
        tools=[FILE_TOOLS, TaskBoard(), open_page_tool(run_dir)],
        agents={"peer": peer_agent(args)},
        workers=len(PEERS),
    )
    g = Graph()
    for name in PEERS:
        others = ", ".join(peer for peer in PEERS if peer != name)
        g.start(TEAM.format(name=name, peers=others, spec=SPEC), name=name, agent="peer")

    started = time.perf_counter()
    try:
        asyncio.run(print_run(flow, g, run_dir / "graph", width=8))
    finally:
        flow.close()

    validation = validate_project(run_dir, browser=True, screenshot=run_dir / "boids-smoke.png")
    write_report(validation, run_dir / "validation.json")
    peers = {peer.id: peer for peer in g.agent_roots()}
    results = {peer.name: peer.final_result() for peer in peers.values()}
    (run_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "harness": args.harness,
                "model": args.claude_model if args.harness == "claude_code" else args.model,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                "usage": asdict(g.usage()),
                "results": results,
                "validation_passed": validation["passed"],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )

    print("\nresults:")
    for name, result in results.items():
        print(f"  {name}: {result}")
    print("\nmessages between peers:")
    for message in g.nodes():
        sender = peers.get(message.producer)
        if sender is not None and message.role == "user":
            text = " ".join(message.text.split())[:160]
            print(f"  {sender.name} -> {message.root.name}: {text}")
    board = TaskBoard.state(g)
    print(f"\nboard ({board['upto']} writes):")
    for task_id, task in board["tasks"].items():
        print(f"  {task_id}: {task['status']} by {task['owner']}: {task['result']}")
    for finding in board["findings"]:
        print(f"  {finding['by']} published {finding['topic']!r}")
    (run_dir / "board.json").write_text(json.dumps(board, indent=2) + "\n")
    print("\nvalidation:")
    for name, check in validation["checks"].items():
        print(f"  {'PASS' if check['passed'] else 'FAIL'} {name}: {str(check['detail'])[:200]}")
    print(f"graph: {run_dir / 'graph'}")


if __name__ == "__main__":
    main()
