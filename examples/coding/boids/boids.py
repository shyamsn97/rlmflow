"""Boids, built by three peer agents with no manager.

``peer1``, ``peer2`` and ``peer3`` are top-level agents in one graph, writing
into one directory. Each gets the same product spec and is told only that the
other two exist: they split the work, agree on interfaces and report bugs to
each other with ``AGENTS[name].send(...)``, and each writes only the files it took on.
Their tools are generic: the file tools, their own Python REPL, and
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

from rlmflow import (  # noqa: E402
    FILE_TOOLS,
    MESSAGE,
    AgentConfig,
    Flow,
    Graph,
    LocalRuntime,
    attach_image,
)

PEERS = ("peer1", "peer2", "peer3")

SPEC = """\
Create a polished browser-based boids simulation in plain HTML, CSS, and
JavaScript, written into the current directory. Use no build tools or external
libraries. Render 2000 boids on a full-viewport 2D canvas over a dark
background, at 60fps. Use spatial partitioning rather than all-pairs scanning,
four visually distinct species, fading trails, orientation-aware shapes, and an
on-canvas FPS readout."""

TEAM = """\
You are {name}, one of three peers ({peers}) building this together in one
shared directory. There is no manager. Split the work between you, agree on
interfaces by messaging each other, and write only the files you took on.

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


def peer_config(args: argparse.Namespace) -> AgentConfig:
    if args.harness == "claude_code":
        return AgentConfig(
            harness="claude_code",
            model=args.claude_model,
            max_depth=0,
            max_iters=args.max_iters,
            harness_options={"sdk": {"permission_mode": "acceptEdits"}},
        )
    return AgentConfig(model=args.model, max_depth=0, max_iters=args.max_iters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", choices=("repl", "claude_code"), default="repl")
    parser.add_argument("--model", default="gpt-5", help="the peers' model with --harness repl")
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
        tools=[FILE_TOOLS, open_page_tool(run_dir)],
        workers=len(PEERS),
    )
    g = Graph()
    for name in PEERS:
        others = ", ".join(peer for peer in PEERS if peer != name)
        g.add_agent(TEAM.format(name=name, peers=others, spec=SPEC), name=name)

    started = time.perf_counter()
    try:
        asyncio.run(print_run(flow, g, run_dir / "graph", width=8, root_config=peer_config(args)))
    finally:
        flow.runtime.close_repls()

    validation = validate_project(run_dir, browser=True, screenshot=run_dir / "boids-smoke.png")
    write_report(validation, run_dir / "validation.json")
    peers = {peer.id: peer for peer in g.agents()}
    results = {peer.name: g.result(peer) for peer in peers.values()}
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
    for message in g.nodes(MESSAGE):
        sender = peers.get(message.data.get("sender"))
        if sender is not None:
            text = " ".join(str(message.data.get("text", "")).split())[:160]
            print(f"  {sender.name} -> {g.owner(message).name}: {text}")
    print("\nvalidation:")
    for name, check in validation["checks"].items():
        print(f"  {'PASS' if check['passed'] else 'FAIL'} {name}: {str(check['detail'])[:200]}")
    print(f"graph: {run_dir / 'graph'}")


if __name__ == "__main__":
    main()
