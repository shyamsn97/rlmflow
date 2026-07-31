"""Generate synthetic Node snapshots and open the minimal viewer.

No LLM or runtime needed. This builds a small recursive graph timeline using
the Node-only API and renders it with the lightweight viewer adapter.

Run:
    python examples/view_demo.py
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

from rlmflow import (
    DoneOutput,
    ErrorOutput,
    ExecOutput,
    LLMOutput,
    Node,
    SupervisingOutput,
    UserQuery,
    start,
)
from rlmflow.view import open_viewer

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import save_example_graph  # noqa: E402


def child(agent_id: str, query: str) -> UserQuery:
    return UserQuery(agent_id=agent_id, content=query)


def snapshots() -> list[Node]:
    root = start("Create a boids simulation in plain HTML + JS")
    out = [deepcopy(root)]

    plan = root.attach(
        LLMOutput(
            content="I'll split this into files and delegate each part.",
            code="results = await launch_subagents([...])",
        )
    )
    supervisor = plan.attach(
        SupervisingOutput(
            output="delegated file work",
            waiting_on=["root.index_html", "root.style_css", "root.script_js"],
        )
    )
    index = supervisor.attach(child("root.index_html", "Write index.html"))
    styles = supervisor.attach(child("root.style_css", "Write style.css"))
    script = supervisor.attach(child("root.script_js", "Write script.js"))
    out.append(deepcopy(root))

    index.attach(DoneOutput(result="Created index.html"))
    styles.attach(DoneOutput(result="Created style.css"))
    script_plan = script.attach(LLMOutput(content="Splitting script.js into smaller pieces."))
    script_supervisor = script_plan.attach(
        SupervisingOutput(
            output="delegated script work",
            waiting_on=[
                "root.script_js.boids_core",
                "root.script_js.renderer",
                "root.script_js.controls",
            ],
        )
    )
    core = script_supervisor.attach(child("root.script_js.boids_core", "Core boids"))
    renderer = script_supervisor.attach(child("root.script_js.renderer", "Canvas renderer"))
    controls = script_supervisor.attach(child("root.script_js.controls", "UI controls"))
    out.append(deepcopy(root))

    core.attach(DoneOutput(result="Implemented flocking"))
    renderer.attach(DoneOutput(result="Implemented renderer"))
    controls.attach(
        ErrorOutput(
            error="missing event handler",
            output="The first controls attempt forgot keyboard input.",
        )
    )
    out.append(deepcopy(root))

    controls.tail().attach(ExecOutput(output="Retried controls with keydown/keyup listeners."))
    controls.tail().attach(DoneOutput(result="Implemented controls"))
    script.tail().attach(DoneOutput(result="Created script.js"))
    out.append(deepcopy(root))

    root.tail().attach(DoneOutput(result="Created boids simulation files"))
    out.append(deepcopy(root))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal viewer demo.")
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    graphs = snapshots()
    print(f"Generated {len(graphs)} minimal graph snapshots.")
    save_example_graph(graphs[-1], "view-demo")
    if args.no_launch:
        for index, graph in enumerate(graphs):
            print(f"snapshot {index}: {graph.tail().type}")
        return
    print("Launching viewer...")
    open_viewer(graphs)


if __name__ == "__main__":
    main()
