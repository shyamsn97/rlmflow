"""Generate synthetic minimal Graph snapshots and open the minimal viewer.

No LLM or runtime needed. This builds a small recursive graph timeline using
``rflow.Graph`` and renders it with the lightweight viewer adapter.

Run:
    python examples/view_demo.py
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

from rflow import (
    DoneOutput,
    ErrorOutput,
    ExecOutput,
    Graph,
    LLMOutput,
    SupervisingOutput,
    UserQuery,
    open_viewer,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import save_example_graph  # noqa: E402


def child(agent_id: str, query: str, parent: str) -> Graph:
    graph = Graph(
        agent_id=agent_id,
        query=query,
        depth=parent.count(".") + 1,
        parent_agent_id=parent,
    )
    graph.commit(UserQuery(content=query))
    return graph


def snapshots() -> list[Graph]:
    root = Graph(query="Create a boids simulation in plain HTML + JS")
    root.commit(UserQuery(content=root.query))
    out = [deepcopy(root)]

    root.commit(
        LLMOutput(
            content="I'll split this into files and delegate each part.",
            code="results = await launch_subagents([...])",
        )
    )
    root.children["root.index_html"] = child(
        "root.index_html",
        "Write index.html",
        "root",
    )
    root.children["root.style_css"] = child(
        "root.style_css",
        "Write style.css",
        "root",
    )
    root.children["root.script_js"] = child(
        "root.script_js",
        "Write script.js",
        "root",
    )
    root.commit(
        SupervisingOutput(
            output="delegated file work",
            waiting_on=list(root.children),
        )
    )
    out.append(deepcopy(root))

    root["root.index_html"].commit(DoneOutput(result="Created index.html"))
    root["root.style_css"].commit(DoneOutput(result="Created style.css"))
    script = root["root.script_js"]
    script.commit(LLMOutput(content="Splitting script.js into smaller pieces."))
    script.children["root.script_js.boids_core"] = child(
        "root.script_js.boids_core",
        "Core boids",
        "root.script_js",
    )
    script.children["root.script_js.renderer"] = child(
        "root.script_js.renderer",
        "Canvas renderer",
        "root.script_js",
    )
    script.children["root.script_js.controls"] = child(
        "root.script_js.controls",
        "UI controls",
        "root.script_js",
    )
    script.commit(
        SupervisingOutput(
            output="delegated script work",
            waiting_on=list(script.children),
        )
    )
    out.append(deepcopy(root))

    root["root.script_js.boids_core"].commit(DoneOutput(result="Implemented flocking"))
    root["root.script_js.renderer"].commit(DoneOutput(result="Implemented renderer"))
    root["root.script_js.controls"].commit(
        ErrorOutput(
            error="missing event handler",
            output="The first controls attempt forgot keyboard input.",
        )
    )
    out.append(deepcopy(root))

    root["root.script_js.controls"].commit(
        ExecOutput(output="Retried controls with keydown/keyup listeners.")
    )
    root["root.script_js.controls"].commit(DoneOutput(result="Implemented controls"))
    root["root.script_js"].commit(DoneOutput(result="Created script.js"))
    out.append(deepcopy(root))

    root.commit(DoneOutput(result="Created boids simulation files"))
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
            print(f"snapshot {index}: {graph.current().type if graph.current() else 'empty'}")
        return
    print("Launching viewer...")
    open_viewer(graphs)


if __name__ == "__main__":
    main()
