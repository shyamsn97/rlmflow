"""Forking a graph: a copy cut after a node, steered with a message and run on its own.

``g.fork(node)`` returns a new graph with the same ids, cut right after
``node``: what came after it is gone, including the agents it launched. The
original is left as it is. A scripted model stands in for the LLM, so this runs
offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Node
from rlmflow.view.text import kind

DELEGATE = "review = await launch_agent('review the plan')\nsubmit('ship it; reviewer: ' + review)"
VARIANTS = {
    "conservative": ("Be conservative.", "submit('ship a small patch release')"),
    "bold": ("Be bold.", "submit('ship a major release candidate')"),
}


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> str:
        said = [str(m["content"]) for m in messages if m["role"] == "user"]
        if "review the plan" in said[0]:
            return "```repl\nsubmit('looks fine')\n```"
        for steer, code in VARIANTS.values():
            if any(steer in text for text in said):
                return f"```repl\n{code}\n```"
        return f"```repl\n{DELEGATE}\n```"


def show(name: str, g: Graph, root: Node) -> None:
    agents = ", ".join(agent.name for agent in g.agents())
    print(f"{name}: {root.final_result()!r} (agents: {agents})")
    print("  " + " -> ".join(kind(node) for node in root.nodes()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = AgentConfig(model="scripted")
    flow = Flow(
        llm_clients={"scripted": ScriptedLLM()},
        configs=[AgentConfig(name="default", model="scripted")],
    )
    g = Graph()
    root = g.start("choose a release plan", name="planner")
    flow.run(g, root_config=config)
    show("original", g, root)

    branches = {"original": g}
    for name, (steer, _) in VARIANTS.items():
        # Cut right after the task: the delegating turn and the reviewer are gone.
        branch = g.fork(root)
        copy = branch.agent(root.id)
        branch.send(copy, steer)
        flow.run(branch, root_config=config)
        show(name, branch, copy)
        branches[name] = branch
    flow.runtime.close()

    assert root.final_result().startswith("ship it") and len(g.agents()) == 2
    if args.out_dir is not None:
        for name, branch in branches.items():
            print("saved:", branch.save(args.out_dir / name))


if __name__ == "__main__":
    main()
