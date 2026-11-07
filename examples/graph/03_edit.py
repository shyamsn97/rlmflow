"""Edit a finished run: change a result or one saved call, and re-run only what used it.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Invalidation

ROOT_CELL = """\
```repl
verdict = await launch_agent("audit the canary", name="audit")
submit(f"plan: {verdict}")
```"""


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> str:
        users = [m["content"] for m in messages if m["role"] == "user"]
        if users[0].startswith("audit the canary"):
            return '```repl\nsubmit("canary failed")\n```'
        if "Account for the rollback window" in users[-1]:
            return '```repl\nsubmit(f"plan: {verdict}; roll back within 10 minutes")\n```'
        return ROOT_CELL


def history(g: Graph, agent) -> str:
    return " -> ".join(node.type for node in g.history(agent))


def describe(g: Graph, inv: Invalidation) -> str:
    rewound = [g.agent(agent_id).name for agent_id in inv.rewound]
    return (
        f"rewound={rewound} removed={len(inv.removed)} "
        f"dropped={len(inv.dropped)} rerun={len(inv.rerun)} cell(s)"
    )


def main() -> None:
    config = AgentConfig(model="scripted", max_depth=1)
    flow = Flow(llm_clients={"scripted": ScriptedLLM()}, configs=[config])
    g = Graph()
    root = g.add_agent("draft a release plan", name="root")
    flow.run(g, root_config=config)
    audit = g.children(root)[0]
    print("first run:", g.result(root))
    print("root history:", history(g, root))

    # Change the child's result. A dry run reports what the edit would invalidate
    # and changes nothing; the real edit rewinds the root to the cell that awaited
    # the child and marks that cell to re-run.
    done = g.history(audit)[-1]
    preview = g.edit(done, result="canary passed", dry_run=True)
    print("\ndry run:", describe(g, preview))
    assert g.result(audit) == "canary failed"

    invalidation = g.edit(done, result="canary passed")
    cell = g.node(invalidation.rerun[0])
    print("edit:", describe(g, invalidation))
    print("root status:", g.status(root), "| cell pending:", cell.pending)
    flow.run(g, root_config=config)
    print("after edit:", g.result(root))
    assert g.result(root) == "plan: canary passed"

    # Or rebind one saved call of a cell: here the `await`, to a value. The child
    # stays as it is; only the cell re-runs, with the await returning the value.
    saved = cell.data["calls"]
    print("\ncalls:", [entry["tool"] for entry in saved])
    index = next(i for i, entry in enumerate(saved) if entry["tool"] == "await")
    print("replace:", describe(g, g.replace(cell, index, "canary flaky")))
    flow.run(g, root_config=config)
    print("after replace:", g.result(root))
    assert g.result(root) == "plan: canary flaky"

    # A message reopens a finished agent; its history and REPL carry on.
    g.send(root, "Account for the rollback window.")
    flow.run(g, root_config=config)
    print("\nafter message:", g.result(root))
    print("root history:", history(g, root))

    print("\nedit log:", [edit["op"] for edit in g.edits])
    flow.runtime.close()


if __name__ == "__main__":
    main()
