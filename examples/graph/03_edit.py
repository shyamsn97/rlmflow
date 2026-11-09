"""Edit a finished run: change a result or one saved call, and re-run only what used it.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Invalidation
from rlmflow.view.text import kind

ROOT_REPLY = """\
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
        return ROOT_REPLY


def history(g: Graph, agent) -> str:
    return " -> ".join(kind(node) for node in g.history(agent))


def describe(g: Graph, inv: Invalidation) -> str:
    rewound = [g.agent(agent_id).root_data.name for agent_id in inv.rewound]
    return (
        f"rewound={rewound} removed={len(inv.removed)} "
        f"dropped={len(inv.dropped)} rerun={len(inv.rerun)} node(s)"
    )


def main() -> None:
    config = AgentConfig(model="scripted", max_depth=1)
    flow = Flow(llm_clients={"scripted": ScriptedLLM()}, configs=[config])
    g = Graph()
    root = g.start("draft a release plan", name="root")
    flow.run(g, root_config=config)
    audit = g.children(root)[0]
    print("first run:", g.result(root))
    print("root history:", history(g, root))

    # Change the child's result. A dry run reports what the edit would invalidate
    # and changes nothing; the real edit rewinds the root to the node that awaited
    # the child and sets that node to run again.
    done = g.last_turn(audit)
    preview = g.edit(done, result="canary passed", dry_run=True)
    print("\ndry run:", describe(g, preview))
    assert g.result(audit) == "canary failed"

    invalidation = g.edit(done, result="canary passed")
    awaiting = g.node(invalidation.rerun[0])
    print("edit:", describe(g, invalidation))
    print("root status:", g.status(root), "| node state:", awaiting.state)
    flow.run(g, root_config=config)
    print("after edit:", g.result(root))
    assert g.result(root) == "plan: canary passed"

    # Or rebind one saved call of a node: here the `await`, to a value. The child
    # stays as it is; only the node re-runs, with the await returning the value.
    print("\ncalls:", [entry["tool"] for entry in awaiting.calls])
    index = next(i for i, entry in enumerate(awaiting.calls) if entry["tool"] == "await")
    print("replace:", describe(g, g.replace(awaiting, index, "canary flaky")))
    flow.run(g, root_config=config)
    print("after replace:", g.result(root))
    assert g.result(root) == "plan: canary flaky"

    # A message reopens a finished agent; its history and REPL carry on.
    g.send(root, "Account for the rollback window.")
    flow.run(g, root_config=config)
    print("\nafter message:", g.result(root))
    print("root history:", history(g, root))
    flow.runtime.close()


if __name__ == "__main__":
    main()
