"""Edit a finished run: change a result or one saved call, and re-run only what used it.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

from typing import Any

from rlmflow import Agent, Flow, Graph
from rlmflow.agents import repl_reply
from rlmflow.graph import Invalidation
from rlmflow.llm import Reply
from rlmflow.view.text import kind

ROOT_REPLY = repl_reply(
    'verdict = await launch_agent("audit the canary", name="audit")\nsubmit(f"plan: {verdict}")'
)


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> Reply:
        users = [m["content"] for m in messages if m["role"] == "user"]
        if users[0].startswith("audit the canary"):
            return repl_reply('submit("canary failed")')
        if "Account for the rollback window" in users[-1]:
            return repl_reply('submit(f"plan: {verdict}; roll back within 10 minutes")')
        return ROOT_REPLY


def chain(agent) -> str:
    return " -> ".join(kind(node) for node in agent.nodes())


def describe(g: Graph, inv: Invalidation) -> str:
    rewound = [g.agent_root(agent_id).name for agent_id in inv.rewound]
    return (
        f"rewound={rewound} removed={len(inv.removed)} "
        f"dropped={len(inv.dropped)} rerun={len(inv.rerun)} node(s)"
    )


def main() -> None:
    agent = Agent(model="scripted", max_depth=1)
    flow = Flow(llm_clients={"scripted": ScriptedLLM()}, agents={"default": agent})
    g = Graph()
    root = g.start("draft a release plan", name="root")
    flow.run(g)
    audit = root.children()[0]
    print("first run:", root.final_result())
    print("root nodes:", chain(root))

    # Change the child's result. A dry run reports what the edit would invalidate
    # and changes nothing; the real edit rewinds the root to the node that awaited
    # the child and sets that node to run again.
    done = audit.last_turn()
    preview = g.edit(done, result="canary passed", dry_run=True)
    print("\ndry run:", describe(g, preview))
    assert audit.final_result() == "canary failed"

    invalidation = g.edit(done, result="canary passed")
    awaiting = g.node(invalidation.rerun[0])
    print("edit:", describe(g, invalidation))
    print("root status:", root.status(), "| node state:", awaiting.state)
    flow.run(g)
    print("after edit:", root.final_result())
    assert root.final_result() == "plan: canary passed"

    # Or rebind one saved call of a node: here the `await`, to a value. The child
    # stays as it is; only the node re-runs, with the await returning the value.
    print("\ncalls:", [entry["tool"] for entry in awaiting.calls])
    index = next(i for i, entry in enumerate(awaiting.calls) if entry["tool"] == "await")
    print("replace:", describe(g, g.replace(awaiting, index, "canary flaky")))
    flow.run(g)
    print("after replace:", root.final_result())
    assert root.final_result() == "plan: canary flaky"

    # A message reopens a finished agent; its nodes and REPL carry on.
    g.send(root, "Account for the rollback window.")
    flow.run(g)
    print("\nafter message:", root.final_result())
    print("root nodes:", chain(root))
    flow.runtime.close()


if __name__ == "__main__":
    main()
