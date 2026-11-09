"""Build agents from host code: finished agents written directly, and a child launched for a new agent.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

import time
from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Node, render_tree


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> str:
        users = [m["content"] for m in messages if m["role"] == "user"]
        if users[0].startswith("re-check"):
            return '```repl\nsubmit("rollback path verified")\n```'
        if any("[result " in text for text in users):
            return '```repl\nsubmit("release blocked; " + audits["candidate"])\n```'
        return '```repl\nprint("waiting for the re-check")\n```'


def main() -> None:
    # Agents written by hand, as if they had already run. A turn marked
    # `terminal` finishes an agent, so the flow leaves them alone.
    g = Graph()
    audits = {}
    for name, task, verdict in (
        ("candidate", "audit the release candidate", "canary failed"),
        ("rollback", "audit the rollback path", "rollback is safe"),
    ):
        audit = g.start(task, name=name)
        # Code with ``finished_at`` set already ran; without it, the flow would run it.
        ran = Node("assistant", code="run_audit()", output=verdict, seen=1)
        ran.finished_at = time.time()
        g.append(audit, ran)
        g.append(audit, Node("assistant", result=verdict, seen=2, terminal=True))
        audits[name] = audit
        print(f"{name}: {g.status(audit)}, {g.result(audit)!r}")

    # Pass their results on to a new agent as arguments, and launch a child for it
    # from host code: start(parent=...) is the host-side launch_agent. The
    # child's result reaches the parent as a message at its next turn.
    root = g.start(
        "decide the release from `audits`",
        name="root",
        audits={name: g.result(agent) for name, agent in audits.items()},
    )
    recheck = g.start("re-check the rollback path", parent=root, name="recheck")
    print(
        "spawn edge:",
        [
            (e.kind, g.owner(g.node(e.source)).root_data.name, g.agent(e.target).root_data.name)
            for e in g.edges(kind="spawn")
        ],
    )

    config = AgentConfig(model="scripted", max_iters=6)
    flow = Flow(llm_clients={"scripted": ScriptedLLM()}, configs=[config])
    flow.run(g, root_config=config)
    flow.runtime.close()

    print()
    print(render_tree(g))
    print("recheck:", g.result(recheck))
    print("root:", g.result(root))
    assert g.result(root) == "release blocked; canary failed"


if __name__ == "__main__":
    main()
