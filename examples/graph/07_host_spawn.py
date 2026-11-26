"""Build agents from host code: finished agents written directly, and a child launched for a new agent.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

import time
from typing import Any

from rlmflow import Agent, Flow, Graph, Node
from rlmflow.agents import repl_reply
from rlmflow.llm import Reply
from rlmflow.view import render_tree


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> Reply:
        users = [m["content"] for m in messages if m["role"] in ("user", "tool")]
        if users[0].startswith("re-check"):
            return repl_reply('submit("rollback path verified")')
        if any("rollback path verified" in text for text in users):
            return repl_reply('submit("release blocked; " + audits["candidate"])')
        return repl_reply('print(await AGENTS["recheck"])')


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
        ran = Node("assistant", code="run_audit()", output=verdict, data={"seen": 1})
        ran.finished_at = time.time()
        g.append(audit, ran)
        g.append(audit, Node("assistant", result=verdict, data={"seen": 2, "terminal": True}))
        audits[name] = audit
        print(f"{name}: {audit.status()}, {audit.final_result()!r}")

    # Pass their results on to a new agent as arguments, and launch a child for it
    # from host code: start(parent=...) is the host-side launch_agent. The
    # parent can await the child just like one launched from its own code.
    root = g.start(
        "decide the release from `audits`",
        name="root",
        audits={name: agent.final_result() for name, agent in audits.items()},
    )
    recheck = g.start("re-check the rollback path", parent=root, name="recheck")
    print(
        "spawn edge:",
        [
            (e.kind, g.node(e.source).root.name, g.agent_root(e.target).name)
            for e in g.edges(kind="spawn")
        ],
    )

    agent = Agent(model="scripted", max_iters=6)
    flow = Flow(llm_clients={"scripted": ScriptedLLM()}, agents={"default": agent})
    flow.run(g)
    flow.runtime.close()

    print()
    print(render_tree(g))
    print("recheck:", recheck.final_result())
    print("root:", root.final_result())
    assert root.final_result() == "release blocked; canary failed"


if __name__ == "__main__":
    main()
