"""Every graph is its own timeline: replay a run to see each earlier state.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Node
from rlmflow.view import replay
from rlmflow.view.text import kind

REPLIES = [
    "```repl\nsource = 'print(1)'\nprint(len(source))\n```",
    "```repl\nsubmit('hello.py is ' + source)\n```",
]


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> str:
        turns = sum(message["role"] == "assistant" for message in messages)
        return REPLIES[min(turns, len(REPLIES) - 1)]


def print_history(g: Graph, agent: Node) -> None:
    print("  " + " -> ".join(kind(node) for node in g.history(agent)))


def main() -> None:
    flow = Flow(llm_clients={"scripted": ScriptedLLM()})
    g = Graph()
    root = g.add_agent("write hello.py")
    flow.run(g, root_config=AgentConfig(model="scripted"))
    flow.runtime.close()

    # Agents and nodes carry the order they were added in, so the graph cut after
    # each one (with the edges between what's kept) is a real earlier graph.
    for step, snapshot in enumerate(replay(g)):
        agent = snapshot.agent(root.id)
        print(f"\nstep {step}: {snapshot.status(agent)}, result={snapshot.result(agent)!r}")
        print_history(snapshot, agent)


if __name__ == "__main__":
    main()
