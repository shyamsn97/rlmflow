"""Every graph is its own timeline: replay a run to see each earlier state.

A scripted model stands in for the LLM, so this runs offline.
"""

from __future__ import annotations

from typing import Any

from rlmflow import Agent, Flow, Graph, Node
from rlmflow.agents import repl_reply
from rlmflow.llm import Reply
from rlmflow.view import replay
from rlmflow.view.text import kind

REPLIES = [
    repl_reply("source = 'print(1)'\nprint(len(source))"),
    repl_reply("submit('hello.py is ' + source)"),
]


class ScriptedLLM:
    def chat(self, messages: list[dict[str, Any]]) -> Reply:
        turns = sum(message["role"] == "assistant" for message in messages)
        return REPLIES[min(turns, len(REPLIES) - 1)]


def print_nodes(agent: Node) -> None:
    print("  " + " -> ".join(kind(node) for node in agent.nodes()))


def main() -> None:
    flow = Flow(
        llm_clients={"scripted": ScriptedLLM()}, agents={"default": Agent(model="scripted")}
    )
    g = Graph()
    root = g.start("write hello.py")
    flow.run(g)
    flow.runtime.close()

    # Agents and nodes carry the order they were added in, so the graph cut after
    # each one (with the edges between what's kept) is a real earlier graph.
    for step, snapshot in enumerate(replay(g)):
        agent = snapshot.agent_root(root.id)
        print(f"\nstep {step}: {agent.status()}, result={agent.final_result()!r}")
        print_nodes(agent)


if __name__ == "__main__":
    main()
