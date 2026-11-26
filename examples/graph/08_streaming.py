"""Stopping a run and picking it up again: ``Flow.stream(..., until=...)``.

A run stops after a set number of nodes, after a node that ran code, or when a
predicate on a node holds; a message sent in between lands at the next turn,
and streaming again resumes where it stopped.

Run:
    python examples/graph/08_streaming.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlmflow import Agent, Flow, Graph, Node
from rlmflow.agents import repl_reply
from rlmflow.llm import Reply
from rlmflow.view.text import kind

RUN_DIR = Path(__file__).resolve().parents[1] / "_runs" / "graph-streaming"


class ScriptedLLM:
    """Tiny deterministic LLM for examples that should run without credentials."""

    def __init__(self, reply: Reply | Callable[[list[dict[str, Any]]], Reply]) -> None:
        self.reply = reply

    def chat(self, messages: list[dict[str, Any]]) -> Reply:
        return self.reply(messages) if callable(self.reply) else self.reply


def scripted_replies(*replies: Reply) -> Callable[[list[dict[str, Any]]], Reply]:
    state = {"index": 0}

    def reply(_messages: list[dict[str, Any]]) -> Reply:
        index = min(state["index"], len(replies) - 1)
        state["index"] += 1
        return replies[index]

    return reply


def first_user(messages: list[dict[str, Any]]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


async def collect(flow: Flow, g: Graph, **kwargs: Any) -> list[str]:
    """A label for each node the run lands, in order, taken as it lands."""
    return [f"{node.root.path()}: {kind(node)}" async for node in flow.stream(g, **kwargs)]


def after_appends(count: int):
    """Halt once ``count`` nodes have landed."""
    seen = 0

    def halt(_node: Node) -> bool:
        nonlocal seen
        seen += 1
        return seen >= count

    return halt


def print_tree(g: Graph, agent: Node, depth: int = 0) -> None:
    """Each agent's nodes on one line, with its children indented below."""
    line = " -> ".join(kind(node) for node in agent.nodes())
    print(f"{'  ' * depth}[{agent.path()}] {line}")
    for child in agent.children():
        print_tree(g, child, depth + 1)


def print_nodes(title: str, labels: list[str], g: Graph, root: Node) -> None:
    print(f"\n=== {title} ===")
    for label in labels:
        print(f"- {label}")
    print_tree(g, root)


async def demo_next_idle_and_resume() -> None:
    def reply(messages: list[dict[str, Any]]) -> Reply:
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return repl_reply('submit("stopped")')
        return repl_reply("print('working')")

    flow = Flow(
        llm_clients={"scripted": ScriptedLLM(reply)},
        agents={"default": Agent(model="scripted", max_iters=8)},
    )
    g = Graph()
    root = g.start("work until I inject a stop instruction")

    events = await collect(flow, g, until=after_appends(3))
    print_nodes("callable boundary: exactly three appended nodes", events, g, root)

    events = await collect(flow, g, until="idle")
    print_nodes("until='idle': settle after a node ran code", events, g, root)

    # Injecting a turn is sending the agent a message; it lands at the next turn.
    g.send(root, "STOP NOW")
    events = await collect(flow, g, until="done")
    print_nodes("resume after injection, until='done'", events, g, root)
    g.save(RUN_DIR / "next-idle-and-resume")


async def demo_idle_heals_errors() -> None:
    replies = scripted_replies(
        repl_reply(""),
        repl_reply("print('fixed')"),
        repl_reply("submit('ok')"),
    )
    flow = Flow(
        llm_clients={"scripted": ScriptedLLM(replies)}, agents={"default": Agent(model="scripted")}
    )
    g = Graph()
    root = g.start("recover from a bad response")

    events = await collect(flow, g, until="idle")
    print_nodes("until='idle': empty code is reported back as an error", events, g, root)
    assert root.nodes()[-1].error
    g.save(RUN_DIR / "idle-heals-errors")


async def demo_callable_boundary_with_child() -> None:
    def reply(messages: list[dict[str, Any]]) -> Reply:
        if first_user(messages) == "parent":
            return repl_reply(
                "child = launch_agent('child')\n"
                "result = await child\n"
                "submit('parent saw ' + result)"
            )
        return repl_reply("submit('child done')")

    flow = Flow(
        llm_clients={"scripted": ScriptedLLM(reply)}, agents={"default": Agent(model="scripted")}
    )
    g = Graph()
    root = g.start("parent")

    def child_done(node: Node) -> bool:
        return node.root.id != root.id and node.result is not None

    events = await collect(flow, g, until=child_done)
    # The halt cuts the root's node off mid-await. It is left ready to run again,
    # and the child it launched stays, finished.
    print_nodes("until=<callable>: observe the child's done", events, g, root)
    [child] = root.children()
    assert child.status() == "done"

    events = await collect(flow, g, until="done")
    print_nodes("resume: the node re-runs in place and gets the same child back", events, g, root)
    assert root.children() == [child]
    assert root.final_result() == "parent saw child done"
    g.save(RUN_DIR / "callable-boundary-with-child")


async def main() -> None:
    await demo_next_idle_and_resume()
    await demo_idle_heals_errors()
    await demo_callable_boundary_with_child()


if __name__ == "__main__":
    asyncio.run(main())
