"""Offline examples for ``Flow.run_streaming(..., until=...)``.

Run:
    python examples/control/streaming_until.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlmflow import AgentConfig, Flow, Graph, Node, Step
from rlmflow.view.text import kind

RUN_DIR = Path(__file__).resolve().parents[1] / "_runs" / "streaming-until"


class ScriptedLLM:
    """Tiny deterministic LLM for examples that should run without credentials."""

    def __init__(self, reply: str | Callable[[list[dict[str, Any]]], str]) -> None:
        self.reply = reply

    def chat(self, messages: list[dict[str, Any]]) -> str:
        return self.reply(messages) if callable(self.reply) else self.reply


def scripted_replies(*replies: str) -> Callable[[list[dict[str, Any]]], str]:
    state = {"index": 0}

    def reply(_messages: list[dict[str, Any]]) -> str:
        index = min(state["index"], len(replies) - 1)
        state["index"] += 1
        return replies[index]

    return reply


def first_user(messages: list[dict[str, Any]]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


async def collect(flow: Flow, g: Graph, **kwargs: Any) -> list[str]:
    """A label for each node the run lands, in order, taken as it lands."""
    return [
        f"{g.path(step.agent)}: {kind(step.node)}" async for step in flow.run_streaming(g, **kwargs)
    ]


def after_appends(count: int):
    """Halt once ``count`` nodes have landed."""
    seen = 0

    def halt(_step: Step) -> bool:
        nonlocal seen
        seen += 1
        return seen >= count

    return halt


def print_tree(g: Graph, agent: Node, depth: int = 0) -> None:
    """Each agent's history on one line, with its children indented below."""
    history = " -> ".join(kind(node) for node in g.history(agent))
    print(f"{'  ' * depth}[{g.path(agent)}] {history}")
    for child in g.children(agent):
        print_tree(g, child, depth + 1)


def print_nodes(title: str, labels: list[str], g: Graph, root: Node) -> None:
    print(f"\n=== {title} ===")
    for label in labels:
        print(f"- {label}")
    print_tree(g, root)


async def demo_next_idle_and_resume() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return '```repl\nsubmit("stopped")\n```'
        return "```repl\nprint('working')\n```"

    flow = Flow(llm_clients={"scripted": ScriptedLLM(reply)})
    g = Graph()
    root = g.start("work until I inject a stop instruction")
    config = AgentConfig(model="scripted", max_iters=8)

    events = await collect(flow, g, until=after_appends(3), root_config=config)
    print_nodes("callable boundary: exactly three appended nodes", events, g, root)

    events = await collect(flow, g, until="idle", root_config=config)
    print_nodes("until='idle': settle after a node ran code", events, g, root)

    # Injecting a turn is sending the agent a message; it lands at the next turn.
    g.send(root, "STOP NOW")
    events = await collect(flow, g, until="done", root_config=config)
    print_nodes("resume after injection, until='done'", events, g, root)
    g.save(RUN_DIR / "next-idle-and-resume")


async def demo_idle_heals_errors() -> None:
    replies = scripted_replies(
        "```repl\n```",
        "```repl\nprint('fixed')\n```",
        "```repl\nsubmit('ok')\n```",
    )
    flow = Flow(
        llm_clients={"scripted": ScriptedLLM(replies)}, configs=[AgentConfig(model="scripted")]
    )
    g = Graph()
    root = g.start("recover from a bad response")

    events = await collect(flow, g, until="idle")
    print_nodes("until='idle': an empty block is reported back as an error", events, g, root)
    assert g.tail(root).error
    g.save(RUN_DIR / "idle-heals-errors")


async def demo_callable_boundary_with_child() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "child = launch_agent('child')\n"
                "result = await child\n"
                "submit('parent saw ' + result)\n"
                "```"
            )
        return "```repl\nsubmit('child done')\n```"

    flow = Flow(
        llm_clients={"scripted": ScriptedLLM(reply)}, configs=[AgentConfig(model="scripted")]
    )
    g = Graph()
    root = g.start("parent")

    def child_done(step: Step) -> bool:
        return step.agent.id != root.id and step.node.data.get("terminal", False)

    events = await collect(flow, g, until=child_done)
    # The halt cuts the root's node off mid-await. It is left ready to run again,
    # and the child it launched stays, finished.
    print_nodes("until=<callable>: observe the child's done", events, g, root)
    [child] = g.children(root)
    assert g.status(child) == "done"

    events = await collect(flow, g, until="done")
    print_nodes("resume: the node re-runs in place and gets the same child back", events, g, root)
    assert g.children(root) == [child]
    assert g.result(root) == "parent saw child done"
    g.save(RUN_DIR / "callable-boundary-with-child")


async def main() -> None:
    await demo_next_idle_and_resume()
    await demo_idle_heals_errors()
    await demo_callable_boundary_with_child()


if __name__ == "__main__":
    asyncio.run(main())
