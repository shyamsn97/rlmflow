"""Offline examples for ``Flow.run_streaming(..., until=...)``.

Run:
    python examples/control/streaming_until.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlmflow import (
    AgentConfig,
    AgentStart,
    Flow,
    Graph,
    Node,
    ops,
    reads,
)

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


async def collect(flow: Flow, g: Graph, **kwargs: Any) -> list[Node]:
    """The nodes a run lands; the edges streamed alongside them are left out."""
    return [item async for item in flow.run_streaming(g, **kwargs) if isinstance(item, Node)]


def after_appends(count: int):
    """Halt once ``count`` nodes have landed."""
    seen = 0

    def halt(item, _g: Graph) -> bool:
        nonlocal seen
        if isinstance(item, Node):
            seen += 1
        return seen >= count

    return halt


def node_label(g: Graph, node: Node) -> str:
    return f"{reads.path(g, g.node(node.agent_id))}: {node.type}"


def print_tree(g: Graph, agent: AgentStart, depth: int = 0) -> None:
    """Each agent's history on one line, with its children indented below."""
    history = " -> ".join(node.type for node in reads.history(g, agent))
    print(f"{'  ' * depth}[{reads.path(g, agent)}] {history}")
    for child in reads.children(g, agent):
        print_tree(g, child, depth + 1)


def print_nodes(title: str, nodes: list[Node], g: Graph, root: AgentStart) -> None:
    print(f"\n=== {title} ===")
    for node in nodes:
        print(f"- {node_label(g, node)}")
    print_tree(g, root)


async def demo_next_idle_and_resume() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return '```repl\nsubmit("stopped")\n```'
        return "```repl\nprint('working')\n```"

    flow = Flow(ScriptedLLM(reply))
    g = Graph()
    root = g.create_agent(
        "work until I inject a stop instruction",
        config=AgentConfig(max_iters=8),
    )

    events = await collect(flow, g, until=after_appends(3))
    print_nodes("callable boundary: exactly three appended nodes", events, g, root)

    events = await collect(flow, g, until="idle")
    print_nodes("until='idle': settle at a clean exec_output", events, g, root)

    # Injecting a turn is sending the agent a message; it lands at the next turn.
    ops.send(g, root, "STOP NOW")
    events = await collect(flow, g, until="done")
    print_nodes("resume after injection, until='done'", events, g, root)
    g.save(RUN_DIR / "next-idle-and-resume")


async def demo_idle_heals_errors() -> None:
    flow = Flow(
        ScriptedLLM(
            scripted_replies(
                "not a repl block",
                "```repl\nprint('fixed')\n```",
                "```repl\nsubmit('ok')\n```",
            )
        )
    )
    g = Graph()
    root = g.create_agent("recover from a bad response")

    events = await collect(flow, g, until="idle")
    print_nodes("until='idle': pass through error_output to exec_output", events, g, root)
    g.save(RUN_DIR / "idle-heals-errors")


async def demo_callable_boundary_with_child() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "child = launch_subagent(task='child')\n"
                "result = await child\n"
                "submit('parent saw ' + result)\n"
                "```"
            )
        return "```repl\nsubmit('child done')\n```"

    flow = Flow(ScriptedLLM(reply))
    g = Graph()
    root = g.create_agent("parent", config=AgentConfig(max_depth=1))

    def child_done(item, _g: Graph) -> bool:
        return isinstance(item, Node) and item.agent_id != root.id and item.type == "done_output"

    events = await collect(flow, g, until=child_done)
    print_nodes("until=<callable>: observe child done_output", events, g, root)
    g.save(RUN_DIR / "callable-boundary-with-child")


async def main() -> None:
    await demo_next_idle_and_resume()
    await demo_idle_heals_errors()
    await demo_callable_boundary_with_child()


if __name__ == "__main__":
    asyncio.run(main())
