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
    AgentStart,
    Flow,
    Node,
    UserQuery,
    persistence,
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


async def collect(flow: Flow, root: Node, **kwargs: Any) -> list[Node]:
    return [node async for node in flow.run_streaming(root, **kwargs)]


def after_appends(count: int):
    """Halt once ``count`` nodes have landed; every node in the stream is an append."""
    seen = 0

    def halt(_node: Node, _root: Node) -> bool:
        nonlocal seen
        seen += 1
        return seen >= count

    return halt


def node_label(node: Node) -> str:
    return f"{node.parent_agent.config.path}: {node.type}"


def print_tree(node: Node, depth: int = 0) -> None:
    """One indented line per node, from ``node`` down."""
    label = f"{node.type} [{node.config.path}]" if isinstance(node, AgentStart) else node.type
    print(f"{'  ' * depth}{label}")
    for child in node.children:
        print_tree(child, depth + 1)


def print_nodes(title: str, nodes: list[Node], graph: Node) -> None:
    print(f"\n=== {title} ===")
    for node in nodes:
        print(f"- {node_label(node)}")
    print_tree(graph)


async def demo_next_idle_and_resume() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return '```repl\ndone("stopped")\n```'
        return "```repl\nprint('working')\n```"

    flow = Flow(ScriptedLLM(reply))
    graph = flow.start("work until I inject a stop instruction", max_iters=8)

    events = await collect(flow, graph, until=after_appends(3))
    print_nodes("callable boundary: exactly three appended nodes", events, graph)

    events = await collect(flow, graph, until="idle")
    print_nodes("until='idle': settle at a clean exec_output", events, graph)

    # Injecting a turn is appending a node to the agent's frontier yourself.
    graph.frontier.append(UserQuery(content="STOP NOW"))
    events = await collect(flow, graph, until="done")
    print_nodes("resume after injection, until='done'", events, graph)
    persistence.save(graph, RUN_DIR / "next-idle-and-resume")


async def demo_idle_heals_errors() -> None:
    flow = Flow(
        ScriptedLLM(
            scripted_replies(
                "not a repl block",
                "```repl\nprint('fixed')\n```",
                "```repl\ndone('ok')\n```",
            )
        )
    )
    graph = flow.start("recover from a bad response")

    events = await collect(flow, graph, until="idle")
    print_nodes("until='idle': pass through error_output to exec_output", events, graph)
    persistence.save(graph, RUN_DIR / "idle-heals-errors")


async def demo_callable_boundary_with_child() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "child = await launch_subagent('child', name='child')\n"
                "result = await child.wait_for_result()\n"
                "done('parent saw ' + result)\n"
                "```"
            )
        return "```repl\ndone('child done')\n```"

    flow = Flow(ScriptedLLM(reply))
    graph = flow.start("parent", max_depth=1)

    def child_done(node: Node, root: Node) -> bool:
        return node.parent_agent is not root and node.type == "done_output"

    events = await collect(flow, graph, until=child_done)
    print_nodes("until=<callable>: observe child done_output", events, graph)
    persistence.save(graph, RUN_DIR / "callable-boundary-with-child")


async def main() -> None:
    await demo_next_idle_and_resume()
    await demo_idle_heals_errors()
    await demo_callable_boundary_with_child()


if __name__ == "__main__":
    asyncio.run(main())
