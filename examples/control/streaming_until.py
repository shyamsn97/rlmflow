"""Offline examples for ``Flow.run_streaming(..., until=...)``.

Run:
    python examples/control/streaming_until.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rflow import Event, Flow, Graph, render_tree

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


async def collect(flow: Flow, **kwargs: Any) -> list[Event]:
    return [event async for event in flow.run_streaming(**kwargs)]


def event_label(event: Event) -> str:
    if event.type == "graph_created":
        return f"{event.graph.agent_id}: graph_created"
    if event.type == "append_node":
        return f"{event.agent_id}: append {event.node_type}"
    if event.type == "add_child":
        return f"{event.parent_agent_id}: add child {event.child.agent_id}"
    return event.type


def print_events(title: str, events: list[Event], graph: Graph) -> None:
    print(f"\n=== {title} ===")
    for event in events:
        print(f"- {event_label(event)}")
    print(render_tree(graph))


async def demo_next_idle_and_resume() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        text = "\n".join(message["content"] for message in messages)
        if "STOP NOW" in text:
            return '```repl\ndone("stopped")\n```'
        return "```repl\nprint('working')\n```"

    flow = Flow(ScriptedLLM(reply), max_iters=8)
    graph = Graph(query="work until I inject a stop instruction")

    events = await collect(flow, graph=graph, until="next", n=3)
    print_events("until='next', n=3: exactly three appended nodes", events, graph)

    events = await collect(flow, graph=graph, until="idle")
    print_events("until='idle': settle at a clean exec_output", events, graph)

    graph.inject("STOP NOW")
    events = await collect(flow, graph=graph, until="done")
    print_events("resume after injection, until='done'", events, graph)
    graph.save(RUN_DIR / "next-idle-and-resume")


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
    graph = Graph(query="recover from a bad response")

    events = await collect(flow, graph=graph, until="idle")
    print_events("until='idle': pass through error_output to exec_output", events, graph)
    graph.save(RUN_DIR / "idle-heals-errors")


async def demo_callable_boundary_with_child() -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents([{'name': 'child', 'query': 'child'}])\n"
                "done('parent saw ' + results[0])\n"
                "```"
            )
        return "```repl\ndone('child done')\n```"

    flow = Flow(ScriptedLLM(reply), max_depth=1)
    graph = Graph(query="parent")

    def child_done(event: Event, current: Graph) -> bool:
        return (
            event.type == "append_node"
            and event.agent_id != current.agent_id
            and event.node_type == "done_output"
        )

    events = await collect(flow, graph=graph, until=child_done)
    print_events("until=<callable>: observe child done_output", events, graph)
    graph.save(RUN_DIR / "callable-boundary-with-child")


async def main() -> None:
    await demo_next_idle_and_resume()
    await demo_idle_heals_errors()
    await demo_callable_boundary_with_child()


if __name__ == "__main__":
    asyncio.run(main())
