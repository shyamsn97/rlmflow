"""Show minimal ``Flow.run_streaming(..., until=...)`` boundaries during delegation.

Minimal Flow does not expose the old ``eager_children`` toggle. Delegation fans
out child agents as independent scheduler tasks; blocking LLM calls are bounded
by the flow's worker pool. This example focuses on the caller-facing control
surface instead: choosing how much of the event stream to consume with
``until=...``.

Run:
    export OPENAI_API_KEY=...
    python examples/control/delegation/step_until.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rflow import Event, Flow, Graph, GraphCheckpointer, render_tree

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client, save_example_graph  # noqa: E402


QUERY = """\
Show minimal `Flow.run_streaming(..., until=...)` boundaries with delegated child agents.

In your first REPL block, launch two child agents in one `await launch_subagents`
call, then call done with their joined results. Use exactly these child names:

- `slow`: ask it to solve a tiny task after sleeping briefly with
  `await asyncio.sleep(1)`.
- `fast`: ask it to solve a tiny task immediately.

This example script is observing the graph event stream, so keep the work small.
"""


def event_label(event: Event) -> str:
    if event.type == "append_node":
        return f"{event.agent_id}: append {event.node_type}"
    if event.type == "add_child":
        return f"{event.parent_agent_id}: add child {event.child.agent_id}"
    if event.type == "graph_created":
        return "graph created"
    if event.type == "remove_child":
        return f"{event.parent_agent_id}: remove child {event.child_agent_id}"
    if event.type == "replace_node":
        return f"{event.agent_id}: replace {event.node_type}"
    if event.type == "remove_node":
        return f"{event.agent_id}: remove node {event.node_id}"
    return event.type


def print_events(title: str, events: list[Event], graph: Graph) -> None:
    print(f"\n=== {title} ===")
    for event in events:
        print(f"- {event_label(event)}")
    print(render_tree(graph))


async def run_example(args: argparse.Namespace) -> Graph:
    flow = Flow(
        build_client(args.model),
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        workers=args.max_concurrency,
    )
    graph = Graph(query=QUERY)
    observed: list[Event] = []
    checkpointer = GraphCheckpointer(Path(args.out_dir))

    async def collect(graph: Graph, **flow_kwargs) -> list[Event]:
        events: list[Event] = []
        async for event in flow.run_streaming(graph=graph, **flow_kwargs):
            checkpointer.handle(event, graph)
            events.append(event)
        return events

    events = await collect(graph, until="next")
    observed.extend(events)
    print_events("until='next': first appended node", events, graph)

    events = await collect(graph, until="supervising")
    observed.extend(events)
    print_events("until='supervising': parent has fanned out children", events, graph)

    def first_child_done(event: Event, graph: Graph) -> bool:
        return (
            event.type == "append_node"
            and event.agent_id != "root"
            and event.node_type == "done_output"
        )

    events = await collect(graph, until=first_child_done)
    observed.extend(events)
    print_events("until=<callable>: stop when any child is done", events, graph)

    events = await collect(graph, until="next", n=2)
    observed.extend(events)
    print_events("until='next', n=2: consume two more global steps", events, graph)

    while not graph.finished:
        events = await collect(graph, until=lambda event, current: current.finished)
        observed.extend(events)
        print_events("until=<callable>: run is finished", events, graph)

    if observed:
        print(f"\nObserved {len(observed)} graph events.")

    checkpointer.close()
    flow.close_repls(graph.graph_id)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show minimal Flow.run_streaming(..., until=...) boundaries."
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[2] / "_runs" / "step-until"),
        help="Save the final run here.",
    )
    args = parser.parse_args()

    graph = asyncio.run(run_example(args))
    print("\n=== verdict ===")
    print("Delegated children fan out as independent scheduler tasks.")
    print("The caller chooses observation boundaries with run_streaming(..., until=...).")
    print(f"Result: {graph.result()}")
    save_example_graph(graph, "step-until", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
