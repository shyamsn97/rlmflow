"""Show minimal ``Flow.step(..., until=...)`` boundaries during delegation.

Minimal Flow does not expose the old ``eager_children`` toggle. Delegation fans
out child agents by default through the shared async pool. This example focuses
on the caller-facing control surface instead: choosing how much of the event
stream to consume with ``until=...``.

Run:
    export OPENAI_API_KEY=...
    python examples/control/delegation/step_until.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rflow.minimal.clients import AnthropicClient, OpenAIClient
from rflow.minimal import Event, Flow, Graph, render_tree


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph: Graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path


QUERY = """\
Show minimal `Flow.step(..., until=...)` boundaries with delegated child agents.

In your first REPL block, launch two child agents in one `await launch_subagents`
call, then call done with their joined results. Use exactly these child names:

- `slow`: ask it to solve a tiny task after sleeping briefly with
  `await asyncio.sleep(1)`.
- `fast`: ask it to solve a tiny task immediately.

This example script is observing the graph event stream, so keep the work small.
"""


def build_llm(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


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
        build_llm(args.model),
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        max_concurrency=args.max_concurrency,
    )
    graph = Graph(query=QUERY)
    observed: list[Event] = []

    events = await flow.step(graph)  # default: until="node"
    observed.extend(events)
    print_events("default step(): first appended node", events, graph)

    events = await flow.step(until="supervising")
    observed.extend(events)
    print_events("until='supervising': parent has fanned out children", events, graph)

    def first_child_done(event: Event, _graph: Graph) -> bool:
        return (
            event.type == "append_node"
            and event.agent_id != "root"
            and event.node_type == "done_output"
        )

    events = await flow.step(until=first_child_done)
    observed.extend(events)
    print_events("until=<callable>: stop when any child is done", events, graph)

    events = await flow.step(until="node", n=2)
    observed.extend(events)
    print_events("until='node', n=2: consume two more node appends", events, graph)

    while not graph.finished:
        events = await flow.step(until=lambda _event, current: current.finished)
        observed.extend(events)
        print_events("until=<callable>: run is finished", events, graph)

    if observed:
        print(f"\nObserved {len(observed)} graph events.")

    flow.close_repls(graph.graph_id)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show minimal Flow.step(..., until=...) boundaries."
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
    print("Delegated children fan out under the shared pool.")
    print("The caller chooses observation boundaries with step(..., until=...).")
    print(f"Result: {graph.result()}")
    _save_example_graph(graph, __file__, "step-until", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
