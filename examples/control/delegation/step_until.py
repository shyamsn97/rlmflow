"""Show minimal ``Flow.run_streaming(..., until=...)`` boundaries during delegation.

Minimal Flow does not expose the old ``eager_children`` toggle. Delegation fans
out child agents as independent asyncio tasks; blocking LLM calls are bounded
by the flow's worker pool. This example focuses on the caller-facing control
surface instead: choosing how much of the Node stream to consume with
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

from rlmflow import (
    Flow,
    Node,
    SupervisingOutput,
    start,
)
from rlmflow.consumers import GraphCheckpointer
from rlmflow.view import render_tree

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

This example script is observing the Node stream, so keep the work small.
"""


def node_label(node: Node) -> str:
    mutation = node.metadata.get("mutation", {}).get("type", "append")
    return f"{node.agent_id}: {mutation} {node.type}"


def print_nodes(title: str, nodes: list[Node], graph: Node) -> None:
    print(f"\n=== {title} ===")
    for node in nodes:
        print(f"- {node_label(node)}")
    print(render_tree(graph))


async def run_example(args: argparse.Namespace) -> Node:
    flow = Flow(
        build_client(args.model),
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        workers=args.max_concurrency,
    )
    graph = start(query=QUERY)
    observed: list[Node] = []
    checkpointer = GraphCheckpointer(Path(args.out_dir))

    async def collect(graph: Node, **flow_kwargs) -> list[Node]:
        nodes: list[Node] = []
        async for node in flow.run_streaming(graph, **flow_kwargs):
            checkpointer.handle(node)
            nodes.append(node)
        return nodes

    events = await collect(graph, until="next")
    observed.extend(events)
    print_nodes("until='next': first appended node", events, graph)

    events = await collect(
        graph,
        until=lambda node, _root: isinstance(node, SupervisingOutput),
    )
    observed.extend(events)
    print_nodes("until='supervising': parent has fanned out children", events, graph)

    def first_child_done(node: Node, graph: Node) -> bool:
        return node.agent_id != graph.agent_id and node.type == "done_output"

    events = await collect(graph, until=first_child_done)
    observed.extend(events)
    print_nodes("until=<callable>: stop when any child is done", events, graph)

    seen = 0

    def two_more_appends(node: Node, _graph: Node) -> bool:
        nonlocal seen
        if node.metadata.get("mutation", {}).get("type") == "append":
            seen += 1
        return seen >= 2

    events = await collect(graph, until=two_more_appends)
    observed.extend(events)
    print_nodes("callable boundary: consume two more global steps", events, graph)

    while not graph.finished():
        events = await collect(graph, until=lambda event, current: current.finished())
        observed.extend(events)
        print_nodes("until=<callable>: run is finished", events, graph)

    if observed:
        print(f"\nObserved {len(observed)} Nodes.")

    checkpointer.close()
    flow.runtime.close_repls(graph.trajectory_id)
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
    print(f"Result: {graph.agent_result()}")
    save_example_graph(graph, "step-until", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
