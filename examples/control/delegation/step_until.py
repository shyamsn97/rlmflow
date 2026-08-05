"""Show minimal ``Flow.run_streaming(..., until=...)`` boundaries during delegation.

Minimal Flow does not expose the old ``eager_children`` toggle. Delegation fans
out child agents as independent asyncio tasks; blocking LLM calls are bounded
by the flow's worker pool. This example focuses on the caller-facing control
surface instead: choosing how much of the Node stream to consume with
``until=...``.

Known limitation: a parent halted part-way through a delegating block re-runs
that block when the stream resumes, and a child it already launched is refused
by name, so the root here usually lands on ``[max_iters exceeded]`` rather than
a clean answer. The boundaries themselves are what this example is showing.

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
    AgentStart,
    Flow,
    Node,
    start,
)

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


async def run_example(args: argparse.Namespace) -> AgentStart:
    flow = Flow(build_client(args.model), workers=args.max_concurrency)
    graph = start(query=QUERY, max_depth=args.max_depth, max_iters=args.max_iters)
    observed: list[Node] = []
    out_dir = Path(args.out_dir)

    async def collect(graph: AgentStart, **flow_kwargs) -> list[Node]:
        nodes: list[Node] = []
        async for node in flow.run_streaming(graph, **flow_kwargs):
            graph.save(out_dir)  # checkpoint as nodes land
            nodes.append(node)
        return nodes

    events = await collect(graph, until="next")
    observed.extend(events)
    print_nodes("until='next': first appended node", events, graph)

    def child_launched(node: Node, root: Node) -> bool:
        # A delegating turn hangs each child off the exec_action that launched it,
        # so the first child AgentStart is the fan-out.
        return isinstance(node, AgentStart) and node is not root

    events = await collect(graph, until=child_launched)
    observed.extend(events)
    print_nodes("until=<callable>: parent has fanned out a child", events, graph)

    def first_child_done(node: Node, root: Node) -> bool:
        return node.parent_agent is not root and node.type == "done_output"

    events = await collect(graph, until=first_child_done)
    observed.extend(events)
    print_nodes("until=<callable>: stop when any child is done", events, graph)

    seen = 0

    def two_more_appends(_node: Node, _root: Node) -> bool:
        nonlocal seen
        seen += 1
        return seen >= 2

    events = await collect(graph, until=two_more_appends)
    observed.extend(events)
    print_nodes("callable boundary: consume two more global steps", events, graph)

    while not graph.terminal:
        events = await collect(graph, until="finished")
        print_nodes("until='finished': run to the end", events, graph)
        observed.extend(events)

    if observed:
        print(f"\nObserved {len(observed)} Nodes.")

    flow.runtime.close_repls()
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
