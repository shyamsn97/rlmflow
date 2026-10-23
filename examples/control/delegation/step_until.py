"""Show minimal ``Flow.run_streaming(..., until=...)`` boundaries during delegation.

Minimal Flow does not expose the old ``eager_children`` toggle. Delegation fans
out child agents as independent asyncio tasks; blocking LLM calls are bounded
by the flow's worker pool. This example focuses on the caller-facing control
surface instead: choosing how much of the Node stream to consume with
``until=...``.

A parent halted part-way through a delegating block re-runs that block when the
stream resumes; its launch calls resolve to the children already in the graph,
so nothing is launched twice.

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
    AgentConfig,
    AgentStart,
    Flow,
    Graph,
    Node,
    reads,
)
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream, save_example_graph  # noqa: E402

QUERY = """\
Show minimal `Flow.run_streaming(..., until=...)` boundaries with delegated child agents.

In your first REPL block, launch two child handles before waiting for either,
then collect both results and call submit. Use exactly these child names:

- `slow`: ask it to solve a tiny task after sleeping briefly with
  `await asyncio.sleep(1)`.
- `fast`: ask it to solve a tiny task immediately.

This example script is observing the graph stream, so keep the work small.
"""


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


async def run_example(args: argparse.Namespace) -> tuple[Graph, AgentStart]:
    flow = Flow(client_for(args.model), workers=args.max_concurrency)
    g = Graph()
    root = g.create_agent(
        QUERY,
        config=AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
    )
    observed: list[Node] = []
    out_dir = Path(args.out_dir)

    async def collect(**flow_kwargs) -> list[Node]:
        stream = checkpoint_stream(flow.run_streaming(g, **flow_kwargs), g, out_dir)
        return [item async for item in stream if isinstance(item, Node)]

    events = await collect(until="next")
    observed.extend(events)
    print_nodes("until='next': first appended node", events, g, root)

    def child_launched(item, _g: Graph) -> bool:
        # The first child AgentStart is the fan-out.
        return isinstance(item, AgentStart) and item is not root

    events = await collect(until=child_launched)
    observed.extend(events)
    print_nodes("until=<callable>: parent has fanned out a child", events, g, root)

    def first_child_done(item, _g: Graph) -> bool:
        return isinstance(item, Node) and item.agent_id != root.id and item.type == "done_output"

    events = await collect(until=first_child_done)
    observed.extend(events)
    print_nodes("until=<callable>: stop when any child is done", events, g, root)

    seen = 0

    def two_more_appends(item, _g: Graph) -> bool:
        nonlocal seen
        if isinstance(item, Node):
            seen += 1
        return seen >= 2

    events = await collect(until=two_more_appends)
    observed.extend(events)
    print_nodes("callable boundary: consume two more global steps", events, g, root)

    while not reads.terminal(g, root):
        events = await collect(until="finished")
        print_nodes("until='finished': run to the end", events, g, root)
        observed.extend(events)

    if observed:
        print(f"\nObserved {len(observed)} Nodes.")

    flow.runtime.close_repls()
    return g, root


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

    g, root = asyncio.run(run_example(args))
    print("\n=== verdict ===")
    print("Delegated children fan out as independent scheduler tasks.")
    print("The caller chooses observation boundaries with run_streaming(..., until=...).")
    print(f"Result: {reads.result(g, root)}")
    save_example_graph(g, "step-until", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
