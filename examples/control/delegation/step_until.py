"""Show minimal ``Flow.run_streaming(..., until=...)`` boundaries during delegation.

Delegation fans out child agents as independent asyncio tasks; ``Flow(workers=n)``
caps how many model calls run at once. This example focuses on the caller-facing
control surface: choosing how much of the node stream to consume with
``until=...``.

A halt cancels whatever is still running. A parent halted part-way through a
delegating node leaves that node ready to run again: the children it launched
stay, and when the stream resumes the node re-runs in place, getting the same
children back from its saved launches.

Run:
    export OPENAI_API_KEY=...
    python examples/control/delegation/step_until.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import AgentConfig, Flow, Graph, Node, Step
from rlmflow.view.text import kind

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


async def run_example(args: argparse.Namespace) -> tuple[Graph, Node]:
    config = AgentConfig(model=args.model, max_depth=args.max_depth, max_iters=args.max_iters)
    flow = Flow(configs=[config], workers=args.max_concurrency)
    g = Graph()
    root = g.start(QUERY)
    observed: list[str] = []
    out_dir = Path(args.out_dir)

    async def collect(**flow_kwargs) -> list[str]:
        """A label per landed node, taken as it lands: a halt can take nodes back."""
        stream = checkpoint_stream(flow.run_streaming(g, **flow_kwargs), g, out_dir)
        return [f"{g.path(step.agent)}: {kind(step.node)}" async for step in stream]

    events = await collect(until="next")
    observed.extend(events)
    print_nodes("until='next': first appended node", events, g, root)

    def child_stepped(step: Step) -> bool:
        return step.agent.id != root.id

    events = await collect(until=child_stepped)
    observed.extend(events)
    print_nodes("until=<callable>: a child has taken its first step", events, g, root)

    def first_child_done(step: Step) -> bool:
        return step.agent.id != root.id and step.node.data.get("terminal", False)

    events = await collect(until=first_child_done)
    observed.extend(events)
    print_nodes("until=<callable>: stop when any child is done", events, g, root)

    seen = 0

    def two_more_appends(_step: Step) -> bool:
        nonlocal seen
        seen += 1
        return seen >= 2

    events = await collect(until=two_more_appends)
    observed.extend(events)
    print_nodes("callable boundary: consume two more global steps", events, g, root)

    while not g.status(root) == "done":
        events = await collect(until="done")
        print_nodes("until='done': run to the end", events, g, root)
        observed.extend(events)

    if observed:
        print(f"\nObserved {len(observed)} nodes.")

    await flow.aclose()
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
    print("Delegated children fan out as independent asyncio tasks.")
    print("The caller chooses observation boundaries with run_streaming(..., until=...).")
    print(f"Result: {g.result(root)}")
    save_example_graph(g, "step-until", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
