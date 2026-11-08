"""Decentralized peers: equal agents in one graph, talking with ``Agent.send``.

There is no root agent. Three peers are top-level agents in one graph, each
holding a private number. Peers can't await each other: ``alice`` asks bob and
carol for their numbers with ``AGENTS[name].send(...)`` and ends her turn; their
replies arrive as messages and reopen her. Once she has both, she sends them the
total. Every message is a node in the receiver's history with a ``write`` edge
from the node whose code sent it, so the saved graph records who told whom what.

    python examples/collaboration/decentralized_peers.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import WRITE, AgentConfig, Flow, Graph, GraphCheckpointer
from rlmflow.view.text import kind

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir  # noqa: E402

VALUES = {"alice": 1, "bob": 2, "carol": 3}

PEERS = """\
You are {name}, one of three equal peers (alice, bob, carol) in one agent graph. There is
no manager. Your private number is in the REPL variable `value`. You can't await a peer:
send it a message with AGENTS[name].send(text), then end your turn by calling submit(...).
A peer's reply arrives as a message and reopens you, with your REPL as you left it.
"""

COLLECTOR = """
Find the sum of all three private numbers:
1. Ask bob and carol for their number: send each "alice asks for your number", then
   submit("waiting for bob and carol").
2. Each reply reopens you. Keep the numbers you have in a dict `numbers`; while one is
   missing, submit("waiting") again.
3. With both, add them to yours, tell bob and carol the total with AGENTS[name].send(...),
   and submit the total as an integer."""

RESPONDER = """
Submit your number now with submit(value). Peers may message you later:
- asked for your number: reply to the asker with AGENTS[asker].send("{name}: <value>"),
  then submit(value) again;
- told the total: submit the string "{name} knows the total is N"."""


async def run(flow: Flow, g: Graph, out_dir: str, config: AgentConfig) -> None:
    """Drive every peer in ``g`` until none can step, checkpointing as items land."""
    checkpointer = GraphCheckpointer(out_dir)
    try:
        async for step in flow.run_streaming(g, root_config=config):
            checkpointer.handle(step, g)
            print(f"{step.agent.root_data.name:<6}  {kind(step.node)}")
    finally:
        checkpointer.close()
        flow.runtime.close_repls()


def build_graph() -> Graph:
    g = Graph()
    for name, value in VALUES.items():
        role = COLLECTOR if name == "alice" else RESPONDER.format(name=name)
        g.add_agent(PEERS.format(name=name) + role, name=name, value=value)
    return g


def print_messages(g: Graph) -> None:
    """Each message a peer sent: a ``write`` edge from the node that sent it."""
    for edge in g.edges(kind=WRITE):
        message = g.node(edge.target)
        sender = g.owner(g.node(edge.source)).root_data.name
        print(f"  {sender} -> {g.owner(message).root_data.name}: {message.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--out-dir", default=str(example_run_dir("decentralized-peers")))
    args = parser.parse_args()

    flow = Flow()
    g = build_graph()
    asyncio.run(
        run(
            flow,
            g,
            args.out_dir,
            AgentConfig(model=args.model, max_depth=0, max_iters=args.max_iters),
        )
    )

    print("\nresults:")
    results = {peer.root_data.name: g.result(peer) for peer in g.agents()}
    for name, result in results.items():
        print(f"  {name}: {result!r}")

    print("\nmessages:")
    print_messages(g)

    print(f"\nexpected total {sum(VALUES.values())}, alice submitted {results['alice']!r}")
    print(f"Graph saved to {args.out_dir}")


if __name__ == "__main__":
    main()
