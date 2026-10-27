"""Decentralized peers: equal agents in one graph, talking over edges they draw.

There is no root agent. Three peers are top-level agents in one graph, each
holding a private number. ``alice`` draws ``peer`` edges to the others with
``connect_agent``, asks each for its number with ``ask_agent`` (which waits for
the peer's next finished turn), then shares the total with ``send_agent``
(which does not wait). Every message is a ``message`` node with a ``from``
edge to the turn that sent it, so the saved graph records who told whom what.

    python examples/collaboration/decentralized_peers.py --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rlmflow import AGENT_START, MESSAGE, AgentConfig, Flow, Graph, GraphCheckpointer, Step
from rlmflow.graph.graph import FROM
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import add_model_args, example_run_dir  # noqa: E402

VALUES = {"alice": 1, "bob": 2, "carol": 3}

PEERS = "You are {name}, one of three equal peers (alice, bob, carol) in one agent graph. There is no manager. Your private number is in the REPL variable `value`."

COLLECTOR = """
Find the sum of all three private numbers:
1. Draw a "peer" edge to bob and to carol with connect_agent.
2. Ask both for their number with ask_agent, concurrently via asyncio.gather.
3. Add their numbers to yours.
4. Tell bob and carol the total with send_agent.
5. submit the total as an integer."""

RESPONDER = """
Submit your number now with submit(value). Peers may message you later:
- asked for your number: submit(value) again;
- told the total: submit the string "{name} knows the total is N"."""


async def run(flow: Flow, g: Graph, out_dir: str) -> None:
    """Drive every peer in ``g`` until none can step, checkpointing as items land."""
    checkpointer = GraphCheckpointer(out_dir)
    try:
        async for item in flow.run_streaming(g):
            checkpointer.handle(item, g)
            if isinstance(item, Step):
                print(f"{item.agent.name:<6}  {item.node.type}")
    finally:
        checkpointer.close()
        flow.runtime.close_repls()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--out-dir", default=str(example_run_dir("decentralized-peers")))
    args = parser.parse_args()

    flow = Flow(client_for(args.model))
    g = Graph()
    for name, value in VALUES.items():
        role = COLLECTOR if name == "alice" else RESPONDER.format(name=name)
        g.create_agent(
            PEERS.format(name=name) + role,
            config=AgentConfig(max_depth=0, max_iters=args.max_iters),
            name=name,
            value=value,
        )

    asyncio.run(run(flow, g, args.out_dir))

    print("\nresults:")
    results = {peer.name: g.result(peer) for peer in g.nodes(AGENT_START)}
    for name, result in results.items():
        print(f"  {name}: {result!r}")

    print("\nedges the peers drew:")
    for edge in g.edges():
        if not edge.structural:
            print(
                f"  {g.node(edge.source).name} -{edge.kind}-> {g.node(edge.target).name}"
            )

    print("\nmessages:")
    for message in g.nodes(MESSAGE):
        senders = [g.node(edge.target) for edge in g.edges(kind=FROM, source=message.id)]
        if not senders:
            continue
        frm = ", ".join(g.node(sender.agent_id).name for sender in senders)
        print(f"  {frm} -> {g.node(message.agent_id).name}: {message['text']}")

    print(f"\nexpected total {sum(VALUES.values())}, alice submitted {results['alice']!r}")
    print(f"Graph saved to {args.out_dir}")


if __name__ == "__main__":
    main()
