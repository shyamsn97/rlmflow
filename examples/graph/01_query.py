"""Querying a graph: agents, nodes, edges, errors, results, and token usage."""

from __future__ import annotations

import time

from rlmflow import RETURN, Graph, LLMUsage, Node
from rlmflow.view.text import kind


def turn(g: Graph, agent: Node, text: str = "", **fields) -> Node:
    """Append one finished model turn to ``agent``: the reply, and the code it ran."""
    node = Node("assistant", text, seen=len(g.history(agent)), **fields)
    node.finished_at = time.time()
    return g.append(agent, node)


def build_graph() -> tuple[Graph, Node]:
    """Hand-build the graph a delegating run would have recorded."""
    g = Graph()
    root = g.add_agent("ship a tiny package", name="root")
    code = (
        "write = launch_agent('write module')\n"
        "test = launch_agent('run pytest')\n"
        "print(await write, await test)"
    )
    # Children are launched by the node whose code called launch_agent.
    launch = turn(
        g,
        root,
        f"splitting into two children\n```repl\n{code}\n```",
        code=code,
        output="wrote pkg/__init__.py 3 passed",
        usage=LLMUsage(input_tokens=120, output_tokens=40).as_dict(),
    )

    writer = g.add_agent("write module", parent=launch, name="write")
    wrote = turn(g, writer, result="wrote pkg/__init__.py", terminal=True)

    tester = g.add_agent("run pytest", parent=launch, name="test")
    turn(g, tester, code="import pkg", output="ModuleNotFoundError: pkg", error=True)
    passed = turn(g, tester, result="3 passed", terminal=True)

    # Each child's final node links to the node whose ``await`` read its result.
    g.link(RETURN, wrote, launch)
    g.link(RETURN, passed, launch)
    turn(g, root, result="package shipped", terminal=True)
    return g, root


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    g, root = build_graph()

    banner("agents / nodes / edges")
    print("agents:", [g.path(agent) for agent in g.agents()])
    print("node kinds:", [kind(node) for node in g.nodes()])
    print("edges:", [(edge.kind, g.path(g.agent(edge.target))) for edge in g.edges(kind="spawn")])
    print("errors:", [(g.path(g.owner(node)), node.output) for node in g.nodes() if node.error])

    banner("tail / status / result / tokens")
    print("root tail:", kind(g.tail(root)))
    print("statuses:", {agent.root_data.name: g.status(agent) for agent in g.agents()})
    print("result:", g.result(root))
    print("tokens:", g.usage())


if __name__ == "__main__":
    main()
