"""Querying a graph: agents, nodes, edges, errors, results, and token usage."""

from __future__ import annotations

from rlmflow import CODE, DONE, LLM, Agent, Graph, LLMUsage, Node


def build_graph() -> tuple[Graph, Agent]:
    """Hand-build the graph a delegating run would have recorded."""
    g = Graph()
    root = g.add_agent("ship a tiny package", name="root")
    g.append(
        root,
        Node(
            LLM,
            text=(
                "splitting into two children\n```repl\n"
                "write = launch_agent('write module')\n"
                "test = launch_agent('run pytest')\n"
                "print(await write, await test)\n```"
            ),
            usage=LLMUsage(input_tokens=120, output_tokens=40).as_dict(),
        ),
    )
    # Children are launched by the cell that called launch_agent.
    cell = g.append(
        root,
        Node(CODE, code="print(await write, await test)", output="wrote pkg/__init__.py 3 passed"),
    )

    writer = g.add_agent("write module", parent=cell, name="write")
    g.append(writer, Node(DONE, result="wrote pkg/__init__.py"))

    tester = g.add_agent("run pytest", parent=cell, name="test")
    g.append(tester, Node(CODE, code="import pkg", output="ModuleNotFoundError: pkg", error=True))
    g.append(tester, Node(DONE, result="3 passed"))

    g.append(root, Node(DONE, result="package shipped"))
    return g, root


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    g, root = build_graph()

    banner("agents / nodes / edges")
    print("agents:", [g.path(agent) for agent in g.agents()])
    print("node types:", [node.type for node in g.nodes()])
    print("edges:", [(edge.kind, g.path(g.agent(edge.target))) for edge in g.edges(kind="spawn")])
    print(
        "errors:",
        [(g.path(g.owner(node)), node["output"]) for node in g.nodes(CODE) if node.get("error")],
    )

    banner("frontier / status / result / tokens")
    print("root frontier:", g.history(root)[-1].type)
    print("statuses:", {agent.name: g.status(agent) for agent in g.agents()})
    print("result:", g.result(root))
    print("tokens:", g.usage())


if __name__ == "__main__":
    main()
