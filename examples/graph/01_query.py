"""Querying a graph: nodes, agents, errors, results, and token usage."""

from __future__ import annotations

from rlmflow import AGENT_START, CODE, DONE, LLM_OUTPUT, Graph, LLMUsage, Node


def build_graph() -> tuple[Graph, Node]:
    """Hand-build the graph a delegating run would have recorded."""
    g = Graph()
    root = g.spawn("ship a tiny package", name="root")
    # Children are launched by the model turn whose cell called launch_agent.
    turn = g.append(
        root,
        Node(
            LLM_OUTPUT,
            text=(
                "splitting into two children\n```repl\n"
                "write = launch_agent('write module')\n"
                "test = launch_agent('run pytest')\n"
                "print(await write, await test)\n```"
            ),
            usage=LLMUsage(input_tokens=120, output_tokens=40).as_dict(),
        ),
    )

    writer = g.spawn("write module", by=turn, name="write")
    g.append(writer, Node(DONE, result="wrote pkg/__init__.py"))

    tester = g.spawn("run pytest", by=turn, name="test")
    g.append(tester, Node(CODE, code="import pkg", output="ModuleNotFoundError: pkg", error=True)
    )
    g.append(tester, Node(DONE, result="3 passed"))

    g.append(
        root,
        Node(CODE, code="print(await write, await test)", output="wrote pkg/__init__.py 3 passed"),
    )
    g.append(root, Node(DONE, result="package shipped"))
    return g, root


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    g, root = build_graph()

    banner("agents / nodes")
    print("agents:", [g.path(agent) for agent in g.nodes(AGENT_START)])
    print("node types:", [node.type for node in g.nodes()])
    print(
        "errors:",
        [
            (g.path(g.node(node.agent_id)), node["output"])
            for node in g.nodes(CODE)
            if node.get("error")
        ],
    )

    banner("frontier / result / tokens")
    print("root frontier:", g.frontier(root).type)
    print("result:", g.result(root))
    print("tokens:", g.usage())


if __name__ == "__main__":
    main()
