"""Querying a minimal Graph."""

from __future__ import annotations

from rflow.minimal import DoneOutput, ErrorOutput, Graph, LLMOutput, UserQuery


def build_graph() -> Graph:
    graph = Graph(query="ship a tiny package")
    graph.commit(UserQuery(content=graph.query))
    graph.commit(
        LLMOutput(
            content="splitting into two children",
            metadata={"usage": {"input_tokens": 120, "output_tokens": 40}},
        )
    )
    graph.commit(DoneOutput(result="package shipped"))

    write = Graph(
        agent_id="root.write",
        query="write module",
        depth=1,
        parent_agent_id="root",
    )
    write.commit(UserQuery(content=write.query))
    write.commit(DoneOutput(result="wrote pkg/__init__.py"))

    test = Graph(
        agent_id="root.test",
        query="run pytest",
        depth=1,
        parent_agent_id="root",
    )
    test.commit(UserQuery(content=test.query))
    test.commit(ErrorOutput(error="exec_error", output="ModuleNotFoundError: pkg"))
    test.commit(DoneOutput(result="3 passed"))

    graph.children = {write.agent_id: write, test.agent_id: test}
    return graph


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    graph = build_graph()
    nodes = [node for agent in graph.walk() for node in agent.nodes]

    banner("agents / nodes")
    print("agents:", list(graph.agents))
    print("node types:", [node.type for node in nodes])
    print("errors:", [(node.agent_id, node.error) for node in nodes if isinstance(node, ErrorOutput)])

    banner("current / result / tokens")
    print("root current:", graph.current().type if graph.current() else None)
    print("result:", graph.result())
    print("tokens:", graph.tokens())


if __name__ == "__main__":
    main()
