"""Querying a minimal Node trajectory."""

from __future__ import annotations

from rlmflow import (
    DoneOutput,
    ErrorOutput,
    LLMOutput,
    Node,
    UserQuery,
    start,
)


def build_graph() -> Node:
    graph = start(query="ship a tiny package")
    split = LLMOutput(
        content="splitting into two children",
        metadata={"usage": {"input_tokens": 120, "output_tokens": 40}},
    )
    graph.attach(split)

    writer = split.attach(UserQuery(agent_id="root.write", content="write module"))
    writer.attach(DoneOutput(result="wrote pkg/__init__.py"))

    tester = split.attach(UserQuery(agent_id="root.test", content="run pytest"))
    error = tester.attach(
        ErrorOutput(error="exec_error", output="ModuleNotFoundError: pkg"),
    )
    error.attach(DoneOutput(result="3 passed"))

    split.attach(DoneOutput(result="package shipped"))
    return graph


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    graph = build_graph()
    nodes = list(graph.walk())

    banner("agents / nodes")
    print("agents:", list(graph.agent_ids()))
    print("node types:", [node.type for node in nodes])
    print(
        "errors:", [(node.agent_id, node.error) for node in nodes if isinstance(node, ErrorOutput)]
    )

    banner("current / result / tokens")
    print("root current:", graph.tail("root").type)
    print("result:", graph.agent_result())
    print("tokens:", graph.tokens())


if __name__ == "__main__":
    main()
