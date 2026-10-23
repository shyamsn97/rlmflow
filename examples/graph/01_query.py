"""Querying a graph: nodes, agents, errors, results, and token usage."""

from __future__ import annotations

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    LLMUsage,
    ops,
    reads,
)


def build_graph() -> tuple[Graph, AgentStart]:
    """Hand-build the graph a delegating run would have recorded."""
    g = Graph()
    root = ops.spawn(g, "ship a tiny package")
    ops.append(
        g,
        root,
        LLMOutput(
            content="splitting into two children",
            code=(
                "write = launch_subagent(task='write module')\n"
                "test = launch_subagent(task='run pytest')"
            ),
            usage=LLMUsage(input_tokens=120, output_tokens=40),
        ),
    )
    # Children are launched by the action; the agent's own sequel is the output
    # of that same step.
    launch = ops.append(g, root, ExecAction(code="write = launch_subagent(task=...)"))

    writer = ops.spawn(g, "write module", by=launch, name="write")
    ops.append(g, writer, DoneOutput(result="wrote pkg/__init__.py"))

    tester = ops.spawn(g, "run pytest", by=launch, name="test")
    ops.append(g, tester, ErrorOutput(error="exec_error", content="ModuleNotFoundError: pkg"))
    ops.append(g, tester, DoneOutput(result="3 passed"))

    ops.append(g, root, ExecOutput(content="['wrote pkg/__init__.py', '3 passed']"))
    ops.append(g, root, DoneOutput(result="package shipped"))
    return g, root


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    g, root = build_graph()

    banner("agents / nodes")
    print("agents:", [reads.path(g, agent) for agent in g.nodes(AgentStart)])
    print("node types:", [node.type for node in g.nodes()])
    print(
        "errors:",
        [(reads.path(g, g.node(node.agent_id)), node.error) for node in g.nodes(ErrorOutput)],
    )

    banner("frontier / result / tokens")
    print("root frontier:", reads.frontier(g, root).type)
    print("result:", reads.result(g, root))
    print("tokens:", reads.usage(g))


if __name__ == "__main__":
    main()
