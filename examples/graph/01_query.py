"""Querying a Node trajectory."""

from __future__ import annotations

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    Node,
    start,
)


def build_graph() -> AgentStart:
    """Hand-build the tree a delegating run would have recorded."""
    root = start(query="ship a tiny package")
    turn = LLMOutput(
        content="splitting into two children",
        code=(
            "write = launch_subagent(task='write module')\n"
            "test = launch_subagent(task='run pytest')"
        ),
        usage=LLMUsage(input_tokens=120, output_tokens=40),
    )
    root.append(turn)
    # Children hang off the action that launched them; the agent's own sequel is
    # the output of that same step.
    launch = ExecAction(code="write = launch_subagent(task=...)")
    turn.append(launch)

    writer = AgentStart(content="write module", config=root.config.child("write"))
    launch.append(writer)
    writer.append(DoneOutput(result="wrote pkg/__init__.py"))

    tester = AgentStart(content="run pytest", config=root.config.child("test"))
    launch.append(tester)
    failed = ErrorOutput(error="exec_error", content="ModuleNotFoundError: pkg")
    tester.append(failed)
    failed.append(DoneOutput(result="3 passed"))

    output = ExecOutput(content="['wrote pkg/__init__.py', '3 passed']")
    launch.append(output)
    output.append(DoneOutput(result="package shipped"))
    return root


def banner(title: str) -> None:
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def main() -> None:
    root = build_graph()
    nodes = list(root.walk())

    banner("agents / nodes")
    print("agents:", [node.config.path for node in nodes if isinstance(node, AgentStart)])
    print("node types:", [node.type for node in nodes])
    print(
        "errors:",
        [(node.parent_agent.config.path, node.error) for node in nodes if _failed(node)],
    )

    banner("frontier / result / tokens")
    print("root frontier:", root.frontier.type)
    print("result:", root.result())
    print("tokens:", root.usage)


def _failed(node: Node) -> bool:
    return isinstance(node, ErrorOutput)


if __name__ == "__main__":
    main()
