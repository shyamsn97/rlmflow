"""Inject controller-authored graph actions into a minimal Flow run.

The minimal controller workflow:

1. The caller owns the root ``Node``.
2. The controller attaches observations with ``flow.append(...)``.
3. "Finalize now" is just another controller instruction in the graph.

Run:
    python examples/control/controller_injection.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rlmflow import (
    ExecOutput,
    Flow,
    LLMUsage,
    Node,
    start,
)
from rlmflow.consumers import GraphCheckpointer

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, save_example_graph  # noqa: E402


OBSERVATION = "Injected controller observation: finalize using this note."


class DemoLLM:
    """Deterministic model so the example runs offline."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        convo = "\n".join(m["content"] for m in messages)
        if "Injected controller observation" in convo:
            return '```repl\ndone("used the injected controller observation")\n```'
        if "Controller stop request" in convo:
            return '```repl\ndone("controller stopped the run")\n```'
        return '```repl\nprint("waiting for controller input")\n```'


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def state_types(graph: Node) -> list[str]:
    return [state.type for state in graph.transcript()]


def assert_types(graph: Node, expected: list[str]) -> None:
    actual = state_types(graph)
    assert actual == expected, f"expected states {expected}, got {actual}"


def print_states(label: str, graph: Node) -> None:
    print(f"\n{label}")
    print("state types:", " -> ".join(state_types(graph)))


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    flow = Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = start(query="Wait for a controller note, then finish.")
    assert_types(graph, ["user_query"])

    flow.append(
        graph.tail(),
        ExecOutput(output=OBSERVATION, content=OBSERVATION),
        injected=True,
    )
    assert_types(graph, ["user_query", "exec_output"])

    extra = graph.tail()
    assert isinstance(extra, ExecOutput)
    assert extra.metadata.get("injected") is True

    print_states("after controller commit(...)", graph)

    projected = flow.messages(graph)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation.")

    run_dir = example_run_dir("controller-injection") / "observation-injection"
    checkpointer = GraphCheckpointer(run_dir)

    async def drive() -> None:
        try:
            async for event in flow.run_streaming(graph):
                checkpointer.handle(event)
        finally:
            checkpointer.close()

    asyncio.run(drive())
    assert graph.agent_result() == "used the injected controller observation"
    print_states("after stepping: run reacted and finished", graph)
    print(f"result={graph.agent_result()!r}")
    save_example_graph(
        graph,
        "controller-injection",
        out_dir=run_dir,
    )


def controller_stop_instruction() -> None:
    banner("2. Inject a controller stop instruction")

    flow = Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = start(query="This run will be stopped by the controller.")
    run_dir = example_run_dir("controller-injection") / "controller-stop-instruction"
    checkpointer = GraphCheckpointer(run_dir)

    async def run_with_controller_stop() -> None:
        # Stream the run to its first resting observation, then react. Because
        # ``until`` halts the run at the boundary (the driver does not run ahead),
        # the instruction we inject is guaranteed to be the next thing the agent
        # reads when we resume it.
        try:
            async for event in flow.run_streaming(graph, until="idle"):
                checkpointer.handle(event)
            flow.append(
                graph.tail(),
                "Controller stop request: finalize now with current state.",
                injected=True,
            )
            async for event in flow.run_streaming(graph, until="done"):
                checkpointer.handle(event)
        finally:
            checkpointer.close()

    asyncio.run(run_with_controller_stop())
    assert graph.agent_result() == "controller stopped the run"
    print_states("after controller stop instruction: clean done(...)", graph)
    print(f"result={graph.agent_result()!r}")
    save_example_graph(
        graph,
        "controller-injection",
        out_dir=run_dir,
    )


def main() -> None:
    observation_injection()
    controller_stop_instruction()


if __name__ == "__main__":
    main()
