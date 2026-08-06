"""Inject controller-authored nodes into a minimal Flow run.

The minimal controller workflow:

1. The caller owns the root ``AgentStart``.
2. The controller appends observations to an agent's frontier itself, with
   ``agent.frontier.append(...)``.
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
    UserQuery,
)

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

    flow = Flow(DemoLLM())
    graph = flow.start("Wait for a controller note, then finish.", max_depth=0, max_iters=4)
    assert_types(graph, ["agent_start"])

    graph.frontier.append(ExecOutput(content=OBSERVATION))
    assert_types(graph, ["agent_start", "exec_output"])

    extra = graph.frontier
    assert isinstance(extra, ExecOutput)

    print_states("after the controller appended its observation", graph)

    projected = flow.messages(graph.frontier)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation.")

    run_dir = example_run_dir("controller-injection") / "observation-injection"

    async def drive() -> None:
        # Saving as nodes land keeps the run directory current step by step.
        async for _node in flow.run_streaming(graph):
            graph.save(run_dir)

    asyncio.run(drive())
    assert graph.result() == "used the injected controller observation"
    print_states("after stepping: run reacted and finished", graph)
    print(f"result={graph.result()!r}")
    save_example_graph(
        graph,
        "controller-injection",
        out_dir=run_dir,
    )


def controller_stop_instruction() -> None:
    banner("2. Inject a controller stop instruction")

    flow = Flow(DemoLLM())
    graph = flow.start("This run will be stopped by the controller.", max_depth=0, max_iters=4)
    run_dir = example_run_dir("controller-injection") / "controller-stop-instruction"

    async def run_with_controller_stop() -> None:
        # Stream the run to its first resting observation, then react. Because
        # ``until`` halts the run at the boundary (the driver does not run ahead),
        # the instruction we inject is guaranteed to be the next thing the agent
        # reads when we resume it.
        async for _node in flow.run_streaming(graph, until="idle"):
            graph.save(run_dir)
        graph.frontier.append(
            UserQuery(content="Controller stop request: finalize now with current state.")
        )
        async for _node in flow.run_streaming(graph, until="done"):
            graph.save(run_dir)

    asyncio.run(run_with_controller_stop())
    assert graph.result() == "controller stopped the run"
    print_states("after controller stop instruction: clean done(...)", graph)
    print(f"result={graph.result()!r}")
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
