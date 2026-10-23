"""Inject controller-authored nodes into a minimal Flow run.

The minimal controller workflow:

1. The caller owns the graph and the agents it starts in it.
2. The controller writes observations into an agent's history itself, with
   ``ops.append(g, agent, ...)``, or queues a message with ``ops.send``.
3. "Finalize now" is just another controller instruction in the graph.

Run:
    python examples/control/controller_injection.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rlmflow import (
    AgentConfig,
    AgentStart,
    ExecOutput,
    Flow,
    Graph,
    LLMUsage,
    Step,
    ops,
    reads,
    running_step,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream, example_run_dir, save_example_graph  # noqa: E402

OBSERVATION = "Injected controller observation: finalize using this note."


class DemoLLM:
    """Deterministic model so the example runs offline."""

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        convo = "\n".join(m["content"] for m in messages)
        if "Injected controller observation" in convo:
            return '```repl\nsubmit("used the injected controller observation")\n```'
        if "Controller stop request" in convo:
            return '```repl\nsubmit("controller stopped the run")\n```'
        return '```repl\nprint("waiting for controller input")\n```'


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def state_types(g: Graph, agent: AgentStart) -> list[str]:
    return [state.type for state in reads.history(g, agent)]


def assert_types(g: Graph, agent: AgentStart, expected: list[str]) -> None:
    actual = state_types(g, agent)
    assert actual == expected, f"expected states {expected}, got {actual}"


def print_states(label: str, g: Graph, agent: AgentStart) -> None:
    print(f"\n{label}")
    print("state types:", " -> ".join(state_types(g, agent)))


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    flow = Flow(DemoLLM())
    g = Graph()
    agent = g.create_agent(
        "Wait for a controller note, then submit.",
        config=AgentConfig(max_depth=0, max_iters=4),
    )
    assert_types(g, agent, ["agent_start"])

    extra = ops.append(g, agent, ExecOutput(content=OBSERVATION))
    assert_types(g, agent, ["agent_start", "exec_output"])
    assert reads.frontier(g, agent) is extra

    print_states("after the controller appended its observation", g, agent)

    with running_step(Step(g, agent, extra)):
        projected = flow.build_messages(extra)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation.")

    run_dir = example_run_dir("controller-injection") / "observation-injection"

    async def drive() -> None:
        async for _item in checkpoint_stream(flow.run_streaming(g), g, run_dir):
            pass

    asyncio.run(drive())
    assert reads.result(g, agent) == "used the injected controller observation"
    print_states("after stepping: run reacted and finished", g, agent)
    print(f"result={reads.result(g, agent)!r}")
    save_example_graph(
        g,
        "controller-injection",
        out_dir=run_dir,
    )


def controller_stop_instruction() -> None:
    banner("2. Inject a controller stop instruction")

    flow = Flow(DemoLLM())
    g = Graph()
    agent = g.create_agent(
        "This run will be stopped by the controller.",
        config=AgentConfig(max_depth=0, max_iters=4),
    )
    run_dir = example_run_dir("controller-injection") / "controller-stop-instruction"

    async def run_with_controller_stop() -> None:
        # Stream the run to its first resting observation, then react. Because
        # ``until`` halts the run at the boundary (the driver does not run ahead),
        # the instruction we inject is guaranteed to be the next thing the agent
        # reads when we resume it.
        async for _item in checkpoint_stream(flow.run_streaming(g, until="idle"), g, run_dir):
            pass
        ops.send(g, agent, "Controller stop request: finalize now with current state.")
        async for _item in checkpoint_stream(flow.run_streaming(g, until="done"), g, run_dir):
            pass

    asyncio.run(run_with_controller_stop())
    assert reads.result(g, agent) == "controller stopped the run"
    print_states("after controller stop instruction: clean submit(...)", g, agent)
    print(f"result={reads.result(g, agent)!r}")
    save_example_graph(
        g,
        "controller-injection",
        out_dir=run_dir,
    )


def main() -> None:
    observation_injection()
    controller_stop_instruction()


if __name__ == "__main__":
    main()
