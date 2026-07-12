"""Inject controller-authored graph actions into a minimal Flow run.

The minimal controller workflow:

1. The caller owns the ``Graph``.
2. The controller applies graph actions with ``flow.append_node(...)`` or
   ``flow.apply_action(...)``.
3. "Finalize now" is just another controller instruction in the graph.

Run:
    python examples/control/controller_injection.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rflow import ExecOutput, Flow, Graph, LLMUsage


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path


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


def state_types(graph) -> list[str]:
    return [state.type for state in graph.nodes]


def assert_types(graph, expected: list[str]) -> None:
    actual = state_types(graph)
    assert actual == expected, f"expected states {expected}, got {actual}"


def print_states(label: str, graph) -> None:
    print(f"\n{label}")
    print("state types:", " -> ".join(state_types(graph)))


def observation_injection() -> None:
    banner("1. Inject an observation and let the LLM react")

    flow = Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = flow.start(Graph(query="Wait for a controller note, then finish."))
    assert_types(graph, ["user_query"])

    flow.append_node(graph, ExecOutput(output=OBSERVATION, content=OBSERVATION))
    assert_types(graph, ["user_query", "exec_output"])

    extra = graph.nodes[-1]
    assert isinstance(extra, ExecOutput)
    assert "injected" not in set(extra.metadata)

    print_states("after controller append_node(...)", graph)

    projected = flow.messages(graph)[-1]["content"]
    assert OBSERVATION in projected
    print("message projection contains the controller observation.")

    flow.run(graph)
    assert graph.result() == "used the injected controller observation"
    print_states("after stepping: run reacted and finished", graph)
    print(f"result={graph.result()!r}")
    _save_example_graph(
        graph,
        __file__,
        "controller-injection",
        out_dir=_example_run_dir(__file__, "controller-injection")
        / "observation-injection",
    )


def controller_stop_instruction() -> None:
    banner("2. Inject a controller stop instruction")

    flow = Flow(DemoLLM(), max_depth=0, max_iters=4)
    graph = Graph(query="This run will be stopped by the controller.")

    async def run_with_controller_stop() -> None:
        # Stream the run to its first resting observation, then react. Because
        # ``until`` halts the run at the boundary (the driver does not run ahead),
        # the instruction we inject is guaranteed to be the next thing the agent
        # reads when we resume it.
        async for _event in flow.run_streaming(graph, until="idle"):
            pass
        graph.inject("Controller stop request: finalize now with current state.")
        async for _event in flow.run_streaming(until="done"):
            pass

    asyncio.run(run_with_controller_stop())
    assert graph.result() == "controller stopped the run"
    print_states("after controller stop instruction: clean done(...)", graph)
    print(f"result={graph.result()!r}")
    _save_example_graph(
        graph,
        __file__,
        "controller-injection",
        out_dir=_example_run_dir(__file__, "controller-injection")
        / "controller-stop-instruction",
    )


def main() -> None:
    observation_injection()
    controller_stop_instruction()


if __name__ == "__main__":
    main()
