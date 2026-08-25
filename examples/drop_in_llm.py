"""Flow as a drop-in LLM.

Wrapping a `Flow` in `FlowLLM` exposes the `LLMClient` protocol (`chat()` /
`completion()`), so you can swap it in anywhere you'd use a raw LLM. Calling
`FlowLLM(flow).chat(messages)` runs the full recursive agent loop under the hood
and returns a plain string — same signature as any other LLM client.

This enables two patterns:

1. **Replace an LLM with an agent.** Any function that takes an `LLMClient`
   (e.g. a summarization helper, a router, a retrieval pipeline) gets agentic
   behavior for free — no code changes.

2. **Nest agents.** An outer `Flow` can use an inner `Flow` as its `llm`.
   The outer agent's every "LLM call" is itself a full recursive sub-agent run.

Run with:
    export OPENAI_API_KEY=...
    python examples/drop_in_llm.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rlmflow import AgentConfig, Flow
from rlmflow.adapters import FlowLLM
from rlmflow.llm import OpenAIClient

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, save_example_graph  # noqa: E402


def ask(llm, question: str) -> str:
    """A generic helper that takes any LLMClient. Doesn't know or care
    whether it got a plain OpenAI client or a full recursive agent."""
    reply = llm.chat([{"role": "user", "content": question}])
    usage = llm.last_usage
    tokens = usage.input_tokens + usage.output_tokens if usage else 0
    print(f"[{type(llm).__name__}] tokens={tokens}")
    return reply


def demo_plain_llm():
    print("=== plain OpenAI client ===")
    llm = OpenAIClient(model="gpt-4o-mini")
    answer = ask(llm, "In one sentence: what is the capital of France?")
    print(answer, "\n")


def demo_flow_as_llm():
    print("=== Flow as LLMClient (drop-in) ===")
    agent = FlowLLM(
        Flow(
            OpenAIClient(model="gpt-4o-mini"),
            root_config=AgentConfig(max_iters=5),
        )
    )
    answer = ask(agent, "Compute 17 * 23 using a ```repl``` block, then call done().")
    print(answer, "\n")
    if agent.last_graph is not None:
        save_example_graph(
            agent.last_graph,
            "drop-in-llm",
            out_dir=example_run_dir("drop-in-llm") / "flow-as-llm",
        )
    agent.close()


def demo_nested_flow():
    print("=== nested Flow (outer agent uses inner agent as its LLM) ===")
    inner = FlowLLM(
        Flow(
            OpenAIClient(model="gpt-4o-mini"),
            root_config=AgentConfig(max_iters=3),
        )
    )
    outer = Flow(inner)
    root = outer.start("What's the 7th Fibonacci number? Use ```repl``` to compute.", max_iters=3)
    run_dir = example_run_dir("drop-in-llm") / "nested-flow"

    async def drive() -> None:
        async for node in outer.run_streaming(root):
            print(f"{node.parent_agent.config.path}  {node.type}")
            root.save(run_dir)

    asyncio.run(drive())
    print(root.result())
    save_example_graph(
        root,
        "drop-in-llm",
        out_dir=run_dir,
    )
    outer.runtime.close_repls()
    inner.close()


if __name__ == "__main__":
    demo_plain_llm()
    demo_flow_as_llm()
    demo_nested_flow()
