"""Flow as a drop-in LLM.

Because `Flow` implements the `LLMClient` protocol (`chat()` / `completion()`),
you can swap it in anywhere you'd use a raw LLM. Calling `flow.chat(messages)`
runs the full recursive agent loop under the hood and returns a plain string —
same signature as any other LLM client.

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

from pathlib import Path

from rflow.clients import OpenAIClient
from rflow.minimal import Flow, FlowLLM, Graph


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
            max_iters=5,
        )
    )
    answer = ask(agent, "Compute 17 * 23 using a ```repl``` block, then call done().")
    print(answer, "\n")
    if agent.last_graph is not None:
        _save_example_graph(
            agent.last_graph,
            __file__,
            "drop-in-llm",
            out_dir=_example_run_dir(__file__, "drop-in-llm") / "flow-as-llm",
        )
    agent.close()


def demo_nested_flow():
    print("=== nested Flow (outer agent uses inner agent as its LLM) ===")
    inner = FlowLLM(
        Flow(
            OpenAIClient(model="gpt-4o-mini"),
            max_iters=3,
        )
    )
    outer = Flow(
        inner,
        max_iters=3,
    )
    graph = Graph(query="What's the 7th Fibonacci number? Use ```repl``` to compute.")
    answer = outer.run(graph)
    print(answer)
    _save_example_graph(
        graph,
        __file__,
        "drop-in-llm",
        out_dir=_example_run_dir(__file__, "drop-in-llm") / "nested-flow",
    )
    outer.close_repls(graph.graph_id)
    inner.close()


if __name__ == "__main__":
    demo_plain_llm()
    demo_flow_as_llm()
    demo_nested_flow()
