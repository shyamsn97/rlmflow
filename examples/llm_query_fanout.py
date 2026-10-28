"""Fan out ordinary ``llm_query(...)`` calls with ``asyncio.gather``.

Deterministic and offline: ``GuidedLLM`` returns a root REPL block that fans
out three one-shot queries, then answers each query itself.

Run:
    python examples/llm_query_fanout.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from rlmflow import AgentConfig, Flow, Graph, LLMUsage

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream, example_run_dir, save_example_graph  # noqa: E402

REVIEWS = [
    "The new search UI is fast and surprisingly easy to use.",
    "The export job failed twice and the error message was useless.",
    "The dashboard loads, but I do not feel strongly about it yet.",
]


class GuidedLLM:
    """Fake model that lets us verify root -> concurrent queries -> done."""

    def __init__(self) -> None:
        self.query_prompts: list[str] = []
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=25, output_tokens=10)
        text = messages[-1]["content"]

        if "Classify this review" in text:
            with self._lock:
                self.query_prompts.append(text)
            return classify_review(text)

        return root_repl_block()


def classify_review(prompt: str) -> str:
    lower = prompt.lower()
    if "fast" in lower or "easy" in lower:
        return "positive"
    if "failed" in lower or "useless" in lower:
        return "negative"
    return "neutral"


def root_repl_block() -> str:
    return (
        "```repl\n"
        f"reviews = {REVIEWS!r}\n"
        "prompts = [\n"
        "    'Classify this review as positive, negative, or neutral: ' + review\n"
        "    for review in reviews\n"
        "]\n"
        "labels = await asyncio.gather(*(llm_query(prompt) for prompt in prompts))\n"
        "print('llm_query returned:', labels)\n"
        "submit('\\n'.join(f'{label}: {review}' for label, review in zip(labels, reviews)))\n"
        "```"
    )


def main() -> None:
    llm = GuidedLLM()
    flow = Flow(llm_clients={"guided": llm}, workers=3)

    g = Graph()
    root = g.create_agent(
        "Classify the reviews. Fan out `llm_query(prompt)` with "
        "`await asyncio.gather(...)`, then call submit(...) with one line per review.",
    )
    config = AgentConfig(model="guided", max_depth=0, max_iters=3)
    out_dir = example_run_dir("llm-query-fanout")

    async def drive() -> None:
        async for step in checkpoint_stream(flow.run_streaming(g, root_config=config), g, out_dir):
            print(f"{g.path(step.agent)}  {step.node.type}")

    asyncio.run(drive())

    print("\nPrompts sent:")
    for prompt in llm.query_prompts:
        print("-", prompt)

    print("\nFinal answer:")
    print(g.result(root))
    save_example_graph(g, "llm-query-fanout")
    flow.runtime.close()


if __name__ == "__main__":
    main()
