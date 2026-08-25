"""Exercise ``llm_query_batched(...)`` as a core REPL tool.

Deterministic and offline: ``GuidedLLM`` returns a root REPL block that must
call ``llm_query_batched(...)``, then answers each batched prompt itself.

Run:
    python examples/llm_query_batched.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from rlmflow import (
    AgentConfig,
    Flow,
    LLMUsage,
)

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
    """Fake model that lets us verify root -> llm_query_batched -> done."""

    def __init__(self) -> None:
        self.batch_prompts: list[str] = []
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        self.last_usage = LLMUsage(input_tokens=25, output_tokens=10)
        text = messages[-1]["content"]

        if "Classify this review" in text:
            with self._lock:
                self.batch_prompts.append(text)
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
        "labels = await llm_query_batched(prompts)\n"
        "print('llm_query_batched returned:', labels)\n"
        "finish('\\n'.join(f'{label}: {review}' for label, review in zip(labels, reviews)))\n"
        "```"
    )


def main() -> None:
    llm = GuidedLLM()
    flow = Flow(
        llm,
        root_config=AgentConfig(max_depth=0, max_iters=3),
        workers=3,
        use_llm_query=True,
    )

    root = flow.start(
        "Classify the reviews. You must use `await "
        "llm_query_batched(prompts)` for the per-review classifications, "
        "then call finish(...) with one line per review."
    )
    out_dir = example_run_dir("llm-query-batched")

    async def drive() -> None:
        async for node in checkpoint_stream(flow.run_streaming(root), out_dir):
            print(f"{node.parent_agent.config.path}  {node.type}")

    asyncio.run(drive())

    print("\nBatched prompts sent:")
    for prompt in llm.batch_prompts:
        print("-", prompt)

    print("\nFinal answer:")
    print(root.result())
    save_example_graph(root, "llm-query-batched")
    flow.runtime.close_repls()


if __name__ == "__main__":
    main()
