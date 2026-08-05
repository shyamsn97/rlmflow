"""Single-call LLM baseline runner."""

from __future__ import annotations

import time

from benchmarks.eval import runner
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("vanilla")
class VanillaRunner(Runner):
    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        del ctx
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the benchmark task directly. Return only the final "
                    "answer value, with no explanation."
                ),
            },
            {"role": "user", "content": _render_example(example)},
        ]
        start = time.perf_counter()
        try:
            answer = model.complete(messages)
            error = None
        except Exception as exc:  # noqa: BLE001 - benchmark rows should record failures
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
        return Prediction(
            answer=answer,
            usage=model.usage(),
            metrics={"time_seconds": time.perf_counter() - start, "iterations": 1},
            error=error,
        )


def _render_example(example: Example) -> str:
    parts = [example.prompt]
    inputs = example.inputs()
    if inputs:
        parts.append("INPUTS:")
        parts.extend(f"INPUT {key}:\n{value}" for key, value in sorted(inputs.items()))
    return "\n\n".join(parts)


__all__ = ["VanillaRunner"]
