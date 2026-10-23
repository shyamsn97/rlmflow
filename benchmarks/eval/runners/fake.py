"""Fake runner for harness smoke tests."""

from __future__ import annotations

from benchmarks.eval import runner
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("fake")
class FakeRunner(Runner):
    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        del ctx
        inputs = "\n".join(f"{name}: {value}" for name, value in example.inputs().items())
        answer = model.complete([{"role": "user", "content": f"{example.prompt}\n\n{inputs}"}])
        return Prediction(
            answer=answer,
            usage=model.usage(),
            metrics={"iterations": 1, "llm_turns": 1},
        )


__all__ = ["FakeRunner"]
