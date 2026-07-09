"""rlmflow local runner."""

from __future__ import annotations

import time

from rflow import Flow, LocalRuntime
from rflow.minimal.clients.llm import LLMClient, LLMUsage

from benchmarks.eval import runner
from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("rflow-local", aliases=["rflow"])
class RFlowLocalRunner(Runner):
    def __init__(
        self,
        max_iters: int = 20,
        max_depth: int = 1,
        live_save: bool = True,
        max_steps: int | None = None,
        include_llm_query: bool = False,
    ) -> None:
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.live_save = live_save
        self.max_steps = max_steps
        self.include_llm_query = include_llm_query

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        graph_dir = ctx.artifact_dir / "graph"
        work_dir = ctx.artifact_dir / "workdir"
        work_dir.mkdir(parents=True, exist_ok=True)
        flow = Flow(
            _ModelClient(model),
            max_iters=self.max_iters,
            max_depth=self.max_depth,
            runtime=LocalRuntime(working_directory=work_dir),
            enable_structured_output=False,
            include_llm_query=self.include_llm_query,
        )
        start = time.perf_counter()
        graph = None
        steps = 0
        error = None
        try:
            graph = flow.start(
                example.prompt,
                example.inputs(),
            )
            if self.live_save:
                graph.save(graph_dir)
            cap = self.max_steps or max(200, self.max_iters * max(1, self.max_depth + 1) * 25)
            while not graph.finished:
                graph = flow.step(graph)
                steps += 1
                if self.live_save:
                    graph.save(graph_dir)
                if steps >= cap:
                    raise RuntimeError(f"run exceeded step cap ({cap})")
            graph.save(graph_dir)
            answer = graph.result()
            input_tokens, output_tokens = graph.tokens()
        except Exception as exc:  # benchmark rows should record failures
            answer = graph.result() if graph is not None else ""
            input_tokens, output_tokens = graph.tokens() if graph is not None else (0, 0)
            error = f"{type(exc).__name__}: {exc}"
            if graph is not None:
                graph.save(graph_dir)
        finally:
            flow.close()
        return Prediction(
            answer=answer,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            metrics={
                "time_seconds": time.perf_counter() - start,
                "iterations": steps,
                "graph": graph_metrics(graph),
            },
            artifacts={"graph_path": str(graph_dir)},
            error=error,
        )


class _ModelClient(LLMClient):
    def __init__(self, model: Model) -> None:
        self.benchmark_model = model
        self.model = model.name
        self.last_usage = LLMUsage()

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text = self.benchmark_model.complete(messages, **kwargs)
        usage = self.benchmark_model.usage()
        self.last_usage = LLMUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
        return text

    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        text = self.chat(messages, *args, **kwargs)
        return text, self.last_usage


__all__ = ["RFlowLocalRunner"]
