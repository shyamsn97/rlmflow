"""rlmflow-local runner."""

from __future__ import annotations

import asyncio
import time

from rlmflow import Flow, Graph, LocalRuntime
from rlmflow.clients.llm import LLMClient, LLMUsage

from benchmarks.eval import runner
from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("rlmflow-local", aliases=["rlmflow", "rflow-local", "rflow"])
class RLMFlowLocalRunner(Runner):
    def __init__(
        self,
        max_iters: int = 20,
        max_depth: int = 1,
        live_save: bool = True,
        max_steps: int | None = None,
        use_llm_query: bool = False,
    ) -> None:
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.live_save = live_save
        self.max_steps = max_steps
        self.use_llm_query = use_llm_query

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
            use_llm_query=self.use_llm_query,
        )
        start = time.perf_counter()
        # Seed the durable graph; `run_streaming` then drives the whole tree,
        # mutating that graph in place and emitting one event per commit.
        graph = Graph(query=example.prompt, inputs=example.inputs() or {})
        cap = self.max_steps or max(200, self.max_iters * max(1, self.max_depth + 1) * 25)
        steps = 0
        error = None

        async def drive() -> None:
            nonlocal steps
            async for event in flow.run_streaming(graph=graph):
                if event.type != "append_node":
                    continue
                steps += 1
                if self.live_save:
                    graph.save(graph_dir)
                if steps >= cap:
                    raise RuntimeError(f"run exceeded step cap ({cap})")

        try:
            if self.live_save:
                graph.save(graph_dir)
            asyncio.run(drive())
        except Exception as exc:  # benchmark rows should record failures
            error = f"{type(exc).__name__}: {exc}"
        finally:
            flow.close_repls()
            graph.save(graph_dir)

        input_tokens, output_tokens = graph.tokens()
        return Prediction(
            answer=graph.result(),
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


__all__ = ["RLMFlowLocalRunner"]
