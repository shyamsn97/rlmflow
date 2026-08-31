"""rlmflow-local runner."""

from __future__ import annotations

import asyncio
import json
import time

from benchmarks.eval import runner
from benchmarks.eval.delegation.conditions import (
    DelegationCondition,
    apply_condition,
    system_prompt_for,
)
from benchmarks.eval.delegation.metrics import delegation_metrics
from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.runners.shared import (
    FACT_LOOKUP_DESCRIPTION,
    example_inputs,
    fact_lookup_data,
    materialize_fixtures,
)
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner
from rlmflow import (
    AgentConfig,
    Flow,
    GraphCheckpointer,
    LocalRuntime,
    persistence,
    start,
    tool,
)
from rlmflow.llm import LLMClient, LLMUsage


@runner("rlmflow-local", aliases=["rlmflow"])
class RLMFlowLocalRunner(Runner):
    def __init__(
        self,
        max_iters: int = 20,
        child_max_iters: int | None = 20,
        max_depth: int = 2,
        max_budget: int | None = 100_000,
        live_save: bool = True,
        max_steps: int | None = None,
        use_llm_query: bool = True,
        prompt_condition: str = "current_policy",
        use_agent_tree: bool = False,
    ) -> None:
        self.max_iters = max_iters
        self.child_max_iters = child_max_iters
        self.max_depth = max_depth
        self.max_budget = max_budget
        self.live_save = live_save
        self.max_steps = max_steps
        self.use_llm_query = use_llm_query
        self.prompt_condition = DelegationCondition.parse(prompt_condition)
        self.use_agent_tree = use_agent_tree

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        graph_dir = ctx.artifact_dir / "graph"
        work_dir = ctx.artifact_dir / "workdir"
        work_dir.mkdir(parents=True, exist_ok=True)
        materialize_fixtures(example, work_dir)
        inputs = _example_inputs(example)
        max_depth = 0 if self.prompt_condition is DelegationCondition.LOCAL else self.max_depth
        flow = Flow(
            _ModelClient(model),
            root_config=AgentConfig(
                max_iters=self.max_iters,
                child_max_iters=self.child_max_iters,
                max_depth=max_depth,
                max_budget=self.max_budget,
            ),
            runtime=LocalRuntime(working_directory=work_dir),
            system_prompt=system_prompt_for(self.prompt_condition),
            tools=_example_tools(example),
            enable_structured_output=False,
            use_llm_query=self.use_llm_query,
            use_agent_tree=self.use_agent_tree,
        )
        apply_condition(flow, self.prompt_condition)
        started_at = time.perf_counter()
        # Seed the durable graph; `run_streaming` then drives the whole tree,
        # mutating that graph in place and publishing each changed Node.
        graph = start(
            query=example.prompt,
            inputs=inputs,
            max_iters=self.max_iters,
            child_max_iters=self.child_max_iters,
            max_depth=max_depth,
            max_budget=self.max_budget,
        )
        cap = self.max_steps or max(200, self.max_iters * max(1, max_depth + 1) * 25)
        steps = 0
        error = None
        checkpointer = GraphCheckpointer(graph_dir) if self.live_save else None

        async def drive() -> None:
            nonlocal steps
            try:
                async for node in flow.run_streaming(graph):
                    steps += 1
                    if checkpointer is not None:
                        checkpointer.handle(node)
                    if steps >= cap:
                        raise RuntimeError(f"run exceeded step cap ({cap})")
            finally:
                if checkpointer is not None:
                    checkpointer.close()
                await flow.aclose()

        try:
            if self.live_save:
                persistence.save(graph, graph_dir)
            asyncio.run(drive())
        except Exception as exc:  # noqa: BLE001 - benchmark rows should record failures
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if checkpointer is None:
                persistence.save(graph, graph_dir)

        spent = graph.usage
        answer = _prediction_answer(graph.result())
        return Prediction(
            answer=answer,
            usage={
                "input_tokens": spent.input_tokens,
                "output_tokens": spent.output_tokens,
            },
            metrics={
                "time_seconds": time.perf_counter() - started_at,
                "iterations": steps,
                "graph": graph_metrics(graph),
                "delegation": {
                    "condition": self.prompt_condition.value,
                    **delegation_metrics(graph),
                },
            },
            artifacts={"graph_path": str(graph_dir)},
            error=error,
        )


def _prediction_answer(result: object) -> str:
    """Serialize typed graph results only at the benchmark text boundary."""
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


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

    def completion(self, messages: list[dict[str, str]], *args, **kwargs) -> tuple[str, LLMUsage]:
        text = self.chat(messages, *args, **kwargs)
        return text, self.last_usage


def _example_inputs(example: Example) -> dict[str, str]:
    """Compatibility wrapper around the runner-neutral public inputs."""
    return example_inputs(example)


def _example_tools(example: Example) -> list:
    facts = fact_lookup_data(example)
    if facts is None:
        return []

    @tool(FACT_LOOKUP_DESCRIPTION)
    def fact_lookup(entity: str) -> str:
        try:
            return str(facts[entity.strip().casefold()])
        except KeyError as exc:
            choices = ", ".join(sorted(facts))
            raise KeyError(f"unknown entity {entity!r}; available: {choices}") from exc

    return [fact_lookup]


__all__ = ["RLMFlowLocalRunner"]
