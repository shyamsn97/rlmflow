"""Shared rlmflow depth runner."""

from __future__ import annotations

import asyncio
import json
import time

from benchmarks.eval import runner
from benchmarks.eval.artifacts import render_problem_artifacts
from benchmarks.eval.delegation.fanout import rlmflow_trace
from benchmarks.eval.delegation.metrics import (
    delegation_metrics,
)
from benchmarks.eval.delegation.parity import comparison_contract
from benchmarks.eval.environments.base import EnvironmentEpisode, EnvironmentFlow
from benchmarks.eval.environments.registry import (
    action_budget,
    build_environment,
    candidate_budget,
)
from benchmarks.eval.metrics import graph_metrics
from benchmarks.eval.runners.shared import (
    FACT_LOOKUP_DESCRIPTION,
    fact_lookup_data,
    fanout_payloads_from_example,
    fanout_shards_from_payloads,
    prepare_run,
    runtime_prompt,
)
from benchmarks.eval.tasks.swarm_fanout import fanout_payloads
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner
from rlmflow import AgentConfig, Flow, Graph, GraphCheckpointer, LocalRuntime, Step, tool
from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm import LLMClient
from rlmflow.llm.openai import OpenAIClient
from rlmflow.view.html import save_html


class RLMFlowLocalRunner(Runner):
    def __init__(
        self,
        max_iters: int = 20,
        max_depth: int = 2,
        max_budget: int | None = 100_000,
        live_save: bool = True,
        max_steps: int | None = None,
        worker_model: str | None = None,
        divide_budget_by_best_of_n: bool = False,
    ) -> None:
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.max_budget = max_budget
        self.live_save = live_save
        self.max_steps = max_steps
        self.worker_model = worker_model
        self.divide_budget_by_best_of_n = divide_budget_by_best_of_n

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        graph_dir = ctx.artifact_dir / "graph"
        prepared = prepare_run(example, ctx, runtime="rlmflow")
        inputs: dict[str, object] = dict(prepared.inputs)
        canonical_payloads = fanout_payloads_from_example(example)
        shards = _fanout_shards(example) or fanout_shards_from_payloads(canonical_payloads)
        if canonical_payloads is not None:
            inputs["fanout_payloads"] = canonical_payloads
        elif shards is not None:
            inputs["fanout_payloads"] = fanout_payloads(
                shards,
                prepared.work_dir / "shards",
            )
        prompt = _runtime_prompt(example, "rlmflow")
        max_depth = self.max_depth
        best_of_n = int(ctx.params.get("best_of_n", 1))
        max_budget = self.max_budget
        if self.divide_budget_by_best_of_n and max_budget is not None:
            max_budget = max(1, max_budget // max(1, best_of_n))
        llm_clients = {}
        configs = [
            AgentConfig(max_iters=self.max_iters, max_depth=max_depth, max_budget=max_budget)
        ]
        graph = Graph()
        if self.worker_model is not None:
            llm_clients["worker"] = OpenAIClient(self.worker_model)
            configs.append(
                AgentConfig(
                    name="worker",
                    model="worker",
                    description=f"runs on {self.worker_model}",
                    max_iters=self.max_iters,
                    max_depth=max_depth,
                    max_budget=max_budget,
                )
            )
        flow_kwargs = dict(
            runtime=LocalRuntime(working_directory=prepared.work_dir),
            tools=_example_tools(example),
            llm_clients=llm_clients or None,
            configs=configs,
        )
        environment = prepared.environment
        environment_episode = None
        started_at = time.perf_counter()
        if isinstance(environment, dict):
            built = build_environment(
                environment,
                artifact_dir=ctx.artifact_dir / "environment",
            )
            environment_episode = EnvironmentEpisode(
                built,
                max_actions=action_budget(built),
                episode_id=prepared.episode_id,
                candidate_budget=candidate_budget(built),
                expected_observation_sha256=environment.get("observation_sha256"),
            )
            flow = EnvironmentFlow(
                _ModelClient(model), environment=environment_episode, **flow_kwargs
            )
            root = flow.create_root(graph, prompt, **inputs)
        else:
            flow = Flow(_ModelClient(model), **flow_kwargs)
            root = graph.create_agent(prompt, **inputs)
        cap = self.max_steps or max(200, self.max_iters * max(1, max_depth + 1) * 25)
        steps = 0
        error = None
        checkpointer = GraphCheckpointer(graph_dir) if self.live_save else None

        async def drive() -> None:
            nonlocal steps
            try:
                async for item in flow.run_streaming(graph):
                    steps += isinstance(item, Step)
                    if checkpointer is not None:
                        checkpointer.handle(item, graph)
                    if steps >= cap:
                        raise RuntimeError(f"run exceeded step cap ({cap})")
            finally:
                if checkpointer is not None:
                    checkpointer.close()
                await flow.aclose()

        try:
            if self.live_save:
                graph.save(graph_dir)
            asyncio.run(drive())
        except Exception as exc:  # noqa: BLE001 - benchmark rows should record failures
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if checkpointer is None:
                graph.save(graph_dir)

        spent = graph.usage()
        answer = _prediction_answer(graph.result(root))
        environment_metrics = (
            environment_episode.metrics() if environment_episode is not None else None
        )
        delegation = delegation_metrics(graph, root)
        metrics = {
            "time_seconds": time.perf_counter() - started_at,
            "iterations": steps,
            "comparison_contract": comparison_contract(
                runtime="rlmflow",
                task_prompt=example.prompt,
                delivered_prompt=prompt,
                inputs=inputs,
                root_model=model.name,
                worker_model=self.worker_model,
                recursive_levels=max_depth,
                max_iterations=self.max_iters,
                max_tokens=max_budget,
                attempt_timeout_seconds=int(ctx.params.get("attempt_timeout", 0)),
                environment=environment if isinstance(environment, dict) else None,
            ),
            "graph": graph_metrics(graph),
            "delegation": {
                "arm": self.name,
                "oracle_ceiling": bool(self.divide_budget_by_best_of_n and best_of_n > 1),
                "effective_max_budget": max_budget,
                **delegation,
            },
        }
        if shards is not None:
            metrics["fanout"] = rlmflow_trace(
                shards,
                delegation["child_runs"],
            )
        if environment_metrics is not None:
            metrics["environment"] = environment_metrics
        artifacts = {"graph_path": str(graph_dir)}
        agent_trace = save_html(
            graph,
            ctx.artifact_dir / "agent_trace.html",
            title=f"{example.id} · {self.name} · seed {ctx.params.get('seed', 0)}",
        )
        artifacts["agent_trace"] = str(agent_trace)
        if environment_episode is not None:
            artifacts.update(environment_episode.artifacts())
            if environment_episode.candidates is not None:
                discovery_tree = environment_episode.candidates.write(
                    ctx.artifact_dir / "discovery_tree.json"
                )
                artifacts["discovery_tree"] = str(discovery_tree)
        artifacts.update(
            render_problem_artifacts(
                example,
                answer,
                graph,
                root,
                ctx.artifact_dir,
                environment=environment_metrics,
            )
        )
        return Prediction(
            answer=answer,
            usage={
                "input_tokens": spent.input_tokens,
                "output_tokens": spent.output_tokens,
                "reasoning_tokens": spent.reasoning_tokens,
            },
            metrics=metrics,
            artifacts=artifacts,
            error=error,
        )


class RLMFlowDepthRunner(RLMFlowLocalRunner):
    """One benchmark harness whose only treatment variable is ``max_depth``."""

    def __init__(self, *, arm_name: str, max_depth: int, **kwargs) -> None:
        super().__init__(max_depth=max_depth, **kwargs)
        self.name = arm_name


def _depth_runner(
    arm_name: str,
    depth: int,
    kwargs: dict,
) -> RLMFlowDepthRunner:
    requested_depth = int(kwargs.pop("max_depth", depth))
    if requested_depth != depth:
        raise ValueError(f"{arm_name} fixes max_depth={depth}; received {requested_depth}")
    return RLMFlowDepthRunner(
        arm_name=arm_name,
        max_depth=depth,
        **kwargs,
    )


@runner(
    "rlmflow-depth-0",
    tags=["delegation-arm", "depth-arm"],
)
def rlmflow_depth_0(**kwargs) -> RLMFlowDepthRunner:
    return _depth_runner("rlmflow-depth-0", 0, kwargs)


@runner(
    "rlmflow-depth-1",
    tags=["delegation-arm", "depth-arm"],
)
def rlmflow_depth_1(**kwargs) -> RLMFlowDepthRunner:
    return _depth_runner("rlmflow-depth-1", 1, kwargs)


@runner(
    "rlmflow-depth-2",
    tags=["delegation-arm", "depth-arm"],
)
def rlmflow_depth_2(**kwargs) -> RLMFlowDepthRunner:
    return _depth_runner("rlmflow-depth-2", 2, kwargs)


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
            reasoning_tokens=usage.get("reasoning_tokens", 0),
        )
        return text

    def completion(self, messages: list[dict[str, str]], *args, **kwargs) -> tuple[str, LLMUsage]:
        text = self.chat(messages, *args, **kwargs)
        return text, self.last_usage


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


def _fanout_shards(example: Example) -> list[dict] | None:
    shards = example.metadata.get("shards")
    return list(shards) if example.metadata.get("fixture") and isinstance(shards, list) else None


def _runtime_prompt(example: Example, runtime: str) -> str:
    shards = _fanout_shards(example)
    if shards is None and example.metadata.get("environment_family") != "fanout":
        return example.prompt
    from benchmarks.eval.tasks.swarm_fanout import fanout_runtime_guidance

    return runtime_prompt(example.prompt, fanout_runtime_guidance(runtime))


__all__ = [
    "RLMFlowDepthRunner",
    "RLMFlowLocalRunner",
    "rlmflow_depth_0",
    "rlmflow_depth_1",
    "rlmflow_depth_2",
]
