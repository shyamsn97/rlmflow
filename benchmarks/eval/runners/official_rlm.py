"""Official RLM comparison runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from benchmarks.eval import runner
from benchmarks.eval.delegation.parity import comparison_contract
from benchmarks.eval.runners.shared import (
    FACT_LOOKUP_DESCRIPTION,
    canonical_delegation_metrics,
    prepare_run,
)
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("official-rlm")
class OfficialRLMRunner(Runner):
    """Run the original `alexzhang13/rlm` package when available.

    This runner intentionally records dependency/setup failures as benchmark
    errors instead of installing a temp venv behind the user's back.
    """

    def __init__(
        self,
        python: str | None = None,
        max_iters: int = 20,
        max_depth: int = 2,
        max_budget: int | None = 100_000,
        worker_model: str | None = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.python = python or sys.executable
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.max_budget = max_budget
        self.worker_model = worker_model
        self.timeout_seconds = timeout_seconds

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        start = time.perf_counter()
        artifact_dir = ctx.artifact_dir.expanduser().resolve()
        log_dir = artifact_dir / "official_rlm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_run(example, ctx, runtime="official-rlm")
        context_file = _write_temp(json.dumps(prepared.inputs, ensure_ascii=False))
        task_file = _write_temp(example.prompt)
        environment = prepared.environment
        try:
            proc = subprocess.run(
                [
                    self.python,
                    "-c",
                    _official_script(
                        context_file=context_file,
                        task_file=task_file,
                        model=model.name,
                        worker_model=self.worker_model,
                        log_dir=log_dir,
                        max_iters=self.max_iters,
                        max_depth=self.max_depth,
                        max_budget=self.max_budget,
                        facts=prepared.facts,
                        environment=environment,
                        environment_artifact_dir=artifact_dir / "environment",
                        episode_id=prepared.episode_id,
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=prepared.work_dir,
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            )
            if proc.returncode != 0:
                return Prediction(
                    answer="",
                    metrics={
                        "time_seconds": time.perf_counter() - start,
                        "iterations": 0,
                    },
                    artifacts={"log_dir": str(log_dir)},
                    error=_short_error(proc.stderr or proc.stdout),
                )
            data = _parse_result(proc.stdout)
            graph = data.get("graph") or {}
            rlm_metrics = data.get("rlm") or {}
            metrics = {
                "time_seconds": float(data.get("time_seconds") or (time.perf_counter() - start)),
                "iterations": int(data.get("iterations") or 1),
                "comparison_contract": comparison_contract(
                    runtime="official-rlm",
                    prompt=example.prompt,
                    inputs=prepared.inputs,
                    root_model=model.name,
                    worker_model=self.worker_model,
                    recursive_levels=max(0, self.max_depth - 1),
                    max_iterations=self.max_iters,
                    max_tokens=self.max_budget,
                    attempt_timeout_seconds=int(ctx.params.get("attempt_timeout", 0)),
                    environment=environment,
                ),
                "graph": graph,
                "delegation": canonical_delegation_metrics(
                    graph=graph,
                    runtime_metrics=rlm_metrics,
                ),
            }
            if data.get("environment"):
                metrics["environment"] = data["environment"]
            return Prediction(
                answer=str(data.get("response") or ""),
                usage={
                    "input_tokens": int(data.get("input_tokens") or 0),
                    "output_tokens": int(data.get("output_tokens") or 0),
                },
                metrics=metrics,
                artifacts={
                    "log_dir": str(log_dir),
                    "log_file_path": str(data.get("log_file_path") or ""),
                    **{
                        str(key): str(value)
                        for key, value in (data.get("environment_artifacts") or {}).items()
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 - benchmark rows should record failures
            return Prediction(
                answer="",
                metrics={"time_seconds": time.perf_counter() - start, "iterations": 0},
                artifacts={"log_dir": str(log_dir)},
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            _unlink(context_file)
            _unlink(task_file)


def _official_script(
    *,
    context_file: str,
    task_file: str,
    model: str,
    worker_model: str | None,
    log_dir: Path,
    max_iters: int,
    max_depth: int,
    max_budget: int | None,
    facts: dict[str, object] | None,
    environment: dict[str, object] | None,
    environment_artifact_dir: Path,
    episode_id: str,
) -> str:
    other_backends = ["openai"] if worker_model is not None else None
    other_backend_kwargs = [{"model_name": worker_model}] if worker_model is not None else None
    return textwrap.dedent(
        f"""
        import json
        import time
        from types import SimpleNamespace

        from rlm import RLM
        from rlm.logger import RLMLogger
        from rlm.utils.exceptions import TokenLimitExceededError
        from benchmarks.eval.environments.official import open_official_environment

        with open({json.dumps(context_file)}, encoding="utf-8") as handle:
            context = json.load(handle)
        with open({json.dumps(task_file)}, encoding="utf-8") as handle:
            task = handle.read()

        raw_facts = {facts!r}
        custom_tools = {{}}
        if raw_facts is not None:
            facts = {{str(key).casefold(): value for key, value in raw_facts.items()}}

            def fact_lookup(entity):
                try:
                    return str(facts[str(entity).strip().casefold()])
                except KeyError as exc:
                    choices = ", ".join(sorted(facts))
                    raise KeyError(
                        f"unknown entity {{entity!r}}; available: {{choices}}"
                    ) from exc

            custom_tools["fact_lookup"] = {{
                "tool": fact_lookup,
                "description": {json.dumps(FACT_LOOKUP_DESCRIPTION)},
            }}

        environment_episode = None
        raw_environment = {environment!r}
        if raw_environment is not None:
            environment_episode = open_official_environment(
                raw_environment,
                artifact_dir={json.dumps(str(environment_artifact_dir))},
                episode_id={json.dumps(episode_id)},
            )
            context["environment_observation"] = json.dumps(
                environment_episode.initial_observation,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            custom_tools.update(environment_episode.tools())
        if not custom_tools:
            custom_tools = None

        logger = RLMLogger(log_dir={json.dumps(str(log_dir))})
        start = time.time()
        def run_completion():
            rlm = RLM(
                backend="openai",
                backend_kwargs={{"model_name": {json.dumps(model)}}},
                other_backends={other_backends!r},
                other_backend_kwargs={other_backend_kwargs!r},
                environment="local",
                max_iterations={max_iters},
                max_depth={max_depth},
                max_tokens={max_budget!r},
                custom_tools=custom_tools,
                custom_sub_tools=custom_tools,
                verbose=False,
                logger=logger,
            )
            return rlm.completion(prompt=context, root_prompt=task)

        token_limit_stop = None
        try:
            if environment_episode is None:
                result = run_completion()
            else:
                with environment_episode.instrument(RLM):
                    with environment_episode.activate(environment_episode.ROOT_OWNER):
                        result = run_completion()
        except TokenLimitExceededError as exc:
            token_limit_stop = {{
                "tokens_used": exc.tokens_used,
                "token_limit": exc.token_limit,
            }}
            result = SimpleNamespace(
                response=exc.partial_answer or "",
                usage_summary=None,
                num_iterations={max_iters},
                metadata=None,
            )
        finally:
            if environment_episode is not None:
                environment_episode.close()
        elapsed = time.time() - start

        usage = getattr(result, "usage_summary", None)
        input_tokens = 0
        output_tokens = 0
        model_summaries = getattr(usage, "model_usage_summaries", None) if usage else None
        if model_summaries:
            for item in model_summaries.values():
                input_tokens += getattr(item, "total_input_tokens", 0)
                output_tokens += getattr(item, "total_output_tokens", 0)
        elif token_limit_stop is not None:
            input_tokens = token_limit_stop["tokens_used"]

        def summarize_trajectory(trajectory, depth=0):
            if not isinstance(trajectory, dict):
                return dict(
                    agents=1,
                    nodes=0,
                    llm_turns=0,
                    max_depth=depth,
                    max_branching=0,
                    iterations=0,
                    code_blocks=0,
                    subcalls=0,
                    recursive_subcalls=0,
                    one_shot_subcalls=0,
                )
            iterations = trajectory.get("iterations") or []
            summary = dict(
                agents=1,
                nodes=len(iterations),
                llm_turns=len(iterations),
                max_depth=depth,
                max_branching=0,
                iterations=len(iterations),
                code_blocks=0,
                subcalls=0,
                recursive_subcalls=0,
                one_shot_subcalls=0,
            )
            for iteration in iterations:
                code_blocks = iteration.get("code_blocks") or []
                summary["code_blocks"] += len(code_blocks)
                summary["nodes"] += len(code_blocks)
                for block in code_blocks:
                    result = block.get("result") or {{}}
                    calls = result.get("rlm_calls") or []
                    recursive_calls = [
                        call for call in calls if isinstance(call, dict) and call.get("metadata")
                    ]
                    summary["subcalls"] += len(calls)
                    summary["recursive_subcalls"] += len(recursive_calls)
                    summary["one_shot_subcalls"] += len(calls) - len(recursive_calls)
                    summary["nodes"] += len(calls)
                    summary["max_branching"] = max(
                        summary["max_branching"],
                        len(recursive_calls),
                    )
                    for call in recursive_calls:
                        child = summarize_trajectory(call.get("metadata"), depth + 1)
                        summary["agents"] += child["agents"]
                        summary["nodes"] += child["nodes"]
                        summary["llm_turns"] += child["llm_turns"]
                        summary["max_depth"] = max(summary["max_depth"], child["max_depth"])
                        summary["max_branching"] = max(
                            summary["max_branching"],
                            child["max_branching"],
                        )
                        summary["iterations"] += child["iterations"]
                        summary["code_blocks"] += child["code_blocks"]
                        summary["subcalls"] += child["subcalls"]
                        summary["recursive_subcalls"] += child["recursive_subcalls"]
                        summary["one_shot_subcalls"] += child["one_shot_subcalls"]
            return summary

        trajectory = getattr(result, "metadata", None) or logger.get_trajectory()
        rlm_metrics = summarize_trajectory(trajectory)
        if token_limit_stop is not None:
            rlm_metrics["token_limit_stop"] = token_limit_stop
        graph_metrics = {{
            "agents": rlm_metrics["agents"],
            "nodes": rlm_metrics["nodes"],
            "llm_turns": rlm_metrics["llm_turns"],
            "max_depth": rlm_metrics["max_depth"],
            "max_branching": rlm_metrics["max_branching"],
        }}
        environment_artifacts = (
            environment_episode.artifacts() if environment_episode is not None else {{}}
        )
        if environment_episode is not None:
            discovery_tree = environment_episode.write_discovery_tree(
                {json.dumps(str(environment_artifact_dir / "discovery_tree.json"))}
            )
            if discovery_tree is not None:
                environment_artifacts["discovery_tree"] = str(discovery_tree)

        print("<<<RESULT>>>")
        print(json.dumps({{
            "response": getattr(result, "response", "") or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_seconds": elapsed,
            "iterations": getattr(result, "num_iterations", None) or 1,
            "log_file_path": getattr(logger, "log_file_path", None),
            "graph": graph_metrics,
            "rlm": rlm_metrics,
            "environment": (
                environment_episode.metrics() if environment_episode is not None else None
            ),
            "environment_artifacts": environment_artifacts,
        }}))
        """
    )


def _parse_result(stdout: str) -> dict[str, object]:
    marker = "<<<RESULT>>>"
    if marker not in stdout:
        raise ValueError(f"official runner output missing {marker}: {_short_error(stdout)}")
    return json.loads(stdout.split(marker, 1)[1].strip())


def _write_temp(content: str) -> str:
    fd, path = tempfile.mkstemp(prefix="rlmflow-bench-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _short_error(text: str, *, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n...<truncated>"


__all__ = ["OfficialRLMRunner"]
