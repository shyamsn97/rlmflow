"""Pinned JAZ runner for runner-neutral fan-out comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks.eval import runner
from benchmarks.eval.delegation.fanout import jaz_trace
from benchmarks.eval.delegation.parity import comparison_contract
from benchmarks.eval.runners.shared import (
    fanout_payloads_from_example,
    fanout_shards_from_payloads,
    prepare_run,
    runtime_prompt,
)
from benchmarks.eval.tasks.swarm_fanout import fanout_payloads
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("jaz", tags=["external-reference", "delegation-arm"])
class JazRunner(Runner):
    def __init__(
        self,
        python: str = "/opt/jaz/bin/python",
        max_iters: int = 20,
        max_depth: int = 2,
        max_budget: int | None = 100_000,
        timeout_seconds: int = 840,
    ) -> None:
        self.python = python
        self.max_iters = max_iters
        self.max_depth = max_depth
        self.max_budget = max_budget
        self.timeout_seconds = timeout_seconds

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        payloads = fanout_payloads_from_example(example)
        shards = _fanout_shards(example)
        shards = shards or fanout_shards_from_payloads(payloads)
        if payloads is None and shards is None:
            return Prediction(
                answer="",
                error="JazRunner currently supports only swarm-fanout fixtures",
            )
        prepared = prepare_run(example, ctx, runtime="jaz")
        prompt = _runtime_prompt(example)
        if payloads is None:
            payloads = fanout_payloads(shards, prepared.work_dir / "shards")
        trace_dir = ctx.artifact_dir / "jaz-trajectory"
        invoke_trace = ctx.artifact_dir / "jaz-invokes.json"
        payload_path = ctx.artifact_dir / "jaz-payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "model": _model_selector(model),
                    "task": prompt,
                    "fanout_payloads": payloads,
                    "trace_dir": str(trace_dir),
                    "max_iters": self.max_iters,
                    "max_depth": self.max_depth,
                    "max_budget": self.max_budget,
                }
            )
        )
        data: dict[str, Any]
        try:
            proc = subprocess.run(
                [
                    self.python,
                    "-c",
                    (
                        "import json,sys;"
                        "from benchmarks.eval.runners.jaz import _run_isolated;"
                        "payload=json.load(open(sys.argv[1]));"
                        "print('JAZ_RESULT='+json.dumps(_run_isolated(payload)))"
                    ),
                    str(payload_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=prepared.work_dir,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            [
                                str(Path(__file__).resolve().parents[3]),
                                os.environ.get("PYTHONPATH", ""),
                            ],
                        )
                    ),
                },
            )
            data = _parse_result(proc.stdout)
            if proc.returncode != 0:
                data["error"] = _short_error(proc.stderr or proc.stdout)
        except Exception as exc:  # noqa: BLE001 - benchmark rows record external failures
            data = {
                "answer": "",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed": 0.0,
                "records": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_calls": 0,
                "repl_execs": 0,
            }

        records = list(data["records"])
        invoke_trace.write_text(json.dumps(records, indent=2, sort_keys=True))
        fanout = jaz_trace(shards, records)
        children = list(fanout["children"])
        retrieved = sum(bool(child.get("receipt")) for child in children)
        graph = {
            "agents": len(records),
            "nodes": len(records) + data["llm_calls"] + data["repl_execs"],
            "llm_turns": data["llm_calls"],
            "max_depth": max((int(record["depth"]) for record in records), default=0),
            "max_branching": len(children),
        }
        return Prediction(
            answer=str(data["answer"]),
            usage={
                "input_tokens": int(data["input_tokens"]),
                "output_tokens": int(data["output_tokens"]),
            },
            metrics={
                "time_seconds": float(data["elapsed"]),
                "iterations": int(data["llm_calls"]),
                "comparison_contract": comparison_contract(
                    runtime="jaz",
                    task_prompt=example.prompt,
                    delivered_prompt=prompt,
                    inputs={"fanout_payloads": payloads},
                    root_model=model.name,
                    worker_model=model.name,
                    recursive_levels=max(0, self.max_depth - 1),
                    max_iterations=self.max_iters,
                    max_tokens=self.max_budget,
                    attempt_timeout_seconds=int(ctx.params.get("attempt_timeout", 0)),
                    environment=None,
                ),
                "graph": graph,
                "delegation": {
                    "arm": "jaz",
                    "children": len(children),
                    "launch_events": len(children),
                    "retrieved_child_results": retrieved,
                    "retrieval_ratio": retrieved / len(children) if children else 0.0,
                    "max_depth": graph["max_depth"],
                    "query_calls": 0,
                    "runtime": "jaz",
                },
                "fanout": fanout,
            },
            artifacts={
                "trajectory": str(trace_dir),
                "invoke_trace": str(invoke_trace),
            },
            error=data.get("error"),
        )


def _run_isolated(payload: dict[str, Any]) -> dict[str, Any]:
    """Run JAZ inside its dependency-isolated interpreter."""
    import jaz
    from jaz.hooks import (
        IterationLimit,
        RecursionLimit,
        ReturnType,
        TrajectoryDirectoryRecorder,
    )
    from jaz.llm import LiteLLM

    evidence_hook = _evidence_hook(
        payload.get("max_budget"),
        dict(payload["fanout_payloads"]),
    )
    started = time.perf_counter()
    answer = ""
    error = None
    try:
        jaz.configure(llm=LiteLLM(model=str(payload["model"])))
        with (
            evidence_hook,
            IterationLimit(max_iterations=int(payload["max_iters"])),
            RecursionLimit(max_depth=int(payload["max_depth"])),
            TrajectoryDirectoryRecorder(str(payload["trace_dir"])),
        ):
            answer = str(
                jaz.invoke(
                    ReturnType(str),
                    task=str(payload["task"]),
                    fanout_payloads=dict(payload["fanout_payloads"]),
                )
            )
    except Exception as exc:  # noqa: BLE001 - benchmark rows record external failures
        error = f"{type(exc).__name__}: {exc}"
    return {
        "answer": answer,
        "error": error,
        "elapsed": time.perf_counter() - started,
        "records": list(evidence_hook.records.values()),
        "input_tokens": evidence_hook.input_tokens,
        "output_tokens": evidence_hook.output_tokens,
        "llm_calls": evidence_hook.llm_calls,
        "repl_execs": evidence_hook.repl_execs,
    }


def _evidence_hook(
    max_budget: int | None,
    fanout_payloads: dict[str, dict[str, str]],
):
    from jaz.hooks import Hook
    from jaz.hooks.effects import Abort
    from jaz.hooks.events import Completed
    from jaz.repl import Raise, Return

    class EvidenceHook(Hook):
        def __init__(self) -> None:
            self.payload_by_task = {
                str(payload["task"]): payload for payload in fanout_payloads.values()
            }
            self.records: dict[str, dict[str, Any]] = {}
            self.input_tokens = 0
            self.output_tokens = 0
            self.llm_calls = 0
            self.repl_execs = 0

        def on_invoke_enter(self, event):
            inputs = dict(event.inputs)
            task = str(inputs.get("task") or "")
            child_payload = self.payload_by_task.get(task)
            self.records[event.invoke_id] = {
                "invoke_id": event.invoke_id,
                "parent_invoke_id": event.parent_invoke_id,
                "parent_repl_iteration": event.parent_repl_iteration,
                "depth": event.depth,
                "worker": (child_payload.get("worker") if child_payload is not None else None),
                "filename": (child_payload.get("filename") if child_payload is not None else None),
                "prompt_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "terminal": False,
                "failed": False,
                "persisted": True,
                "receipt": False,
            }
            return []

        def on_invoke_exit(self, event):
            record = self.records[event.invoke_id]
            record["terminal"] = True
            outcome = event.outcome
            completed_return = isinstance(outcome, Completed) and isinstance(outcome.result, Return)
            record["failed"] = not completed_return or (
                isinstance(outcome, Completed) and isinstance(outcome.result, Raise)
            )
            record["receipt"] = event.depth == 2 and completed_return
            return []

        def on_llm_query_enter(self, event):
            used = self.input_tokens + self.output_tokens
            if max_budget is not None and used >= max_budget:
                return [Abort(error=RuntimeError(f"JAZ token budget exhausted: {used}"))]
            return []

        def on_llm_query_exit(self, event):
            if isinstance(event.outcome, Completed):
                response = event.outcome.result
                self.input_tokens += int(response.prompt_tokens or 0)
                self.output_tokens += int(response.completion_tokens or 0)
                self.llm_calls += 1
            return []

        def on_repl_exec_exit(self, event):
            self.repl_execs += 1
            return []

    return EvidenceHook()


def _fanout_shards(example: Example) -> list[dict[str, Any]] | None:
    shards = example.metadata.get("shards")
    return list(shards) if example.metadata.get("fixture") and isinstance(shards, list) else None


def _runtime_prompt(example: Example) -> str:
    from benchmarks.eval.tasks.swarm_fanout import fanout_runtime_guidance

    return runtime_prompt(example.prompt, fanout_runtime_guidance("jaz"))


def _model_selector(model: Model) -> str:
    provider = str(getattr(model, "provider", "") or "openai")
    return f"{provider}/{model.name}"


def _parse_result(stdout: str) -> dict[str, Any]:
    prefix = "JAZ_RESULT="
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            value = json.loads(line.removeprefix(prefix))
            if isinstance(value, dict):
                return value
    raise ValueError(f"JAZ subprocess emitted no result: {stdout[-2000:]}")


def _short_error(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-20:])[-4000:] or "JAZ subprocess failed"


__all__ = ["JazRunner"]
