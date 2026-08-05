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
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("official-rlm")
class OfficialRLMRunner(Runner):
    """Run the original `alexzhang13/rlm` package when available.

    This runner intentionally records dependency/setup failures as benchmark
    errors instead of installing a temp venv behind the user's back.
    """

    def __init__(self, python: str | None = None, max_iters: int = 20) -> None:
        self.python = python or sys.executable
        self.max_iters = max_iters

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        start = time.perf_counter()
        log_dir = ctx.artifact_dir / "official_rlm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        context_file = _write_temp(_render_context(example))
        task_file = _write_temp(example.prompt)
        try:
            proc = subprocess.run(
                [
                    self.python,
                    "-c",
                    _official_script(
                        context_file=context_file,
                        task_file=task_file,
                        model=model.name,
                        log_dir=log_dir,
                        max_iters=self.max_iters,
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=max(300, self.max_iters * 60),
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
            return Prediction(
                answer=str(data.get("response") or ""),
                usage={
                    "input_tokens": int(data.get("input_tokens") or 0),
                    "output_tokens": int(data.get("output_tokens") or 0),
                },
                metrics={
                    "time_seconds": float(
                        data.get("time_seconds") or (time.perf_counter() - start)
                    ),
                    "iterations": int(data.get("iterations") or 1),
                    "graph": data.get("graph") or {},
                    "rlm": data.get("rlm") or {},
                },
                artifacts={
                    "log_dir": str(log_dir),
                    "log_file_path": str(data.get("log_file_path") or ""),
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


def _render_context(example: Example) -> str:
    inputs = example.inputs()
    if set(inputs) == {"context"}:
        return inputs["context"]
    return "\n\n".join(f"INPUT {key}:\n{value}" for key, value in sorted(inputs.items()))


def _official_script(
    *,
    context_file: str,
    task_file: str,
    model: str,
    log_dir: Path,
    max_iters: int,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        import time

        from rlm import RLM
        from rlm.logger import RLMLogger

        with open({json.dumps(context_file)}, encoding="utf-8") as handle:
            context = handle.read()
        with open({json.dumps(task_file)}, encoding="utf-8") as handle:
            task = handle.read()

        logger = RLMLogger(log_dir={json.dumps(str(log_dir))})
        start = time.time()
        rlm = RLM(
            backend="openai",
            backend_kwargs={{"model_name": {json.dumps(model)}}},
            environment="local",
            max_iterations={max_iters},
            verbose=False,
            logger=logger,
        )
        result = rlm.completion(prompt=context, root_prompt=task)
        elapsed = time.time() - start

        usage = getattr(result, "usage_summary", None)
        input_tokens = 0
        output_tokens = 0
        model_summaries = getattr(usage, "model_usage_summaries", None) if usage else None
        if model_summaries:
            for item in model_summaries.values():
                input_tokens += getattr(item, "total_input_tokens", 0)
                output_tokens += getattr(item, "total_output_tokens", 0)

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
        graph_metrics = {{
            "agents": rlm_metrics["agents"],
            "nodes": rlm_metrics["nodes"],
            "llm_turns": rlm_metrics["llm_turns"],
            "max_depth": rlm_metrics["max_depth"],
            "max_branching": rlm_metrics["max_branching"],
        }}

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
