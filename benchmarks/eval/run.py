"""CLI and orchestration for the clean benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import queue
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.eval import DATASETS, LOGGERS, MODELS, RUNNERS
from benchmarks.eval.loggers import MultiLogger
from benchmarks.eval.loggers.jsonl import load_rows
from benchmarks.eval.metrics import summarize
from benchmarks.eval.types import (
    ComponentSpec,
    Example,
    ModelSpec,
    Prediction,
    Row,
    RunContext,
    Score,
    SuiteConfig,
)

# Import built-ins so decorators register.
for module_name in (
    "benchmarks.eval.loggers",
    "benchmarks.eval.models",
    "benchmarks.eval.runners",
    "benchmarks.eval.tasks",
):
    importlib.import_module(module_name)


def run_suite(config: SuiteConfig) -> list[Row]:
    config.root.mkdir(parents=True, exist_ok=True)
    logger = build_logger(config)
    rows = load_rows(config.root / "rows.jsonl") if config.resume else []
    seen = {
        (row.dataset, row.example_id, row.runner, row.model, row.seed)
        for row in rows
    }
    jobs = _build_jobs(config, seen=seen)

    logger.start(config.to_dict())
    try:
        for job in jobs:
            logger.example_start(
                job["example"],
                runner=job["payloads"][0]["runner"]["name"],
                model=config.model.name,
            )
        for row in _run_jobs(config, jobs):
            rows.append(row)
            seen.add((row.dataset, row.example_id, row.runner, row.model, row.seed))
            logger.row(row)
        logger.summary(rows)
        return rows
    finally:
        logger.finish()


def _build_jobs(config: SuiteConfig, *, seen: set[tuple]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    job_root = _job_root(config)
    for dataset_spec in config.datasets:
        dataset = DATASETS.make(dataset_spec.name, **dataset_spec.params)
        for seed in config.seeds:
            examples = dataset.examples(split=config.split, limit=config.limit, seed=seed)
            for example in examples:
                for runner_spec in config.runners:
                    key = (
                        dataset.name,
                        example.id,
                        runner_spec.name,
                        config.model.name,
                        seed,
                    )
                    if key in seen:
                        continue
                    job_index = len(jobs)
                    jobs.append(
                        {
                            "dataset": dataset,
                            "example": example,
                            "payloads": [
                                {
                                    "run_id": config.run_id,
                                    "root": str(job_root),
                                    "dataset_name": dataset.name,
                                    "runner": asdict(runner_spec),
                                    "model": asdict(config.model),
                                    "seed": seed,
                                    "example": asdict(example),
                                    "job_index": job_index,
                                    "attempt": attempt,
                                    "best_of_n": config.best_of_n,
                                    "attempt_timeout": config.attempt_timeout,
                                }
                                for attempt in range(config.best_of_n)
                            ],
                        }
                    )
    return jobs


def _job_root(config: SuiteConfig) -> Path:
    if config.executor == "modal":
        return Path("/tmp/rlmflow-benchmarks") / config.run_id
    return config.root


def _run_jobs(config: SuiteConfig, jobs: list[dict[str, Any]]) -> Iterable[Row]:
    if not jobs:
        return []
    payloads = [payload for job in jobs for payload in job["payloads"]]
    if config.executor == "modal":
        return _stream_best_rows_from_results(jobs, _run_jobs_modal(config, payloads))
    if config.executor != "local":
        raise ValueError("executor must be 'local' or 'modal'")
    if config.parallelism <= 1:
        return _best_rows_from_results(
            jobs,
            [_run_job_payload(payload) for payload in payloads],
        )
    results_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.parallelism) as pool:
        futures = {
            pool.submit(_run_job_payload, payload): index
            for index, payload in enumerate(payloads)
        }
        for future in as_completed(futures):
            results_by_index[futures[future]] = future.result()
    return _best_rows_from_results(
        jobs,
        [results_by_index[index] for index in range(len(payloads))],
    )


def _run_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    attempt_timeout = int(payload.get("attempt_timeout") or 0)
    if attempt_timeout > 0:
        return _run_job_payload_with_timeout(payload, attempt_timeout)
    return _run_job_payload_inner(payload)


def _run_job_payload_with_timeout(payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    output: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_run_job_payload_child, args=(payload, output))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        return _error_payload_result(
            payload,
            error=f"TimeoutError: attempt exceeded {timeout}s",
            time_seconds=float(timeout),
        )
    try:
        return output.get_nowait()
    except queue.Empty:
        return _error_payload_result(
            payload,
            error=f"RuntimeError: attempt process exited with code {process.exitcode} and no result",
        )


def _run_job_payload_child(payload: dict[str, Any], output: mp.Queue) -> None:
    try:
        output.put(_run_job_payload_inner(payload))
    except BaseException as exc:
        output.put(_error_payload_result(payload, error=f"{type(exc).__name__}: {exc}"))


def _run_job_payload_inner(payload: dict[str, Any]) -> dict[str, Any]:
    runner_spec = ComponentSpec(**payload["runner"])
    model_spec = ModelSpec(**payload["model"])
    dataset_name = str(payload["dataset_name"])
    runner_name = runner_spec.name
    example = Example(**payload["example"])
    seed = payload["seed"]
    attempt = int(payload.get("attempt", 0))
    best_of_n = int(payload.get("best_of_n", 1))
    _log_worker_start(
        dataset=dataset_name,
        example=example,
        runner=runner_name,
        model=model_spec.name,
        seed=seed,
        attempt=attempt,
        best_of_n=best_of_n,
    )
    try:
        runner = RUNNERS.make(runner_spec.name, **runner_spec.params)
        model = MODELS.make(model_spec.provider, name=model_spec.name, **model_spec.params)
        artifact_dir = (
            Path(payload["root"])
            / "artifacts"
            / dataset_name
            / example.id
            / runner.name
        )
        if best_of_n > 1:
            artifact_dir = artifact_dir / f"attempt_{attempt:02d}"
        ctx = RunContext(
            run_id=payload["run_id"],
            root=Path(payload["root"]),
            artifact_dir=artifact_dir,
        )
        prediction = runner.run(example, model, ctx)
        return {
            "run_id": payload["run_id"],
            "dataset": dataset_name,
            "example_id": example.id,
            "runner": runner.name,
            "model": model.name,
            "seed": seed,
            "prediction": asdict(prediction),
            "metadata": example.metadata,
            "job_index": payload.get("job_index"),
            "attempt": attempt,
            "best_of_n": best_of_n,
        }
    except Exception as exc:
        return _error_payload_result(payload, error=f"{type(exc).__name__}: {exc}")


def _error_payload_result(
    payload: dict[str, Any],
    *,
    error: str,
    time_seconds: float = 0.0,
) -> dict[str, Any]:
    runner_spec = ComponentSpec(**payload["runner"])
    model_spec = ModelSpec(**payload["model"])
    example = Example(**payload["example"])
    prediction = Prediction(
        answer="",
        metrics={"time_seconds": time_seconds, "iterations": 0},
        error=error,
    )
    return {
        "run_id": payload["run_id"],
        "dataset": str(payload["dataset_name"]),
        "example_id": example.id,
        "runner": runner_spec.name,
        "model": model_spec.name,
        "seed": payload["seed"],
        "prediction": asdict(prediction),
        "metadata": example.metadata,
        "job_index": payload.get("job_index"),
        "attempt": int(payload.get("attempt", 0)),
        "best_of_n": int(payload.get("best_of_n", 1)),
    }


def _log_worker_start(
    *,
    dataset: str,
    example: Example,
    runner: str,
    model: str,
    seed: int | None,
    attempt: int,
    best_of_n: int,
) -> None:
    question = " ".join(example.prompt.split())
    if len(question) > 240:
        question = question[:237].rstrip() + "..."
    attempt_label = f" attempt={attempt + 1}/{best_of_n}" if best_of_n > 1 else ""
    print(
        "[bench-worker] "
        f"dataset={dataset} example={example.id} runner={runner} model={model} "
        f"seed={seed}{attempt_label} question={question!r}",
        flush=True,
    )


def _best_rows_from_results(jobs: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[Row]:
    rows: list[Row] = []
    offset = 0
    for job in jobs:
        count = len(job["payloads"])
        attempt_results = results[offset : offset + count]
        offset += count
        attempts = [_row_from_result(job, result) for result in attempt_results]
        rows.append(_select_best_attempt(attempts))
    return rows


def _stream_best_rows_from_results(
    jobs: list[dict[str, Any]],
    results: Iterable[dict[str, Any]],
) -> Iterable[Row]:
    attempts_by_job: dict[int, list[Row]] = {}
    expected_by_job = {index: len(job["payloads"]) for index, job in enumerate(jobs)}
    for result in results:
        job_index = int(result["job_index"])
        attempts = attempts_by_job.setdefault(job_index, [])
        attempts.append(_row_from_result(jobs[job_index], result))
        if len(attempts) == expected_by_job[job_index]:
            yield _select_best_attempt(attempts)
            del attempts_by_job[job_index]


def _row_from_result(job: dict[str, Any], result: dict[str, Any]) -> Row:
    prediction = Prediction(**result["prediction"])
    dataset = job["dataset"]
    example = job["example"]
    if prediction.error:
        score = Score(value=0.0, correct=False, details={"error": prediction.error})
    else:
        try:
            score = dataset.score(example, prediction)
        except Exception as exc:
            prediction = Prediction(
                answer=prediction.answer,
                usage=prediction.usage,
                metrics=prediction.metrics,
                artifacts=prediction.artifacts,
                error=f"{type(exc).__name__}: {exc}",
            )
            score = Score(value=0.0, correct=False, details={"error": prediction.error})
    metadata = result.get("metadata", {})
    if result.get("best_of_n", 1) > 1:
        metadata = {
            **metadata,
            "attempt": result.get("attempt"),
            "best_of_n": result.get("best_of_n"),
        }
    return Row(
        run_id=result["run_id"],
        dataset=result["dataset"],
        example_id=result["example_id"],
        runner=result["runner"],
        model=result["model"],
        seed=result["seed"],
        prediction=prediction,
        score=score,
        metadata=metadata,
    )


def _select_best_attempt(rows: list[Row]) -> Row:
    if len(rows) == 1:
        return rows[0]
    best = max(rows, key=_attempt_rank)
    metadata = {
        **best.metadata,
        "best_of_n": len(rows),
        "selected_attempt": best.metadata.get("attempt"),
        "attempt_scores": [row.score.value for row in rows],
        "attempt_correct": [row.score.correct for row in rows],
        "attempt_errors": sum(1 for row in rows if row.prediction.error),
    }
    return replace(best, metadata=metadata)


def _attempt_rank(row: Row) -> tuple[float, bool, bool]:
    return (
        row.score.value,
        row.score.correct is True,
        row.prediction.error is None,
    )


def _run_jobs_modal(config: SuiteConfig, payloads: list[dict[str, Any]]):
    try:
        import modal  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Modal execution requires the modal extra: pip install -e '.[modal,eval]'"
        ) from exc

    app = modal.App(config.modal_app_name)
    image = _modal_image(modal)
    secrets = _modal_secrets(modal)

    @app.function(
        image=image,
        cpu=config.modal_cpu,
        timeout=config.modal_timeout,
        max_containers=config.parallelism,
        secrets=secrets,
        serialized=True,
    )
    def run_benchmark_row(payload: dict[str, Any]) -> dict[str, Any]:
        return _run_job_payload(payload)

    with modal.enable_output():
        with app.run():
            for result in run_benchmark_row.map(
                payloads,
                order_outputs=False,
                return_exceptions=False,
            ):
                yield result


def _modal_image(modal):
    repo_root = Path(__file__).resolve().parents[2]
    remote_repo = "/opt/rlmflow"
    return (
        modal.Image.debian_slim()
        .add_local_dir(
            repo_root,
            remote_path=remote_repo,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "wandb",
                "benchmarks/eval/runs",
                "examples/_runs",
                "media",
            ],
        )
        .run_commands(
            "apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*",
            f"python -m pip install -e '{remote_repo}[eval,openai,anthropic]'",
            "python -m pip install 'git+https://github.com/alexzhang13/rlm'",
        )
    )


def _modal_secrets(modal) -> list:
    names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN",
    ]
    available = [name for name in names if os.environ.get(name)]
    return [modal.Secret.from_local_environ(available)] if available else []


def build_logger(config: SuiteConfig) -> MultiLogger:
    instances = []
    for spec in config.loggers:
        params = dict(spec.params)
        if spec.name in {"jsonl", "report", "wandb"}:
            params.setdefault("root", config.root)
        instances.append(LOGGERS.make(spec.name, **params))
    return MultiLogger(instances)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rlmflow benchmarks.")
    parser.add_argument("--dataset", "--datasets", nargs="+", default=["oolong"])
    parser.add_argument("--runner", "--runners", nargs="+", default=["rlmflow-local"])
    parser.add_argument("--model", default="openai:gpt-5-mini")
    parser.add_argument("--logger", "--loggers", nargs="+", default=["jsonl", "console", "report"])
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument(
        "--seed",
        type=int,
        help="Single dataset sampling seed. Use with --limit for one shared sample set.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/eval/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-param", action="append", default=[])
    parser.add_argument("--runner-param", action="append", default=[])
    parser.add_argument("--model-param", action="append", default=[])
    parser.add_argument("--logger-param", action="append", default=[])
    parser.add_argument("--wandb", action="store_true", help="Add the W&B logger.")
    parser.add_argument("--wandb-project", default="rlmflow-eval")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--executor",
        choices=["local", "modal"],
        default="local",
        help="Where to run row jobs. `modal` maps each row to a Modal function.",
    )
    parser.add_argument("--modal", action="store_true", help="Shortcut for --executor modal.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum row jobs to run concurrently. Defaults to sequential execution.",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="Run each logical row N times and keep only the best-scoring attempt.",
    )
    parser.add_argument(
        "--attempt-timeout",
        type=int,
        default=900,
        help="Hard timeout in seconds for one runner attempt before recording an error row.",
    )
    parser.add_argument("--modal-app-name", default="rlmflow-benchmarks")
    parser.add_argument("--modal-cpu", type=float, default=1.0)
    parser.add_argument("--modal-timeout", type=int, default=3600)
    return parser


def config_from_args(args: argparse.Namespace) -> SuiteConfig:
    dataset_names = DATASETS.expand(_flatten(args.dataset))
    runner_names = RUNNERS.expand(_flatten(args.runner))
    logger_names = LOGGERS.expand(_flatten(args.logger))
    if args.wandb and "wandb" not in logger_names:
        logger_names.append("wandb")
    model_spec = parse_model_spec(args.model, parse_params(args.model_param).get("_", {}))
    dataset_params = parse_params(args.dataset_param)
    runner_params = parse_params(args.runner_param)
    logger_params = parse_params(args.logger_param)
    if args.wandb:
        logger_params.setdefault("wandb", {})
        logger_params["wandb"].setdefault("project", args.wandb_project)
        if args.wandb_entity:
            logger_params["wandb"]["entity"] = args.wandb_entity
    executor = "modal" if args.modal else args.executor
    parallelism = max(1, args.parallel)
    best_of_n = max(1, args.best_of_n)
    attempt_timeout = max(0, args.attempt_timeout)
    config = SuiteConfig(
        run_id=args.run_id
        or make_run_id(
            datasets=dataset_names,
            runners=runner_names,
            model=model_spec.label,
        ),
        datasets=[
            ComponentSpec(name=name, params=dataset_params.get(name, {}))
            for name in dataset_names
        ],
        runners=[
            ComponentSpec(name=name, params=runner_params.get(name, {}))
            for name in runner_names
        ],
        model=model_spec,
        loggers=[
            ComponentSpec(name=name, params=logger_params.get(name, {}))
            for name in logger_names
        ],
        seeds=[args.seed] if args.seed is not None else parse_seed_spec(args.seeds),
        split=args.split,
        limit=args.limit,
        output_root=args.out_dir,
        resume=args.resume,
        executor=executor,
        parallelism=parallelism,
        best_of_n=best_of_n,
        attempt_timeout=attempt_timeout,
        modal_app_name=args.modal_app_name,
        modal_cpu=args.modal_cpu,
        modal_timeout=args.modal_timeout,
    )
    return config


def parse_model_spec(value: str, params: dict[str, Any] | None = None) -> ModelSpec:
    params = params or {}
    if ":" not in value:
        provider = value
        name = value
    else:
        provider, name = value.split(":", 1)
    return ModelSpec(provider=provider, name=name, params=params)


def parse_params(values: list[str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"parameter must be key=value: {value}")
        raw_key, raw_value = value.split("=", 1)
        if "." in raw_key:
            scope, key = raw_key.split(".", 1)
        else:
            scope, key = "_", raw_key
        grouped.setdefault(scope, {})[key] = _parse_value(raw_value)
    return grouped


def parse_seed_spec(spec: str) -> list[int]:
    if ":" in spec:
        parts = [int(part) for part in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"invalid seed range: {spec}")
        return list(range(start, stop, step))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def make_run_id(*, datasets: list[str], runners: list[str], model: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = f"{stamp}_{_slug(model)}_{_compact(datasets)}_{_compact(runners)}"
    return text[:180].rstrip("-")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    rows = run_suite(config)
    summary = summarize(rows)
    print(json.dumps(summary.get("overall", {}), indent=2, sort_keys=True))
    print(f"Saved results to {config.root}")
    return 0


def _flatten(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")


def _compact(values: list[str], *, max_length: int = 80) -> str:
    text = "-".join(_slug(value) for value in values)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[: max_length - 9].rstrip('-')}-{digest}"


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_logger",
    "build_parser",
    "config_from_args",
    "main",
    "make_run_id",
    "parse_model_spec",
    "parse_params",
    "parse_seed_spec",
    "run_suite",
]
