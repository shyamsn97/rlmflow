"""rflow runner for autoresearch on Modal."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rflow.clients import AnthropicClient, OpenAIClient
from rflow import (
    DEFAULT_BUILDER,
    DockerRuntime,
    FILE_TOOLS,
    Flow,
    Graph,
    LiveTreeRenderer,
    LocalRuntime,
    SubprocessRuntime,
    render_tree,
    tool,
)

try:  # Allow both `python examples/autoresearch/run.py` and imports.
    from .modal_runner import ModalConfig, preflight, submit, validate_gpu
except ImportError:  # pragma: no cover
    from modal_runner import ModalConfig, preflight, submit, validate_gpu


UPSTREAM_FILES = ("README.md", "prepare.py", "train.py", "program.md", "pyproject.toml", "uv.lock")
RUNNING_STATUSES = {"created", "submitted"}
SUBMITTED_STATUSES = {"submitted", "succeeded", "crashed", "oom", "timeout", "infra_error"}
FAIL_STATUSES = {"crashed", "oom", "timeout", "infra_error", "preflight_failed"}
BASELINE_RESULT_PATH = Path(__file__).with_name("baseline_result.json")

ADAPTER_PROMPT = """
Autoresearch loop policy:
- `INPUTS["task_instructions"]` is task context only. Inspect it first.
- Do not use git or run training manually. Use `submit_trial(slug, hypothesis)`.
- Call `run_baseline()` once; it uses the cached baseline result and does not
  submit a Modal job.
- Use root -> planner -> implementation children. Children may block while
  `submit_trial(...)` runs the Modal job; multiple children run in parallel.
- Planner focus areas: optimizer/schedule, size/depth/width, normalization,
  regularization, attention, or exploit current best.
- Implementation children edit only `INPUTS["trial_dir"]/train.py` and call
  `submit_trial(INPUTS["slug"], INPUTS["hypothesis"])`. If it returns
  `preflight_failed`, fix the syntax/import error and submit again. If it returns
  any other failure row, return it to the parent.
- All `launch_subagents(..., inputs=...)` values must be strings. Use `str(...)`
  for counts and JSON strings for structured values.

Root sketch:
```repl
baseline = run_baseline()
best, status = best_run(), submission_status()
if status["remaining_submissions"] == 0:
    done(str({"status": "complete", "best": best, "runs": list_runs()}))
parent = best["slug"] if best else "baseline"
results = await launch_subagents([{
    "name": "optimizer_schedule_hypotheses",
    "query": "Plan optimizer/schedule trials; create_trial then launch implementers.",
    "inputs": {
        "task_instructions": INPUTS["task_instructions"],
        "focus": "optimizer_schedule",
        "parent_slug": parent,
        "remaining_submissions": str(status["remaining_submissions"]),
    },
}])
print(results, list_runs(), best_run(), submission_status())
```

Planner sketch:
```repl
ideas = [("rmsnorm_on_best", "Replace LayerNorm with RMSNorm only.")]
trials = [create_trial(slug, hyp, parent_slug=INPUTS["parent_slug"]) for slug, hyp in ideas]
children = [{
    "name": row["slug"],
    "query": "Edit only INPUTS['trial_dir']/train.py, then submit_trial(...).",
    "inputs": {"trial_dir": row["agent_trial_dir"], "slug": row["slug"], "hypothesis": row["hypothesis"]},
} for row in trials]
child_results = await launch_subagents(children)
done(str(child_results))
```
"""


def build_prompt_builder():
    return DEFAULT_BUILDER.section(
        "autoresearch_adapter",
        ADAPTER_PROMPT,
        title="Autoresearch Adapter",
        before="tools",
    )


class ExperimentCrashed(RuntimeError):
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        tail = (row.get("stderr_tail") or row.get("stdout_tail") or "").strip()
        super().__init__(f"{row.get('status')} slug={row.get('slug')!r}\n{tail}")


class SubmissionError(RuntimeError):
    pass


class AutoresearchState:
    def __init__(
        self,
        *,
        example_dir: Path,
        out_dir: Path,
        modal_config: ModalConfig,
        max_submissions: int | None,
        created_timeout_s: int,
        submitted_timeout_s: int,
    ) -> None:
        self.example_dir = example_dir.resolve()
        self.out_dir = out_dir.resolve()
        self.base_dir = self.out_dir / "upstream_base"
        self.trials_dir = self.out_dir / "trials"
        self.ledger_path = self.out_dir / "ledger.jsonl"
        self.modal_config = modal_config
        self.max_submissions = max_submissions
        self.created_timeout_s = created_timeout_s
        self.submitted_timeout_s = submitted_timeout_s
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.lock = threading.RLock()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)

    def prepare_base(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for name in UPSTREAM_FILES:
            src = self.example_dir / name
            if src.exists():
                dst = self.base_dir / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text())

    def run_baseline(self) -> dict[str, Any]:
        row = self.latest("baseline")
        if row and row.get("status") == "succeeded":
            return dict(row)
        return self.seed_cached_baseline()

    def seed_cached_baseline(self) -> dict[str, Any]:
        """Seed the ledger from the checked-in baseline result without Modal."""
        row = self.latest("baseline")
        if row is None:
            row = self.create_trial(
                "baseline",
                "Cached upstream train.py baseline.",
                parent_slug=None,
            )
        cached = json.loads(BASELINE_RESULT_PATH.read_text())
        trial_dir = Path(str(row["trial_dir"]))
        stdout = str(cached.get("stdout_tail") or "")
        stderr = str(cached.get("stderr_tail") or "")
        final = {
            **row,
            **cached,
            "n": int(row["n"]),
            "slug": "baseline",
            "parent_slug": None,
            "hypothesis": str(row.get("hypothesis") or "Cached upstream train.py baseline."),
            "status": "succeeded",
            "trial_dir": str(trial_dir),
            "agent_trial_dir": _relative_to(trial_dir, self.out_dir),
            "source_path": str(trial_dir / "train.py"),
            "agent_source_path": _relative_to(trial_dir / "train.py", self.out_dir),
            "log_path": str(trial_dir / "run.log"),
            "stdout_path": str(trial_dir / "stdout.txt"),
            "stderr_path": str(trial_dir / "stderr.txt"),
            "result_path": str(trial_dir / "result.json"),
            "ts": time.time(),
        }
        (trial_dir / "stdout.txt").write_text(stdout)
        (trial_dir / "stderr.txt").write_text(stderr)
        (trial_dir / "run.log").write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""))
        (trial_dir / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True))
        self.append(final)
        return final

    def create_trial(
        self,
        slug: str,
        hypothesis: str,
        parent_slug: str | None = None,
    ) -> dict[str, Any]:
        self.reap_timeouts()
        slug = _slugify(slug)
        with self.lock:
            existing = self.latest(slug)
            if existing:
                return dict(existing)
            if slug != "baseline":
                self._check_budget(count_created=True)

            parent_dir, resolved_parent = self._parent_dir(parent_slug)
            n = self.next_n()
            trial_dir = self.trials_dir / f"{n:03d}_{slug}"
            shutil.copytree(
                parent_dir,
                trial_dir,
                ignore=shutil.ignore_patterns(
                    "run.log",
                    "stdout.txt",
                    "stderr.txt",
                    "result.json",
                    "metadata.json",
                    ".venv",
                    "__pycache__",
                    ".pytest_cache",
                ),
            )
            row = {
                "n": n,
                "slug": slug,
                "hypothesis": hypothesis,
                "parent_slug": resolved_parent,
                "status": "created",
                "created_at": time.time(),
                "trial_dir": str(trial_dir),
                "agent_trial_dir": _relative_to(trial_dir, self.out_dir),
                "source_path": str(trial_dir / "train.py"),
                "agent_source_path": _relative_to(trial_dir / "train.py", self.out_dir),
                "ts": time.time(),
            }
            (trial_dir / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True))
            self.append(row)
            return dict(row)

    def submit_trial(self, slug: str, hypothesis: str = "") -> dict[str, Any]:
        self.reap_timeouts()
        slug = _slugify(slug)
        row = self.latest(slug)
        if row is None:
            raise SubmissionError(f"unknown trial slug: {slug}")
        if row.get("status") in {"succeeded", "submitted"}:
            return dict(row)
        if slug != "baseline" and row.get("status") not in SUBMITTED_STATUSES:
            self._check_budget(count_created=False)

        trial_dir = Path(str(row["trial_dir"]))
        preflight = _preflight(trial_dir)
        if preflight:
            final = self.finalize(row, preflight, hypothesis)
            if slug == "baseline":
                raise ExperimentCrashed(final)
            return final

        self.append({**row, "status": "submitted", "ts": time.time()})
        try:
            result = submit(
                self.modal_config,
                path=trial_dir,
                slug=slug,
                n=int(row["n"]),
                run_id=self.run_id,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "infra_error",
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc!r}",
                "returncode": None,
            }

        final = self.finalize(row, result, hypothesis)
        if slug == "baseline" and final["status"] in FAIL_STATUSES:
            raise ExperimentCrashed(final)
        return final

    def list_runs(self) -> list[dict[str, Any]]:
        self.reap_timeouts()
        rows = list(self.latest_by_n().values())
        rows.sort(key=_rank_key)
        return [_summary_row(row) for row in rows]

    def best_run(self) -> dict[str, Any] | None:
        self.reap_timeouts()
        scored = [
            row
            for row in self.latest_by_n().values()
            if row.get("status") == "succeeded" and row.get("val_bpb") is not None
        ]
        return dict(min(scored, key=lambda row: float(row["val_bpb"]))) if scored else None

    def get_run(self, n: int) -> dict[str, Any] | None:
        self.reap_timeouts()
        row = self.latest_by_n().get(n)
        return dict(row) if row else None

    def submission_status(self) -> dict[str, int | None]:
        self.reap_timeouts()
        rows = [row for row in self.latest_by_slug().values() if row.get("slug") != "baseline"]
        used = sum(1 for row in rows if row.get("status") in SUBMITTED_STATUSES)
        created = sum(1 for row in rows if row.get("status") == "created")
        submitted_running = sum(1 for row in rows if row.get("status") == "submitted")
        succeeded = sum(1 for row in rows if row.get("status") == "succeeded")
        failed = sum(1 for row in rows if row.get("status") in FAIL_STATUSES)
        remaining = None if self.max_submissions is None else max(0, self.max_submissions - used - created)
        return {
            "max_submissions": self.max_submissions,
            "used_submissions": used,
            "created_not_submitted": created,
            "submitted_running": submitted_running,
            "succeeded": succeeded,
            "failed": failed,
            "remaining_submissions": remaining,
        }

    def finalize(
        self,
        row: dict[str, Any],
        result: dict[str, Any],
        hypothesis: str = "",
    ) -> dict[str, Any]:
        trial_dir = Path(str(row["trial_dir"]))
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        (trial_dir / "stdout.txt").write_text(stdout)
        (trial_dir / "stderr.txt").write_text(stderr)
        (trial_dir / "run.log").write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""))

        val_bpb = result.get("val_bpb")
        val_bpb = float(val_bpb) if val_bpb is not None else None
        status = str(result.get("status") or ("succeeded" if val_bpb is not None else "crashed"))
        final = {
            **row,
            "hypothesis": hypothesis or row.get("hypothesis", ""),
            "status": status,
            "val_bpb": val_bpb,
            "score": -val_bpb if val_bpb is not None else None,
            "elapsed_s": float(result.get("elapsed_s") or 0.0),
            "gpu": result.get("gpu") or self.modal_config.gpu,
            "job_id": result.get("job_id"),
            "returncode": result.get("returncode"),
            "log_path": str(trial_dir / "run.log"),
            "stdout_path": str(trial_dir / "stdout.txt"),
            "stderr_path": str(trial_dir / "stderr.txt"),
            "result_path": str(trial_dir / "result.json"),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "ts": time.time(),
        }
        for key, value in result.items():
            if key not in final and key not in {"stdout", "stderr"}:
                final[key] = value
        (trial_dir / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True))
        self.append(final)
        return final

    def reap_timeouts(self) -> None:
        now = time.time()
        for row in list(self.latest_by_slug().values()):
            age = now - float(row.get("ts") or row.get("created_at") or now)
            if row.get("status") == "created" and age >= self.created_timeout_s:
                self.append({**row, "status": "abandoned", "stale_age_s": age, "ts": now})
            elif row.get("status") == "submitted" and age >= self.submitted_timeout_s:
                self.append({**row, "status": "timeout", "stale_age_s": age, "ts": now})

    def rows(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("n"), int):
                out.append(row)
        return out

    def append(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def latest_by_slug(self) -> dict[str, dict[str, Any]]:
        latest = {}
        for row in self.rows():
            latest[str(row["slug"])] = row
        return latest

    def latest_by_n(self) -> dict[int, dict[str, Any]]:
        latest = {}
        for row in self.rows():
            latest[int(row["n"])] = row
        return latest

    def latest(self, slug: str) -> dict[str, Any] | None:
        return self.latest_by_slug().get(slug)

    def next_n(self) -> int:
        return max((int(row["n"]) for row in self.rows()), default=-1) + 1

    def _parent_dir(self, parent_slug: str | None) -> tuple[Path, str | None]:
        if parent_slug:
            parent = self.latest(parent_slug)
            if parent is None or parent.get("status") != "succeeded":
                raise SubmissionError(f"parent {parent_slug!r} is not a succeeded trial")
            return Path(str(parent["trial_dir"])), str(parent["slug"])
        best = self.best_run()
        if best:
            return Path(str(best["trial_dir"])), str(best["slug"])
        return self.base_dir, None

    def _check_budget(self, *, count_created: bool) -> None:
        if self.max_submissions is None:
            return
        status = self.submission_status()
        used = int(status["used_submissions"] or 0)
        created = int(status["created_not_submitted"] or 0) if count_created else 0
        if used + created >= self.max_submissions:
            raise SubmissionError(f"too many trials: max_submissions={self.max_submissions}")


def build_autoresearch_tools(state: AutoresearchState) -> list[Callable[..., object]]:
    @tool("Create or return the baseline trial and run it once.", proxy=True)
    def run_baseline() -> dict[str, Any]:
        return state.run_baseline()

    @tool("Create a fresh copied trial directory for one hypothesis.", proxy=True)
    def create_trial(slug: str, hypothesis: str, parent_slug: str | None = None) -> dict[str, Any]:
        return state.create_trial(slug, hypothesis, parent_slug)

    @tool("Submit an existing trial slug and return the ledger row.", proxy=True)
    def submit_trial(slug: str, hypothesis: str = "") -> dict[str, Any]:
        return state.submit_trial(slug, hypothesis)

    @tool("Compact best-first ledger view.", proxy=True)
    def list_runs() -> list[dict[str, Any]]:
        return state.list_runs()

    @tool("Best successful/scored trial, or None if none has scored.", proxy=True)
    def best_run() -> dict[str, Any] | None:
        return state.best_run()

    @tool("Full latest ledger row for trial number n.", proxy=True)
    def get_run(n: int) -> dict[str, Any] | None:
        return state.get_run(n)

    @tool("Submission budget and pending created trial counts.", proxy=True)
    def submission_status() -> dict[str, int | None]:
        return state.submission_status()

    return [
        ExperimentCrashed,
        SubmissionError,
        run_baseline,
        create_trial,
        submit_trial,
        list_runs,
        best_run,
        get_run,
        submission_status,
    ]


def run(args: argparse.Namespace) -> None:
    example_dir = Path(__file__).resolve().parent
    modal_config = ModalConfig(
        app_name=args.app_name,
        gpu=args.gpu,
        parallel=args.parallel,
        timeout_s=args.modal_timeout_s,
    )
    validate_gpu(modal_config)

    state = AutoresearchState(
        example_dir=example_dir,
        out_dir=args.out,
        modal_config=modal_config,
        max_submissions=args.max_submissions,
        created_timeout_s=args.created_trial_timeout_s,
        submitted_timeout_s=args.submitted_trial_timeout_s,
    )
    state.prepare_base()

    print("[autoresearch] preflighting Modal...", flush=True)
    config = {
        "model": args.model,
        "gpu": args.gpu,
        "parallel": args.parallel,
        "max_submissions": args.max_submissions,
        "modal_timeout_s": args.modal_timeout_s,
        "created_trial_timeout_s": args.created_trial_timeout_s,
        "submitted_trial_timeout_s": args.submitted_trial_timeout_s,
        "agent_runtime": args.agent_runtime,
        "docker_image": args.docker_image,
        "preflight": preflight(modal_config),
        "run_id": state.run_id,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    runtime = build_runtime(args.agent_runtime, args.docker_image, args.out)
    runtime.register_tools(FILE_TOOLS)
    runtime.register_tools(build_autoresearch_tools(state))

    flow = Flow(
        build_llm(args.model),
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        max_concurrency=args.parallel,
        prompt_builder=build_prompt_builder(),
    )

    query = f"""\
Run autoresearch for up to {args.max_submissions} non-baseline submissions.

Use INPUTS["task_instructions"] for task context. Use the system prompt for the
rflow loop policy and examples.

Start with run_baseline(); it is cached and does not run Modal. Then repeatedly
inspect list_runs()/best_run()/submission_status(), launch one small guided
planner batch, and print results. Continue until
submission_status()["remaining_submissions"] == 0.
"""
    graph_dir = args.out / "graph"
    graph = Graph(query=query)
    inputs = {"task_instructions": (example_dir / "program.md").read_text()}
    try:
        graph.save(graph_dir)
        renderer = LiveTreeRenderer(clear=not args.no_live)

        async def drive() -> None:
            async for event in flow.run_streaming(graph, inputs=inputs):
                graph.save(graph_dir)
                if args.no_live:
                    print(render_tree(graph), flush=True)
                else:
                    renderer.handle(event, graph)

        asyncio.run(drive())
        print(graph.result())
        write_run_report(state, args.out)
    finally:
        flow.close_repls(graph.graph_id)


def build_llm(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def build_runtime(kind: str, docker_image: str, workdir: Path):
    if kind == "docker":
        return DockerRuntime(docker_image, working_directory=workdir)
    if kind == "local":
        return LocalRuntime(working_directory=workdir)
    return SubprocessRuntime(working_directory=workdir)


def write_run_report(state: AutoresearchState, out_dir: Path) -> None:
    rows = state.list_runs()
    best = state.best_run()
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps({"best": best, "runs": rows}, indent=2, sort_keys=True))
    print(f"\n[autoresearch] report={report_path}")
    print(f"[autoresearch] ledger={state.ledger_path}")
    if best:
        print(f"[autoresearch] best={best.get('slug')} val_bpb={best.get('val_bpb')}")


def _preflight(trial_dir: Path) -> dict[str, Any] | None:
    if not _is_relative_to(trial_dir.resolve(), trial_dir.parents[1].resolve()):
        raise SubmissionError(f"bad trial path: {trial_dir}")
    train = trial_dir / "train.py"
    if not train.exists():
        raise SubmissionError(f"missing train.py: {trial_dir}")
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(train)],
        cwd=str(trial_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode == 0:
        return None
    return {
        "status": "preflight_failed",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
        "elapsed_s": 0.0,
    }


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n",
        "slug",
        "status",
        "val_bpb",
        "score",
        "elapsed_s",
        "gpu",
        "job_id",
        "parent_slug",
        "hypothesis",
        "trial_dir",
        "agent_trial_dir",
        "source_path",
        "agent_source_path",
        "log_path",
    )
    return {key: row.get(key) for key in keys}


def _rank_key(row: dict[str, Any]) -> tuple[int, float, int]:
    if row.get("status") == "succeeded" and row.get("val_bpb") is not None:
        return (0, float(row["val_bpb"]), int(row.get("n", 0)))
    if row.get("status") in RUNNING_STATUSES:
        return (1, 0.0, int(row.get("n", 0)))
    return (2, 0.0, int(row.get("n", 0)))


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower())
    return (value.strip("_") or "trial")[:80]


def _relative_to(path: Path, parent: Path) -> str:
    try:
        return str(path.resolve().relative_to(parent.resolve()))
    except ValueError:
        return str(path)


def _tail(value: str, limit: int = 4000) -> str:
    return value[-limit:]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "_runs" / "autoresearch" / time.strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-submissions", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--app-name", default="rlmflow-autoresearch")
    parser.add_argument("--modal-timeout-s", type=int, default=1200)
    parser.add_argument("--created-trial-timeout-s", type=int, default=1800)
    parser.add_argument("--submitted-trial-timeout-s", type=int, default=1500)
    parser.add_argument("--agent-runtime", choices=("subprocess", "local", "docker"), default="subprocess")
    parser.add_argument("--docker-image", default="rlmflow:local")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-live", action="store_true")
    args = parser.parse_args()
    if args.out is None:
        args.out = default_out_dir()
    if args.parallel < 1 or args.max_submissions < 0 or args.max_depth < 1:
        raise SystemExit("bad parallel/max-submissions/max-depth")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
