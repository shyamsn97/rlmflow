"""rlmflow runner for autoresearch on Modal."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlmflow import (
    FILE_TOOLS,
    AgentStart,
    DockerRuntime,
    Flow,
    LocalRuntime,
    SubprocessRuntime,
    SystemPromptBuilder,
    start,
    tool,
)
from rlmflow.consumers import WorkspaceSync

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client  # noqa: E402

try:  # Allow both `python examples/autoresearch/run.py` and imports.
    from .modal_runner import ModalConfig, preflight, submit, validate_gpu
    from .plot_progress import load_trials, plot_matplotlib, plot_svg, plot_tree_svg
except ImportError:  # pragma: no cover
    from modal_runner import ModalConfig, preflight, submit, validate_gpu
    from plot_progress import load_trials, plot_matplotlib, plot_svg, plot_tree_svg


PROGRESS_TITLE = "Tiny Autoresearch — (TinyStories + GPT2)"


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
- Hierarchy is flat: root is the planner. Each turn root plans a wave, fans out
  several implementer children in ONE `launch_subagents([...])` call (they run in
  parallel), then those children come back and root plans the next wave from the
  new results. Anything passed in a single `launch_subagents` list runs
  concurrently, so prefer wide waves over launching one child at a time. Children
  may block while `submit_trial(...)` runs its Modal job.

Diagnose before proposing:
- Read at least one prior log with `get_run(n)` before planning a wave (start
  with the baseline). Every hypothesis must cite a number from a log, e.g. the
  final `smooth`-loss trend, the final `lr`, `num_steps`, `total_tokens_M`, or
  `peak_vram_mb` headroom. Do not propose blind.
- Let the loss curve pick the knob, not habit. From the log judge whether the
  run is UNDER-trained/under-fit (smooth train loss still clearly decreasing at
  the final step; far from plateau) or OVER-fit/unstable (loss flat, rising, or
  spiky). If under-fit, favor knobs that increase effective optimization or
  capacity per second of the fixed time budget; if over-fit/unstable, favor
  knobs that regularize or stabilize. NEVER apply regularization (dropout,
  weight decay, grad clip) to a run that is still under-fit — it can only make
  an under-trained model worse.

Cover diverse hypotheses each wave:
- Within a wave, vary which knob each child changes so the wave explores
  different directions instead of the same one many times. Any knob listed in
  the task's Constraints is fair game; it is up to you to work out which ones
  actually move `val_bpb`.
- Think across FAMILIES of approaches, not just scalar knobs: optimization and
  LR schedule, data/batching/tokens-per-step, and — where the code allows —
  ARCHITECTURE changes (attention variants, activation/normalization choices,
  weight tying, initialization, depth/width trade-offs). A run that only tunes
  numbers leaves the interesting wins on the table; when scalar tuning plateaus,
  reach for a structural change. Stay within the task's Constraints and edit
  only `train.py`.
- Map sensitivity first, tune second. Spend the FIRST wave(s) on a
  one-factor-at-a-time scan: each child changes a DIFFERENT single knob by a
  LARGE amount from baseline (multiply/divide, not ±10%). The goal of early
  waves is to measure which knobs move `val_bpb` at all, not to win. Only once
  you can rank knobs by observed effect should you invest the rest of the budget
  in the 2-3 most sensitive ones.
- Perturb big, then bracket. Textbook default values (e.g. dropout 0.1, wd 0.1)
  reveal almost nothing; expose sensitivity with order-of-magnitude moves. When
  a knob helps, submit one trial pushing it FURTHER and, if plausible, one the
  OTHER way, to confirm direction and locate the sweet spot before combining
  winners.
- Don't fixate: if a knob has produced only noise-scale moves across the trials
  so far, branch toward a different, untried direction rather than re-tuning it.

Build on what worked; never repeat a trial:
- Before proposing, call `list_runs()` and read every prior `slug`, `hypothesis`,
  and `val_bpb`. NEVER propose a hypothesis or slug that already has a row
  (succeeded, created, or submitted). Re-running finished work wastes the budget;
  `create_trial` will reject an exact-duplicate hypothesis.
- Exploit (about two thirds): branch from `best_run()` and push the knob that
  helped it further, or COMBINE two changes that each beat their parent.
- Diversify (about one third): branch from `sample_valid_run()` (a random
  already-succeeded trial, not the best) toward a genuinely new knob combination.
- Treat improvements smaller than run-to-run eval noise as "no effect"; don't
  build a long greedy chain of ~0.001 gains.
- Each turn, print a knob -> best observed change-in-`val_bpb` table and spend
  the next wave proportional to measured effect; explicitly deprioritize any
  knob whose best move is within noise. Never create `<slug>_alt` near-duplicates
  of an idea already tried — that is wasted budget.

Spend the WHOLE budget — a plateau means pivot, not stop:
- The ONLY reason to call `done(...)` is `submission_status()["remaining_submissions"]
  == 0`. While it is > 0 you are NOT finished, no matter how the last wave went.
  A flat wave is not a stopping signal — it is a signal to change direction.
- Plan ONE wave per turn, then FINISH the turn (print your results and stop the
  code block) so your NEXT turn can reason about the newest results and invent
  fresh hypotheses. Do NOT write a single script with a `while` loop that
  pre-enumerates many waves from a fixed candidate list and calls `done()` at
  the end — that hardcodes a finite menu and quits the moment it is drained.
  Returning between waves is NOT stopping; the run continues on your next turn.
- Running out of ideas is NOT a stop condition. If your candidate list dedups to
  empty, that means you must BRAINSTORM genuinely new hypotheses, not quit:
  reach for an untried knob, a larger/opposite perturbation, an architecture
  change (attention/normalization/activation/weight-tying/init), or a fresh
  COMBINATION of two prior winners. NEVER break out of planning or call `done()`
  with the reasoning "no new ideas / only duplicates left" while budget remains
  — there is always another untried direction within the task Constraints.
- Keep waves wide: size each `launch_subagents([...])` list up to your
  `parallel` limit and up to `remaining_submissions`, so the budget is spent in
  a few wide waves rather than trickled out. Re-check `submission_status()` at
  the start of every turn and keep going until `remaining_submissions == 0`.

Editing train.py (do it the robust way):
- Implementation children edit only `INPUTS["trial_dir"]/train.py`, then call
  `submit_trial(INPUTS["slug"], INPUTS["hypothesis"])`.
- ALWAYS edit by full-file rewrite: read the entire file, then `write_file` the
  COMPLETE updated contents yourself, changing only the constants/lines your
  hypothesis needs. Do NOT use `edit_file`, and NEVER construct source by string
  mutation in the REPL (`code.replace(...)`, `.format(...)`, or f-string
  templating of the source) — that has repeatedly corrupted lines (e.g. turning
  `flush=True)` into `flush = 1798`).
- If `submit_trial` returns `preflight_failed`, read the reported line number,
  re-read the current file, and fix it with ONE more full-file `write_file`. If
  it still fails, re-read `INPUTS["trial_dir"]/train.py` from scratch and rewrite
  cleanly. Do not retry more than twice; if still broken, return the failure row.
- If it returns any other failure row, return it to the parent.
- All `launch_subagents(..., inputs=...)` values must be strings. Use `str(...)`
  for counts and JSON strings for structured values.

Root sketch (root is the planner: plan a wave, fan out implementers, repeat):
```repl
run_baseline()
status = submission_status()
remaining = status["remaining_submissions"]
if remaining == 0:
    done(str({"status": "complete", "best": best_run(), "runs": list_runs()}))
runs = list_runs()                 # everything already tried (slug, hypothesis, val_bpb)
print(get_run(0))                  # cite a number from a log in each hypothesis
best, sample = best_run(), sample_valid_run()   # exploit the leader; diversify from a random winner
# Propose only NEW ideas; branch each from a chosen parent so the wave spreads
# out. These are placeholders showing the SHAPE only — replace them with your
# own ideas, each citing a number you just read from a log:
#   (slug, hypothesis, parent_seed)
candidates = [
    ("idea_a", "<change ONE knob; cite the log number that motivates it>", best),
    ("idea_b", "<combine two changes that each helped their parent trial>", best),
    ("idea_c", "<branch a random winner toward a different, untried knob>", sample),
]
tried = {r.get("slug") for r in runs} | {r.get("hypothesis") for r in runs}
ideas = [(s, h, seed) for s, h, seed in candidates if s not in tried and h not in tried][:remaining]
trials = [create_trial(s, h, parent_slug=(seed["slug"] if seed else "baseline")) for s, h, seed in ideas]
wave = [{
    "name": row["slug"],
    "query": "Read INPUTS['trial_dir']/train.py in full, then write_file the complete "
             "updated file changing only the constants your hypothesis needs, then submit_trial(...).",
    "inputs": {"trial_dir": row["agent_trial_dir"], "slug": row["slug"], "hypothesis": row["hypothesis"]},
} for row in trials]
results = await launch_subagents(wave)     # implementer children run concurrently
# Print results and STOP here — this is ONE wave for ONE turn. Do not wrap this
# in a `while` loop over a fixed candidate list; finishing the turn lets your
# NEXT turn reason about `results` and brainstorm genuinely new hypotheses. If
# submission_status()["remaining_submissions"] > 0 you are not done; plan another
# wave next turn (pivot to a new/untried direction). Only call done() at 0.
print(results, list_runs(), best_run(), submission_status())
```
"""


def build_prompt_builder():
    prompt = SystemPromptBuilder()
    prompt.sections.add(
        "autoresearch_adapter",
        ADAPTER_PROMPT,
        title="Autoresearch Adapter",
        before="tools",
    )
    return prompt


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
        write_run_report(self, self.out_dir)

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
                if hypothesis:
                    for other in self.latest_by_slug().values():
                        if other.get("hypothesis") == hypothesis and other.get("status") in (
                            RUNNING_STATUSES | {"succeeded"}
                        ):
                            raise SubmissionError(
                                f"duplicate hypothesis already tried as {other['slug']!r}; "
                                "propose a new idea (see list_runs())"
                            )

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

    def sample_valid_run(self) -> dict[str, Any] | None:
        """A random already-succeeded trial (not necessarily the best).

        Use as a diversification seed so the search branches off known-good
        solutions instead of only creeping off the current leader.
        """
        self.reap_timeouts()
        scored = [
            row
            for row in self.latest_by_n().values()
            if row.get("status") == "succeeded" and row.get("val_bpb") is not None
        ]
        return dict(random.choice(scored)) if scored else None

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
        remaining = (
            None if self.max_submissions is None else max(0, self.max_submissions - used - created)
        )
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
            write_run_report(self, self.out_dir)

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

    @tool("Random already-succeeded trial to diversify from (not the best).", proxy=True)
    def sample_valid_run() -> dict[str, Any] | None:
        return state.sample_valid_run()

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
        sample_valid_run,
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

    flow = Flow(
        build_client(args.model),
        runtime=runtime,
        tools=[*FILE_TOOLS, *build_autoresearch_tools(state)],
        workers=args.parallel,
        system_prompt=build_prompt_builder(),
    )

    query = f"""\
Run autoresearch for up to {args.max_submissions} non-baseline submissions.

Use INPUTS["task_instructions"] for task context. Use the system prompt for the
rlmflow loop policy and examples.

Start with run_baseline(); it is cached and does not run Modal. You are the
planner: each turn inspect list_runs()/best_run()/submission_status(), then fan
out a PARALLEL WAVE of implementer children in a single launch_subagents([...])
call — several trials at once, not one at a time. The children return here; plan
the next wave from their results.

Keep launching waves until submission_status()["remaining_submissions"] == 0.
That is the ONLY stop condition — do NOT call done() while submissions remain,
even if the last wave did not improve on the best (a plateau means pivot to a
new direction, not stop). After each wave, re-check submission_status() and
launch the next one.
"""
    graph_dir = args.out / "graph"
    root = start(
        query,
        inputs={"task_instructions": (example_dir / "program.md").read_text()},
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )
    sync = (
        WorkspaceSync(args.out, args.sync_dir, every_s=args.sync_every_s)
        if args.sync_dir is not None
        else None
    )

    async def drive(root: AgentStart) -> None:
        async for node in flow.run_streaming(root):
            print(f"{node.parent_agent.config.path:<20} {node.type}", flush=True)
            root.save(graph_dir)
            if sync is not None:
                sync.handle(node)

    try:
        root.save(graph_dir)
        if sync is not None:
            sync.sync()
        asyncio.run(drive(root))
        print(root.result())
        write_run_report(state, args.out, announce=True)
    finally:
        if sync is not None:
            sync.close()
        flow.runtime.close_repls()


def build_runtime(kind: str, docker_image: str, workdir: Path):
    if kind == "docker":
        return DockerRuntime(docker_image, working_directory=workdir)
    if kind == "local":
        return LocalRuntime(working_directory=workdir)
    return SubprocessRuntime(working_directory=workdir)


def write_run_report(
    state: AutoresearchState,
    out_dir: Path,
    *,
    announce: bool = False,
) -> None:
    """Refresh the small, human-facing report derived from the full ledger."""
    rows = list(state.latest_by_n().values())
    rows.sort(key=_rank_key)
    solutions = [_report_row(row) for row in rows]
    scored = [
        row for row in rows if row.get("status") == "succeeded" and row.get("score") is not None
    ]
    best = max(scored, key=lambda row: float(row["score"])) if scored else None
    best_summary = _report_row(best) if best else None

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps({"best": best_summary, "solutions": solutions}, indent=2) + "\n"
    )
    _write_progress_plots(state.ledger_path, out_dir, announce=announce)
    (out_dir / "summary.md").write_text(_markdown_report(best_summary, solutions))

    if announce:
        print(f"\n[autoresearch] report={report_path}")
        print(f"[autoresearch] summary={out_dir / 'summary.md'}")
        print(f"[autoresearch] progress={out_dir / 'progress.svg'}")
        print(f"[autoresearch] lineage={out_dir / 'lineage.svg'}")
        print(f"[autoresearch] ledger={state.ledger_path}")
        if best_summary:
            print(f"[autoresearch] best={best_summary['name']} score={best_summary['score']}")


def _report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("slug"),
        "hypothesis": row.get("hypothesis"),
        "time_elapsed": round(float(row.get("elapsed_s") or 0.0), 3),
        "n": row.get("n"),
        "score": row.get("score"),
    }


def _markdown_report(
    best: dict[str, Any] | None,
    solutions: list[dict[str, Any]],
) -> str:
    lines = ["# Autoresearch report", ""]
    if best:
        lines.extend(
            [
                "## Best",
                "",
                f"**{_markdown_cell(best['name'])}** — score `{_format_score(best['score'])}` "
                f"(trial {best['n']}, {_format_elapsed(best['time_elapsed'])})",
                "",
                str(best.get("hypothesis") or ""),
                "",
            ]
        )
    lines.extend(
        [
            "## Solutions",
            "",
            "| n | name | hypothesis | time elapsed | score |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in solutions:
        lines.append(
            f"| {row['n']} | {_markdown_cell(row['name'])} | "
            f"{_markdown_cell(row['hypothesis'])} | "
            f"{_format_elapsed(row['time_elapsed'])} | "
            f"{_format_score(row['score'])} |"
        )
    lines.extend(
        [
            "",
            "## Progress",
            "",
            "![Autoresearch progress](progress.svg)",
            "",
            "## Lineage",
            "",
            "![Autoresearch lineage](lineage.svg)",
            "",
        ]
    )
    return "\n".join(lines)


def _write_progress_plots(ledger_path: Path, out_dir: Path, *, announce: bool = False) -> None:
    """Render the Karpathy-style progress plot (running best + kept/discarded).

    Uses the shared `plot_progress` helpers so the live report and the
    standalone script stay identical. Always writes `progress.svg` (fast,
    dependency-free); on the final call also tries a `progress.png` via
    matplotlib if it is available.
    """
    svg_path = out_dir / "progress.svg"
    tree_path = out_dir / "lineage.svg"
    trials = load_trials(ledger_path) if ledger_path.exists() else []
    if not trials:
        svg_path.write_text(_empty_progress_svg())
        tree_path.write_text(_empty_progress_svg())
        return
    plot_svg(trials, title=PROGRESS_TITLE, out=svg_path)
    plot_tree_svg(trials, title=PROGRESS_TITLE, out=tree_path)
    if announce:
        try:
            plot_matplotlib(trials, title=PROGRESS_TITLE, out=out_dir / "progress.png")
        except Exception:  # noqa: BLE001 - matplotlib is optional on the host
            pass


def _empty_progress_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" '
        'viewBox="0 0 1200 700"><rect width="100%" height="100%" fill="white"/>'
        '<text x="600" y="350" text-anchor="middle" font-family="sans-serif" '
        'fill="#6b7280" font-size="16">No succeeded trials yet</text></svg>\n'
    )


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", r"\|").replace("\n", " ")


def _format_elapsed(value: Any) -> str:
    seconds = float(value or 0.0)
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s" if minutes else f"{remaining:.1f}s"


def _format_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


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
    return (
        Path(__file__).resolve().parents[1]
        / "_runs"
        / "autoresearch"
        / time.strftime("%Y%m%d-%H%M%S")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-submissions", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--app-name", default="rlmflow-autoresearch")
    parser.add_argument("--modal-timeout-s", type=int, default=1200)
    parser.add_argument("--created-trial-timeout-s", type=int, default=1800)
    parser.add_argument("--submitted-trial-timeout-s", type=int, default=1500)
    parser.add_argument(
        "--agent-runtime", choices=("subprocess", "local", "docker"), default="subprocess"
    )
    parser.add_argument("--docker-image", default="rlmflow:local")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sync-dir", type=Path, default=None)
    parser.add_argument("--sync-every-s", type=float, default=2.0)
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
