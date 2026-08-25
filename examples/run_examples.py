"""Run the example suite as smoke tests.

Default usage runs deterministic/offline examples only:

    python examples/run_examples.py

Opt into heavier examples when you have the dependencies/credentials:

    python examples/run_examples.py --include-optional
    python examples/run_examples.py --include-live
    python examples/run_examples.py --all

``--all`` covers every category except ``slow``: those examples take long enough
that a full suite run should not wait on them, so they need ``--include-slow``.

Every run also writes a markdown report naming each example that passed, failed,
or was skipped, with the captured output of each failure. The console summary
scrolls away behind the examples' own output; the report does not.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

Category = Literal["offline", "optional", "live", "sandbox", "manual", "slow"]
Status = Literal["pass", "fail", "skip"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "examples" / "_runs" / "examples_report.md"


@dataclass(frozen=True)
class Example:
    name: str
    path: str
    category: Category = "offline"
    args: tuple[str, ...] = ()
    env: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    timeout: int = 120
    stdin: str | None = None
    note: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)

    def command(self, tmpdir: Path) -> list[str]:
        return [
            sys.executable,
            str(REPO_ROOT / self.path),
            *[arg.format(tmp=tmpdir) for arg in self.args],
        ]


EXAMPLES: list[Example] = [
    Example("controller-injection", "examples/control/controller_injection.py"),
    Example("streaming-until", "examples/control/streaming_until.py"),
    Example("reuse-repl", "examples/control/delegation/reuse_repl.py"),
    Example("nonblocking-delegation", "examples/control/delegation/nonblocking.py"),
    Example(
        "step-until",
        "examples/control/delegation/step_until.py",
        category="live",
        args=("--max-iters", "8", "--out-dir", "{tmp}/step-until"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
        note="uses a live model to demonstrate minimal step(..., until=...) boundaries",
    ),
    Example("llm-query-batched", "examples/llm_query_batched.py"),
    Example(
        "structured-output",
        "examples/structured_output.py",
        category="live",
        args=("--max-iters", "8", "--out-dir", "{tmp}/structured-output"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
        note="uses a live model to validate root and child structured outputs",
    ),
    Example(
        "best-of-n",
        "examples/control/branching/best_of_n.py",
        args=("--n", "4", "--root-dir", "{tmp}/best_of_n"),
    ),
    Example(
        "fork-repair",
        "examples/control/branching/fork_repair.py",
        args=("--root-dir", "{tmp}/fork_repair"),
        modules=("pytest",),
    ),
    Example(
        "showcase",
        "examples/showcase.py",
        args=("--out-dir", "{tmp}/showcase"),
    ),
    Example("graph-query", "examples/graph/01_query.py"),
    Example("graph-navigate", "examples/graph/02_navigate.py"),
    Example(
        "graph-save-load",
        "examples/graph/04_save_load.py",
        args=("--out-dir", "{tmp}/graph-save-load"),
    ),
    Example("graph-timeline", "examples/graph/05_timeline.py"),
    Example(
        "graph-fork",
        "examples/graph/06_fork.py",
        args=("--out-dir", "{tmp}/graph-fork"),
    ),
    Example(
        "graph-deep-tree",
        "examples/graph/deep_tree.py",
        args=("--depth", "2", "--branch", "2", "--out-dir", "{tmp}/graph-deep-tree"),
        note="builds an agent tree by hand and plots it; no model",
    ),
    Example("autoresearch-help", "examples/autoresearch/run.py", args=("--help",)),
    Example(
        "drop-in-llm",
        "examples/drop_in_llm.py",
        category="live",
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "skills",
        "examples/skills.py",
        category="live",
        args=("--out-dir", "{tmp}/skills"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
        note="on-disk SKILL.md loaded into a callable prompt section; live LLM",
    ),
    Example(
        "dynamic-skills",
        "examples/dynamic_skills.py",
        args=("--scripted", "--print-prompt", "--out-dir", "{tmp}/dynamic-skills"),
        note="agent installs a skill mid-run via add_skill; prompt grows next turn",
    ),
    Example(
        "shepherd",
        "examples/shepherd/shepherd.py",
        category="slow",
        args=("--out-dir", "{tmp}/shepherd"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        # Solving Sokoban under a meta-agent runs past 10 minutes, so it is opt-in
        # and gets a ceiling it can actually finish inside of.
        timeout=2400,
        note="backtrack-and-branch: worker dead-ends in Sokoban, meta-agent rewinds + fans out",
    ),
    Example(
        "summarizer",
        "examples/summarizer.py",
        category="live",
        args=(
            "--sections",
            "6",
            "--max-iters",
            "8",
            "--out-dir",
            "{tmp}/summarizer",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "needle-haystack",
        "examples/needle/haystack.py",
        category="live",
        args=(
            "--num-lines",
            "2000",
            "--max-iters",
            "8",
            "--out-dir",
            "{tmp}/needle/haystack",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "needle-haystack-filesystem",
        "examples/needle/filesystem.py",
        category="live",
        args=(
            "--num-files",
            "50",
            "--max-iters",
            "8",
            "--out-dir",
            "{tmp}/needle-filesystem",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "delegation-behavior",
        "examples/behavior/delegation.py",
        category="live",
        args=("--out-dir", "{tmp}/delegation-behavior"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        # Three scenarios, one of which fans out; the heavy boids scenario is
        # opt-in via --scenario and is not part of this run.
        timeout=1200,
        note="grades delegation both ways: fan out on substantial work, stay local on trivial work",
    ),
    Example(
        "dspy-drop-in",
        "examples/providers/dspy_drop_in.py",
        category="live",
        env=("OPENAI_API_KEY",),
        modules=("dspy", "openai"),
        timeout=300,
    ),
    Example(
        "mcp-weather",
        "examples/providers/mcp_weather.py",
        category="live",
        args=("--max-iters", "8", "--out-dir", "{tmp}/mcp-weather"),
        env=("OPENAI_API_KEY",),
        modules=("mcp", "openai"),
        timeout=300,
        note="uses Open-Meteo through a local MCP server",
    ),
    Example(
        "injection-word-search",
        "examples/control/injection/word_search.py",
        category="live",
        args=("--out-dir", "{tmp}/word-search-baseline"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=600,
        note="generates the baseline graph used by injection-variants",
    ),
    Example(
        "injection-variants",
        "examples/control/injection/inject_variants.py",
        category="live",
        args=("--source", "{tmp}/word-search-baseline", "--out", "{tmp}/word-search"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=600,
        note="injects alternate prompts into the baseline generated by injection-word-search",
    ),
    Example(
        "sandbox-docker",
        "examples/sandboxes/docker_agent.py",
        category="sandbox",
        args=("--max-iters", "2"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=600,
    ),
    Example(
        "sandbox-modal",
        "examples/sandboxes/modal_agent.py",
        category="sandbox",
        args=("--max-iters", "2"),
        env=("OPENAI_API_KEY",),
        modules=("modal", "openai"),
        timeout=900,
    ),
    Example(
        "coding-agent-interactive",
        "examples/coding/agent.py",
        category="manual",
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        stdin="quit\n",
        note="interactive coding agent; the smoke run types quit and exits",
    ),
]


@dataclass
class Result:
    example: Example
    status: Status
    seconds: float = 0.0
    #: Skip reason, or the failure message with the tail of the captured output.
    detail: str = ""
    command: str = ""

    @property
    def summary(self) -> str:
        """The first line of ``detail``, for the report's one-row-per-example table."""
        return self.detail.splitlines()[0] if self.detail else ""


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def should_include(example: Example, args: argparse.Namespace) -> bool:
    if (
        args.pattern
        and args.pattern.lower() not in example.name.lower()
        and args.pattern not in example.path
    ):
        return False
    return (
        example.category == "offline"
        or (example.category == "optional" and args.include_optional)
        or (example.category == "live" and args.include_live)
        or (example.category == "sandbox" and args.include_sandbox)
        or (example.category == "manual" and args.include_manual)
        or (example.category == "slow" and args.include_slow)
    )


def skip_reason(example: Example) -> str | None:
    missing_env = [name for name in example.env if not os.environ.get(name)]
    if missing_env:
        return "missing env: " + ", ".join(missing_env)
    missing_modules = [name for name in example.modules if not module_exists(name)]
    if missing_modules:
        return "missing modules: " + ", ".join(missing_modules)
    if not (REPO_ROOT / example.path).exists():
        return "missing path"
    return None


def run_example(
    example: Example,
    tmpdir: Path,
    *,
    verbose: bool,
    installed_package: bool,
) -> tuple[str, float]:
    command = example.command(tmpdir)
    env = os.environ.copy()
    if installed_package:
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(example.extra_env)

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        input=example.stdin,
        text=True,
        capture_output=True,
        timeout=example.timeout,
    )
    elapsed = time.perf_counter() - start
    output = (proc.stdout + proc.stderr).strip()
    if verbose and output:
        print(output)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise RuntimeError(f"exit code {proc.returncode}\n{tail}")
    return output, elapsed


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def cell(text: str, limit: int = 120) -> str:
    """One table cell: single line, pipes escaped, clipped."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.replace("|", "\\|")


def fenced(text: str) -> list[str]:
    """Fence a captured block, outrunning any backticks the example printed itself."""
    body = text or "(no output captured)"
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence, body, fence]


def report_markdown(results: list[Result], args: argparse.Namespace, seconds: float) -> str:
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("pass", "fail", "skip")
    }
    included = [
        name
        for name in ("optional", "live", "sandbox", "manual", "slow")
        if getattr(args, f"include_{name}")
    ]

    lines = [
        "# Example suite report",
        "",
        f"- generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        f"- commit: `{git_commit()}`",
        f"- python: {platform.python_version()} on {platform.system().lower()}",
        f"- categories: offline{''.join(f' + {name}' for name in included)}",
        f"- total time: {seconds:.1f}s",
        "",
        f"**{counts['pass']} passed · {counts['fail']} failed · {counts['skip']} skipped**",
    ]
    if args.pattern:
        lines += ["", f"Filtered to names/paths containing `{args.pattern}`."]
    if args.strict_skips and counts["skip"]:
        lines += ["", "`--strict-skips` is on, so the skips below also fail the run."]

    lines += [
        "",
        "| result | example | category | time | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    marks = {"pass": "pass", "fail": "**FAIL**", "skip": "skip"}
    for result in results:
        elapsed = f"{result.seconds:.1f}s" if result.status != "skip" else "—"
        lines.append(
            f"| {marks[result.status]} "
            f"| `{cell(result.example.name)}` "
            f"| {result.example.category} "
            f"| {elapsed} "
            f"| {cell(result.summary or result.example.note)} |"
        )

    failures = [result for result in results if result.status == "fail"]
    if failures:
        lines += ["", "## Failures", ""]
        for result in failures:
            lines += [
                f"### {result.example.name}",
                "",
                f"`{result.example.path}`, ran for {result.seconds:.1f}s as:",
                "",
                *fenced(result.command),
                "",
                *fenced(result.detail),
                "",
            ]

    skips = [result for result in results if result.status == "skip"]
    if skips:
        lines += ["", "## Skipped", ""]
        lines += [f"- `{result.example.name}` — {result.detail}" for result in skips]
        lines += [""]

    text = "\n".join(lines)
    while "\n\n\n" in text:  # sections each pad themselves; don't double up
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip() + "\n"


def write_report(
    path: Path, results: list[Result], args: argparse.Namespace, seconds: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown(results, args, seconds), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-optional", action="store_true", help="run optional-dependency examples"
    )
    parser.add_argument(
        "--include-live", action="store_true", help="run examples that call live LLM APIs"
    )
    parser.add_argument(
        "--include-sandbox", action="store_true", help="run Modal/E2B/Daytona examples"
    )
    parser.add_argument(
        "--include-manual", action="store_true", help="include interactive/manual smoke checks"
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="run long examples that --all deliberately leaves out",
    )
    parser.add_argument(
        "--all", action="store_true", help="enable every include flag except --include-slow"
    )
    parser.add_argument(
        "--list", action="store_true", help="list selected examples without running them"
    )
    parser.add_argument(
        "--pattern", help="only include examples whose name/path contains this text"
    )
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failure")
    parser.add_argument(
        "--strict-skips", action="store_true", help="treat missing env/deps as failure"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print full output for successful examples"
    )
    parser.add_argument(
        "--installed-package",
        action="store_true",
        help="run examples against the installed rlmflow package instead of the source tree",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"markdown report path (default: {DEFAULT_REPORT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--no-report", action="store_true", help="skip writing the report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all:
        args.include_optional = True
        args.include_live = True
        args.include_sandbox = True
        args.include_manual = True

    selected = [example for example in EXAMPLES if should_include(example, args)]
    if args.list:
        for example in selected:
            reason = skip_reason(example)
            status = "skip: " + reason if reason else "ready"
            note = f" ({example.note})" if example.note else ""
            print(f"{example.name:<28} {example.category:<8} {status}{note}")
        return 0

    results: list[Result] = []
    run_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="rlmflow-example-runs-") as raw_tmpdir:
        tmpdir = Path(raw_tmpdir)
        print(f"Running {len(selected)} selected examples. temp outputs: {tmpdir}")
        for example in selected:
            reason = skip_reason(example)
            label = f"{example.name} ({example.path})"
            if reason:
                print(f"SKIP {label}: {reason}")
                results.append(Result(example, "skip", detail=reason))
                continue

            print(f"RUN  {label}")
            # Repo-relative, so a failing command can be pasted into an issue as-is.
            command = " ".join(example.command(tmpdir)).replace(f"{REPO_ROOT}{os.sep}", "")
            started = time.perf_counter()
            try:
                _output, elapsed = run_example(
                    example,
                    tmpdir,
                    verbose=args.verbose,
                    installed_package=args.installed_package,
                )
            except subprocess.TimeoutExpired:
                detail = f"timed out after {example.timeout}s"
                print(f"FAIL {example.name}: {detail}")
                results.append(
                    Result(
                        example,
                        "fail",
                        seconds=time.perf_counter() - started,
                        detail=detail,
                        command=command,
                    )
                )
                if args.fail_fast:
                    break
            except Exception as exc:  # noqa: BLE001 - any failure is one example's result
                print(f"FAIL {example.name}: {exc}")
                results.append(
                    Result(
                        example,
                        "fail",
                        seconds=time.perf_counter() - started,
                        detail=str(exc),
                        command=command,
                    )
                )
                if args.fail_fast:
                    break
            else:
                note = f" [{example.note}]" if example.note else ""
                print(f"PASS {example.name} ({elapsed:.1f}s){note}")
                results.append(Result(example, "pass", seconds=elapsed, command=command))

    total_seconds = time.perf_counter() - run_started
    failures = [result for result in results if result.status == "fail"]
    skipped = [result for result in results if result.status == "skip"]

    print("\nSummary")
    print(f"  passed : {sum(result.status == 'pass' for result in results)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed : {len(failures)}")

    if skipped:
        print("\nSkipped")
        for result in skipped:
            print(f"  - {result.example.name}: {result.detail}")
    if failures:
        print("\nFailures")
        for result in failures:
            print(f"  - {result.example.name}: {result.summary}")

    if not args.no_report:
        write_report(args.report, results, args, total_seconds)
        print(f"\nReport: {args.report}")

    return 1 if failures or (args.strict_skips and skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
