"""Run the example suite as smoke tests.

Default usage runs deterministic/offline examples only:

    python examples/run_examples.py

Opt into heavier examples when you have the dependencies/credentials:

    python examples/run_examples.py --include-optional
    python examples/run_examples.py --include-live
    python examples/run_examples.py --all
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Category = Literal["offline", "optional", "live", "sandbox", "manual"]

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        args=("--no-viz", "--out-dir", "{tmp}/showcase"),
    ),
    Example("graph-query", "examples/graph/01_query.py"),
    Example("graph-navigate", "examples/graph/02_navigate.py"),
    Example("graph-mutate", "examples/graph/03_mutate.py"),
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
    Example("graph-render", "examples/graph/07_render.py"),
    Example("autoresearch-help", "examples/autoresearch/run.py", args=("--help",)),
    Example(
        "view-demo",
        "examples/view_demo.py",
        args=("--no-launch",),
        note="opens a minimal render_tree-backed viewer or prints a fallback",
    ),
    Example(
        "tui-chat",
        "examples/tui_chat.py",
        category="manual",
        env=("OPENAI_API_KEY",),
        modules=("openai", "textual"),
        note="opens the live Textual TUI against a real OpenAI-backed Flow",
    ),
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
        "shepherd-self-test",
        "examples/shepherd/shepherd.py",
        args=("--self-test",),
        note="offline check of the Sokoban env (trap deadlocks, verified line solves)",
    ),
    Example(
        "shepherd",
        "examples/shepherd/shepherd.py",
        category="live",
        args=("--out-dir", "{tmp}/shepherd"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=600,
        note="backtrack-and-branch: worker dead-ends in Sokoban, meta-agent rewinds + fans out",
    ),
    Example(
        "summarizer",
        "examples/summarizer.py",
        category="live",
        args=(
            "--sections", "6", "--no-viz", "--max-iters", "8",
            "--out-dir", "{tmp}/summarizer",
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
            "--num-lines", "2000", "--no-viz", "--max-iters", "8",
            "--out-dir", "{tmp}/needle/haystack",
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
            "--num-files", "50", "--no-viz", "--max-iters", "8",
            "--out-dir", "{tmp}/needle-filesystem",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
    ),
    Example(
        "needle-minimal-filesystem",
        "examples/needle/minimal_filesystem.py",
        category="live",
        args=(
            "--num-files", "30", "--lines-per-file", "40", "--max-iters", "8",
            "--out-dir", "{tmp}/needle-minimal-filesystem",
        ),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=300,
        note="uses the tiny rflow graph/event stack with FILE_TOOLS",
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
        args=("--no-viz", "--max-iters", "8", "--out-dir", "{tmp}/mcp-weather"),
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
        args=("--max-iters", "2", "--no-live"),
        env=("OPENAI_API_KEY",),
        modules=("openai",),
        timeout=600,
    ),
    Example(
        "sandbox-modal",
        "examples/sandboxes/modal_agent.py",
        category="sandbox",
        args=("--max-iters", "2", "--no-live"),
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
        note="interactive shell; smoke only starts and exits",
    ),
]


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def should_include(example: Example, args: argparse.Namespace) -> bool:
    if args.pattern and args.pattern.lower() not in example.name.lower() and args.pattern not in example.path:
        return False
    return (
        example.category == "offline"
        or (example.category == "optional" and args.include_optional)
        or (example.category == "live" and args.include_live)
        or (example.category == "sandbox" and args.include_sandbox)
        or (example.category == "manual" and args.include_manual)
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


def run_example(example: Example, tmpdir: Path, *, verbose: bool) -> tuple[str, float]:
    command = example.command(tmpdir)
    env = os.environ.copy()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true", help="run optional-dependency examples")
    parser.add_argument("--include-live", action="store_true", help="run examples that call live LLM APIs")
    parser.add_argument("--include-sandbox", action="store_true", help="run Modal/E2B/Daytona examples")
    parser.add_argument("--include-manual", action="store_true", help="include interactive/manual smoke checks")
    parser.add_argument("--all", action="store_true", help="enable every include flag")
    parser.add_argument("--list", action="store_true", help="list selected examples without running them")
    parser.add_argument("--pattern", help="only include examples whose name/path contains this text")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failure")
    parser.add_argument("--strict-skips", action="store_true", help="treat missing env/deps as failure")
    parser.add_argument("--verbose", action="store_true", help="print full output for successful examples")
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

    failures: list[str] = []
    skipped: list[str] = []
    passed = 0

    with tempfile.TemporaryDirectory(prefix="rlmflow-example-runs-") as raw_tmpdir:
        tmpdir = Path(raw_tmpdir)
        print(f"Running {len(selected)} selected examples. temp outputs: {tmpdir}")
        for example in selected:
            reason = skip_reason(example)
            label = f"{example.name} ({example.path})"
            if reason:
                print(f"SKIP {label}: {reason}")
                skipped.append(f"{example.name}: {reason}")
                if args.strict_skips:
                    failures.append(f"{example.name}: {reason}")
                continue

            print(f"RUN  {label}")
            try:
                _output, elapsed = run_example(example, tmpdir, verbose=args.verbose)
            except subprocess.TimeoutExpired:
                message = f"{example.name}: timed out after {example.timeout}s"
                print(f"FAIL {message}")
                failures.append(message)
                if args.fail_fast:
                    break
            except Exception as exc:  # noqa: BLE001
                message = f"{example.name}: {exc}"
                print(f"FAIL {message}")
                failures.append(message)
                if args.fail_fast:
                    break
            else:
                passed += 1
                note = f" [{example.note}]" if example.note else ""
                print(f"PASS {example.name} ({elapsed:.1f}s){note}")

    print("\nSummary")
    print(f"  passed : {passed}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed : {len(failures)}")

    if skipped:
        print("\nSkipped")
        for item in skipped:
            print(f"  - {item}")
    if failures:
        print("\nFailures")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
