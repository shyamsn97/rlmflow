"""Compare repair branches with the Node-only API.

Each repair attempt is a separate ``flow.start(...)`` run in its own working
directory; the resulting graphs are directly comparable.

Usage:
    python examples/control/branching/fork_repair.py
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path

from rlmflow import (
    FILE_TOOLS,
    Flow,
    GraphCheckpointer,
    LLMUsage,
    LocalRuntime,
)

TASK = "Implement slugify(text) in slugify.py so tests/test_slugify.py passes."

TESTS = """from slugify import slugify


def test_basic_words():
    assert slugify("Hello, World!") == "hello-world"


def test_preserves_numbers():
    assert slugify("Version 2.0") == "version-2-0"


def test_transliterates_unicode():
    assert slugify("Café déjà vu") == "cafe-deja-vu"


def test_collapses_repeated_separators():
    assert slugify("A---B___C") == "a-b-c"


def test_strips_edges():
    assert slugify("  Already Sluggy  ") == "already-sluggy"


def test_empty_after_cleanup():
    assert slugify("!!!") == ""
"""

BAD_IMPLEMENTATION = """import re


def slugify(text: str) -> str:
    return re.sub(r"[^a-z]+", "-", text.lower()).strip("-")
"""

GOOD_IMPLEMENTATION = """import re
import unicodedata


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = re.sub(r"-+", "-", ascii_text)
    return ascii_text.strip("-")
"""


class RepairLLM:
    def __init__(self, implementation: str, label: str) -> None:
        self.implementation = implementation
        self.label = label

    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(input_tokens=100, output_tokens=50)
        return (
            f"```repl\nwrite_file('slugify.py', {self.implementation!r})\nfinish({self.label!r})\n```"
        )


def setup_project(workdir: Path) -> None:
    (workdir / "tests").mkdir(parents=True, exist_ok=True)
    (workdir / "slugify.py").write_text(
        "def slugify(text: str) -> str:\n    raise NotImplementedError\n"
    )
    (workdir / "tests" / "test_slugify.py").write_text(TESTS)


def run_tests(files_dir: Path) -> tuple[bool, str]:
    for cache_dir in files_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q"],
        cwd=files_dir,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output


def run_branch(root: Path, name: str, implementation: str, label: str):
    workdir = root / name
    setup_project(workdir)
    # File tools run inside this branch's own working directory.
    runtime = LocalRuntime(working_directory=workdir)
    flow = Flow(RepairLLM(implementation, label), tools=FILE_TOOLS, runtime=runtime)
    graph = flow.start(TASK, max_depth=0, max_iters=3)

    async def drive() -> None:
        checkpoint = GraphCheckpointer(workdir / "graph")
        try:
            async for node in flow.run_streaming(graph):
                checkpoint.handle(node)
        finally:
            checkpoint.close()

    asyncio.run(drive())
    passed, output = run_tests(workdir)
    flow.runtime.close_repls()
    return workdir, graph, passed, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repair branches")
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).resolve().parents[2] / "_runs" / "fork-repair"),
        help="where to drop per-branch working dirs (default: examples/_runs/fork-repair/)",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    branches = [
        ("bad", BAD_IMPLEMENTATION, "wrote naive regex implementation"),
        ("unicode", GOOD_IMPLEMENTATION, "wrote unicode-normalizing implementation"),
    ]

    results = []
    for name, implementation, label in branches:
        workdir, graph, passed, output = run_branch(root, name, implementation, label)
        results.append((passed, workdir, graph, output))
        print(f"{name}: tests={'PASS' if passed else 'FAIL'} result={graph.result()!r}")
        print("  " + brief_test_output(output).replace("\n", "\n  "))

    winner = next((item for item in results if item[0]), results[0])
    _passed, workdir, _graph, _output = winner
    print(f"\n[best] {workdir.name} dir={workdir}")


def brief_test_output(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no test output)"
    interesting = [
        line
        for line in lines
        if "failed" in line.lower()
        or "passed" in line.lower()
        or "error" in line.lower()
        or line.startswith("E ")
    ]
    return "\n".join(interesting[-4:] or lines[-4:])


if __name__ == "__main__":
    main()
