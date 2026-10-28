"""Every way of running Claude Code with rlmflow, live: Claude as a harness (mode 2),
your own Claude session as the root through the plugin (mode 4), and forking both.

Each test seeds its haystack, checks the answer and the graph's shape, and saves
the run under ``tests/live/_runs/``.
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rlmflow import CODE, Graph
from rlmflow.agent_tools.hooks import PLUGIN_DIR
from rlmflow.cli.serve import INTERACTIVE
from rlmflow.graph.config import HARNESSES

pytestmark = [pytest.mark.live, pytest.mark.timeout(1800)]

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
RUNS = Path(__file__).parent / "_runs"
NEEDLE_TASK = (
    "There are {n} text files in haystack/. Exactly one line in one file matches "
    "`The magic number is <number>`. Find and return the number. Split the files "
    "into batches and give each batch to a worker."
)


def claude_env() -> dict[str, str]:
    """The environment for ``claude``: its own sign-in, and this Python's ``rlmflow`` first on PATH."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    return env


def example(script: str, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        cwd=REPO,
        env=claude_env(),
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout


def needle_project(tmp_path: Path, num_files: int, seed: int) -> tuple[Path, str]:
    sys.path.insert(0, str(EXAMPLES / "needle"))
    from filesystem import generate_haystack

    random.seed(seed)
    project = tmp_path / "project"
    (project / "haystack").mkdir(parents=True)
    answer = generate_haystack(project / "haystack", num_files=num_files)
    (project / "rlmflow.toml").write_text('[run]\nfast_model = "gpt-5-mini"\ntools = "none"\nmax_depth = 1\n')
    return project, answer


CLAUDE_FLOW = """\
from dataclasses import replace
from pathlib import Path

from rlmflow import AgentConfig
from rlmflow.cli import resolve_options
from rlmflow.cli.run import build_flow


def make_flow():
    project = Path(__file__).parent
    options, _ = resolve_options({}, cwd=project)
    flow = build_flow(replace(options, flow=None), workdir=project)
    flow.add_config(AgentConfig(name="claude", harness="claude_code", max_depth=1))
    return flow
"""


def plugin_session(project: Path, prompt: str, *extra: str) -> dict:
    """``claude -p`` in ``project`` with the rlmflow plugin; returns its JSON result."""
    argv = ["claude", "-p", prompt, "--plugin-dir", str(PLUGIN_DIR), "--output-format", "json"]
    argv += ["--allowedTools", "Bash(rlmflow:*)", "Read", "Grep", "Glob", *extra]
    done = subprocess.run(argv, cwd=project, env=claude_env(), capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return json.loads(done.stdout)


def saved_run(project: Path, name: str) -> Path:
    [graph_dir] = list((project / ".rlmflow" / "runs").iterdir())
    target = RUNS / name
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(graph_dir, target)
    return graph_dir


def test_claude_harness_roots_find_the_needles():
    out = example(
        "needle/haystack.py", "--harness", "claude_code", "--num-lines", "50000", "--no-viz",
        "--out-dir", str(RUNS / "mode2-haystack"),
    )
    assert "Correct:        True" in out
    out = example(
        "needle/filesystem.py", "--harness", "claude_code", "--num-files", "100", "--no-viz",
        "--out-dir", str(RUNS / "mode2-filesystem"),
    )
    assert "Correct:        True" in out
    g = Graph.load(RUNS / "mode2-filesystem")
    [root] = [a for a in g.agents() if g.launcher(a) is None]
    assert g.children(root) and root.config.harness == HARNESSES["claude_code"]


def test_fork_halfway_through_a_claude_harness_run():
    out = example("needle/fork_halfway.py", "--num-files", "100", "--out-dir", str(RUNS / "mode2-fork-halfway"))
    assert "correct:          True" in out and "new launches: 0" in out


def test_your_claude_session_as_the_root_finds_the_needle_and_forks(tmp_path):
    project, answer = needle_project(tmp_path, num_files=100, seed=3)
    (project / "claude_flow.py").write_text(CLAUDE_FLOW)
    with (project / "rlmflow.toml").open("a") as toml:
        toml.write('flow = "claude_flow:make_flow"\n')
    result = plugin_session(project, "/rlmflow:run " + NEEDLE_TASK.format(n=100))
    assert answer in result["result"]
    assert not (project / ".rlmflow" / "serve.json").exists()

    graph_dir = saved_run(project, "mode4-filesystem")
    g = Graph.load(graph_dir)
    [root] = g.agents()[:1]
    assert root.config.harness == INTERACTIVE and root.name == "root"
    assert g.children(root), "the root should have launched workers"
    assert answer in str(g.result(root))
    turn_ends = [n for n in g.history(root) if n.type in ("waiting", "done") and n.data.get("session_id")]
    assert {n.data["session_id"] for n in turn_ends} == {result["session_id"]}

    launching = next(n for n in g.history(root) if n.type == CODE and n.launched())
    forked = subprocess.run(
        [sys.executable, "-m", "rlmflow", "fork", str(graph_dir), launching.id,
         "--config", "claude", "--name", "fork", "--project", str(project)],
        cwd=project, env=claude_env(), capture_output=True, text=True, timeout=1500,
    )
    assert forked.returncode == 0, forked.stdout[-3000:] + forked.stderr[-3000:]
    g = Graph.load(graph_dir)
    [fork] = [a for a in g.agents() if a.name == "fork"]
    assert answer in str(g.result(fork))
    assert g.children(fork) == [], "the fork should reuse the workers, not relaunch"
    shutil.rmtree(RUNS / "mode4-fork-halfway", ignore_errors=True)
    shutil.copytree(graph_dir, RUNS / "mode4-fork-halfway")


def test_claude_code_children_of_a_repl_root():
    out = example(
        "providers/claude_code_delegation.py",
        "--workdir", str(RUNS / "delegation-workspace"),
        "--out-dir", str(RUNS / "delegation"),
    )
    assert "Result:" in out
    g = Graph.load(RUNS / "delegation")
    claude_children = [a for a in g.agents() if a.config.harness == HARNESSES["claude_code"]]
    assert len(claude_children) == 2 and all(g.terminal(a) for a in claude_children)
