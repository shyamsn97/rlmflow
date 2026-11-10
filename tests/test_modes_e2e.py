"""The same scripted Claude session as a harness (rlmflow drives Claude) and with the
plugin (Claude drives rlmflow), end to end through real hooks, CLIs and HTTP."""

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import ScriptedFlow, StubLLM, code_nodes, own
from helpers import reasoning_repl as block

from rlmflow import AgentConfig, Flow, Graph
from rlmflow.agent_tools import hooks
from rlmflow.cli.fork import resume_command
from rlmflow.cli.serve import INTERACTIVE, Serve, until

pytestmark = pytest.mark.timeout(120)

REPO = Path(__file__).resolve().parents[1]
FAKE_CLAUDE = Path(__file__).parent / "fixtures" / "fake_claude.py"
WORKER = block("submit(chunk.count('x'))")

COUNT = (
    "Count the x's",
    [
        "rlmflow repl \"doc = 'xxaaxbxx'\nchunks = [doc[:4], doc[4:]]\nnotes = 'split doc in two'\nprint(len(doc))\"",
        "rlmflow repl \"handles = [launch_agent('count', config='worker', chunk=c) for c in chunks]\"",
    ],
    "Launched the counters. {outputs}",
)
SUM = (
    "Agents you started have finished",
    ["rlmflow repl 'submit(sum(h.result() for h in handles))'"],
    "Done.",
)
FROM_NOTES = (
    "You are continuing an earlier session",
    ["rlmflow repl 'submit(notes + \"!\")'"],
    "From notes.",
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """``claude`` (the stand-in) and ``rlmflow`` on PATH; returns ``(script(entries), calls())``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_CLAUDE} "$@"\n')
    (bin_dir / "rlmflow").write_text(
        f'#!/bin/sh\nPYTHONPATH={REPO} exec {sys.executable} -m rlmflow "$@"\n'
    )
    for shim in bin_dir.iterdir():
        shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-offline")
    for name in ("RLMFLOW_URL", "RLMFLOW_TOKEN", "RLMFLOW_AGENT"):
        monkeypatch.delenv(name, raising=False)
    log = tmp_path / "claude.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))

    def script(*entries):
        monkeypatch.setenv("FAKE_CLAUDE_SCRIPT", json.dumps([list(e) for e in entries]))

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return script, calls


def worker_flow() -> Flow:
    return ScriptedFlow(
        StubLLM(lambda messages: WORKER),
        configs=[AgentConfig(name="worker", description="Counts x's in one chunk.")],
    )


def shapes(g):
    """Each agent's node shapes and result, by path."""
    return {
        a.path(): ([(n.role, n.runs_code, _flags(n)) for n in own(a)], a.final_result())
        for a in g.agents()
    }


def _flags(node):
    return sorted(flag for flag in ("terminal", "idle", "seen") if flag in node.data)


def idle_turn(g, agent):
    return next(n for n in agent.nodes() if n.data.get("idle"))


async def claude(project: Path, *argv: str) -> dict:
    """Your own session: the stand-in with the plugin, in ``project``."""
    process = await asyncio.create_subprocess_exec(
        "claude",
        "--interactive",
        "--plugin-dir",
        str(hooks.PLUGIN_DIR),
        *argv,
        cwd=project,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    assert process.returncode == 0, err.decode()
    return json.loads(out.decode().strip().splitlines()[-1])


async def serve_in_process(project: Path, graph: Graph, **kwargs) -> tuple[Serve, asyncio.Task]:
    serve = Serve(worker_flow(), graph, graph_dir=project / "run", project=project, **kwargs)
    task = asyncio.create_task(serve.run())
    assert await until(lambda: serve.serve_file.exists() or None, 5)
    return serve, task


def interactive_run(project: Path) -> tuple[Graph, str]:
    """``/rlmflow:run`` the counting task against an in-process serve; the graph and session id."""

    async def main():
        serve, task = await serve_in_process(project, Graph())
        session = await claude(project, "/rlmflow:run Count the x's")
        await asyncio.wait_for(task, 10)
        return serve.graph, session["session_id"]

    return asyncio.run(main())


def test_same_conversation_same_graph_in_both_modes(env, tmp_path, monkeypatch):
    script, calls = env
    script(SUM, COUNT)

    harness_project = tmp_path / "harness"
    harness_project.mkdir()
    headless = Graph()
    root = headless.start("Count the x's")

    async def as_harness():
        flow = worker_flow()
        try:
            await flow.arun(
                headless,
                root_config=AgentConfig(
                    harness="claude_code", harness_options={"command": "claude"}
                ),
            )
        finally:
            await flow.aclose()

    monkeypatch.chdir(harness_project)
    asyncio.run(as_harness())

    plugin_project = tmp_path / "plugin"
    plugin_project.mkdir()
    interactive, session_id = interactive_run(plugin_project)

    assert shapes(headless) == shapes(interactive)
    assert root.final_result() == 5
    [iroot] = [a for a in interactive.agents() if a.name == "root"]
    assert iroot.config.harness == INTERACTIVE
    idle = idle_turn(interactive, iroot)
    assert idle.data["session_id"] == session_id and idle.text.startswith(
        "Launched the counters. 8"
    )
    assert interactive.usage([iroot]).input_tokens == 20
    harness_calls = [argv for argv in calls() if "-p" in argv]
    assert len(harness_calls) == 2 and "--resume" in harness_calls[1]


def test_forking_an_interactive_graph_resumes_at_turn_ends_and_restarts_mid_turn(env, tmp_path):
    script, calls = env
    script(SUM, COUNT)
    project = tmp_path / "proj"
    project.mkdir()
    g, session_id = interactive_run(project)
    [root] = [a for a in g.agents() if a.name == "root"]
    idle = idle_turn(g, root)
    first_code = code_nodes(g, root)[0]

    # The copy's children already delivered their results, so its resumed turn opens on the prompt alone.
    script(FROM_NOTES, SUM, ("Continue the task", SUM[1], SUM[2]))
    at_turn_end = g.fork(idle)
    command = resume_command(at_turn_end, at_turn_end.agent(root.id), hooks.PLUGIN_DIR)
    assert f"--resume {session_id} --fork-session" in command

    async def continue_fork(branch, argv):
        serve, task = await serve_in_process(project, branch, root=branch.agent(root.id))
        await claude(project, *argv)
        await asyncio.wait_for(task, 10)

    asyncio.run(continue_fork(at_turn_end, shlex.split(command)[1:]))
    assert at_turn_end.agent(root.id).final_result() == 5
    resumed = calls()[-1]
    assert resumed[resumed.index("--resume") + 1] == session_id and "--fork-session" in resumed

    mid_turn = g.fork(first_code)
    command = resume_command(mid_turn, mid_turn.agent(root.id), hooks.PLUGIN_DIR)
    assert "--resume" not in command
    asyncio.run(continue_fork(mid_turn, shlex.split(command)[1:]))
    assert mid_turn.agent(root.id).final_result() == "split doc in two!"
    assert mid_turn.agent(root.id).children() == []
    assert len(root.children()) == 2


def run_cli(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["rlmflow", *argv], cwd=cwd, capture_output=True, text=True, timeout=60)


def test_rlmflow_run_starts_serve_and_a_restarted_serve_continues_the_session(env, tmp_path):
    script, _ = env
    script(
        ("what are your notes", ["rlmflow repl 'submit(notes)'"], "ok"),
        ("remember this", ["rlmflow repl \"notes = 'kept'\"", "rlmflow repl 'submit(1)'"], "noted"),
    )
    project = tmp_path / "proj"
    project.mkdir()

    session = asyncio.run(claude(project, "/rlmflow:run remember this"))["session_id"]
    assert until_gone(project / ".rlmflow" / "serve.json")
    [graph_dir] = list((project / ".rlmflow" / "runs").iterdir())
    g = Graph.load(graph_dir)
    [root] = g.agents()
    assert root.config.harness == INTERACTIVE and root.final_result() == 1

    hooks.start_serve(project, ["--graph", str(graph_dir)])
    asyncio.run(claude(project, "--resume", session, "what are your notes?"))
    assert until_gone(project / ".rlmflow" / "serve.json")
    g = Graph.load(graph_dir)
    assert g.agent(root.id).final_result() == "kept"


def until_gone(path: Path, timeout: float = 15) -> bool:
    return asyncio.run(until(lambda: True if not path.exists() else None, timeout)) is True


def test_rlmflow_fork_interactive_prints_the_resume_command_and_serves_the_fork(env, tmp_path):
    script, _ = env
    script(SUM, COUNT)
    project = tmp_path / "proj"
    project.mkdir()
    g, session_id = interactive_run(project)
    [root] = [a for a in g.agents() if a.name == "root"]
    idle = idle_turn(g, root)

    done = run_cli(
        "fork",
        str(project / "run"),
        idle.id,
        "--interactive",
        "--project",
        str(project),
        cwd=project,
    )
    assert done.returncode == 0, done.stderr
    assert (
        f"claude --plugin-dir {hooks.PLUGIN_DIR} --resume {session_id} --fork-session"
        in done.stdout
    )
    serve_file = project / ".rlmflow" / "serve.json"
    info = json.loads(serve_file.read_text())
    assert info["session_id"] is None and info["root"] == root.id
    [out] = [line.split(" into ")[1] for line in done.stdout.splitlines() if " into " in line]
    saved = Graph.load(out)
    assert saved.metadata["forked_from"] == {"graph": g.id, "node": idle.id}
    assert len(Graph.load(project / "run")) == len(g)

    hooks.run("end", {"cwd": str(project)})
    assert until_gone(serve_file)

    headless = run_cli("fork", str(project / "run"), idle.id, cwd=project)
    assert headless.returncode == 1 and "was interactive; pass --config" in headless.stderr
