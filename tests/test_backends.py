import asyncio
import json
from pathlib import Path

import pytest
from helpers import TestRuntime, agent_graph, repl, run_result, stepping

from rlmflow import (
    AgentConfig,
    BackendRun,
    ClaudeCodeBackend,
    DoneOutput,
    Flow,
    Graph,
    Input,
    VerificationFailure,
    current_step,
    ops,
    reads,
)


class FakeBackend:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def run(self, agent, action, *, resume):
        self.calls.append((agent.config.name, action.run_id, resume))
        return self.results.pop(0)


class BarrierBackend:
    def __init__(self):
        self.calls = []
        self.ready = asyncio.Event()

    async def run(self, agent, action, *, resume):
        self.calls.append(agent.content)
        if len(self.calls) == 2:
            self.ready.set()
        await self.ready.wait()
        return agent.content


class DependencyBackend:
    async def run(self, agent, action, *, resume):
        if agent.content == "source":
            return 7
        return reads.inputs(current_step().graph, agent)["value"] + 1


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)

    def chat(self, _messages):
        return self.replies.pop(0)


def test_flow_configures_external_children_without_changing_launch_api():
    backend = FakeBackend("child result")
    llm = ScriptedLLM(
        repl("child = launch_subagent(task='external task')\nprint(await child)"),
        repl("submit('parent done')"),
    )
    g, root = agent_graph("delegate", max_depth=1, child_config={"backend": "fake"})

    flow = Flow(llm, runtime=TestRuntime(), backends={"fake": backend})
    result = run_result(flow, g, root)

    assert result == "parent done"
    [child] = reads.children(g, root)
    assert child.config.backend == "fake"
    assert backend.calls[0][0] == "child0"
    assert reads.path(g, child) == "root.child0"
    assert any(isinstance(node, BackendRun) for node in reads.history(g, child))
    assert reads.result(g, child) == "child result"


def test_each_agent_runs_on_the_backend_its_config_names():
    first, second = FakeBackend("from first"), FakeBackend("from second")
    g = Graph()
    a = g.create_agent("a", config=AgentConfig(name="a", backend="first"))
    b = g.create_agent("b", config=AgentConfig(name="b", backend="second"))

    flow = Flow(object(), backends={"first": first, "second": second})
    flow.run(g)

    assert reads.result(g, a) == "from first"
    assert reads.result(g, b) == "from second"


def test_backend_names_are_checked():
    with pytest.raises(ValueError, match="cannot reuse"):
        AgentConfig(backend="fake", reuse_worker=True)
    with pytest.raises(ValueError, match="child_config cannot set"):
        AgentConfig(child_config={"name": "x"})
    with pytest.raises(ValueError, match="other than 'rlmflow'"):
        Flow(object(), backends={"rlmflow": FakeBackend()})
    g, root = agent_graph("external", config=AgentConfig(backend="fake"))
    done = asyncio.run(Flow(object()).step(g, root))
    assert "requires backend 'fake'" in done.content


def test_external_candidate_uses_schema_repair_path():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    backend = FakeBackend({"wrong": 1}, {"answer": 42})
    g, root = agent_graph("external", config=AgentConfig(backend="fake", output_schema=schema))
    flow = Flow(object(), backends={"fake": backend})

    first_action, rejected, repair, second_action, done = (
        asyncio.run(flow.step(g, root)) for _ in range(5)
    )

    assert isinstance(rejected, VerificationFailure)
    assert isinstance(done, DoneOutput)
    assert done.result == {"answer": 42}
    assert [resume for _path, _run_id, resume in backend.calls] == [False, False]


def test_external_candidate_uses_launch_verifier():
    backend = FakeBackend("bad", "good")
    config = AgentConfig(
        backend="fake",
        verifier_id="verify",
        verifier_hash="fingerprint",
    )
    g, root = agent_graph("external", config=config)
    flow = Flow(object(), backends={"fake": backend})
    flow._verifiers["verify"] = lambda value: None if value == "good" else "not good"

    first_action, rejected, repair, second_action, done = (
        asyncio.run(flow.step(g, root)) for _ in range(5)
    )

    assert isinstance(rejected, VerificationFailure)
    assert rejected.kind == "verifier"
    assert isinstance(done, DoneOutput)
    assert done.result == "good"
    assert done.verified


def test_external_children_run_concurrently():
    backend = BarrierBackend()
    llm = ScriptedLLM(
        repl(
            "a = launch_subagent(task='A')\n"
            "b = launch_subagent(task='B')\n"
            "print(await asyncio.gather(a, b))"
        ),
        repl("submit('done')"),
    )

    flow = Flow(llm, runtime=TestRuntime(), backends={"barrier": backend})
    result = run_result(
        flow, *agent_graph("delegate", max_depth=1, child_config={"backend": "barrier"})
    )

    assert result == "done"
    assert backend.calls == ["A", "B"]


def test_external_child_waits_for_handle_valued_inputs():
    llm = ScriptedLLM(
        repl(
            "source = launch_subagent(task='source')\n"
            "consumer = launch_subagent(task='consumer', value=source)\n"
            "print(await consumer)"
        ),
        repl("submit('done')"),
    )
    g, root = agent_graph("delegate", max_depth=1, child_config={"backend": "dependency"})

    flow = Flow(llm, runtime=TestRuntime(), backends={"dependency": DependencyBackend()})
    result = run_result(flow, g, root)

    assert result == "done"
    source, consumer = reads.children(g, root)
    assert reads.result(g, source) == 7
    assert reads.result(g, consumer) == 8


def test_started_external_action_resumes_after_persistence(tmp_path):
    backend = FakeBackend("done")
    g, root = agent_graph("external", config=AgentConfig(backend="fake"))
    action = ops.append(g, root, BackendRun(backend="fake", started=True))
    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")

    flow = Flow(object(), backends={"fake": backend})
    done = asyncio.run(flow.step(loaded, loaded.node(root.id)))

    assert isinstance(done, DoneOutput)
    assert backend.calls == [("root", action.run_id, True)]


def test_terminal_external_agent_is_not_run_again_after_load(tmp_path):
    backend = FakeBackend()
    g, root = agent_graph("external", config=AgentConfig(backend="fake"))
    ops.append(g, root, DoneOutput(result="saved"))
    g.save(tmp_path / "run")
    loaded = Graph.load(tmp_path / "run")

    flow = Flow(object(), backends={"fake": backend})
    assert run_result(flow, loaded, loaded.node(root.id)) == "saved"
    assert backend.calls == []


async def _run(backend, g, agent, action, *, resume):
    with stepping(g, agent, action):
        return await backend.run(agent, action, resume=resume)


def _claude_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys, time\n"
        "with open(os.environ['CLAUDE_ARGS'], 'w') as handle:\n"
        "    json.dump(sys.argv[1:], handle)\n"
        f"{body}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_claude_code_backend_builds_headless_structured_invocation(tmp_path, monkeypatch):
    args_path = tmp_path / "args.json"
    monkeypatch.setenv("CLAUDE_ARGS", str(args_path))
    script = _claude_script(
        tmp_path,
        "print(json.dumps({'result': 'ignored', 'structured_output': {'answer': 7}}))",
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    g, root = agent_graph(
        "solve it",
        args={"data": Input([1, 2], description="Two integers")},
        config=AgentConfig(backend="claude-code", output_schema=schema),
    )
    action = ops.append(g, root, BackendRun(backend="claude-code"))
    backend = ClaudeCodeBackend(
        command=str(script),
        working_directory=tmp_path,
        permission_mode="acceptEdits",
        allowed_tools=("Read", "Edit"),
    )

    result = asyncio.run(_run(backend, g, root, action, resume=False))

    argv = json.loads(args_path.read_text())
    assert result == {"answer": 7}
    assert argv[0] == "-p"
    assert "--output-format" in argv and "json" in argv
    assert argv[argv.index("--session-id") + 1] == action.run_id
    assert "--json-schema" in argv
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    allowed_index = argv.index("--allowedTools")
    assert argv[allowed_index + 1 : allowed_index + 3] == ["Read", "Edit"]
    assert "Two integers" in argv[1]


def test_claude_code_backend_defaults_to_flow_working_directory(tmp_path):
    backend = ClaudeCodeBackend()

    Flow(
        object(),
        runtime=TestRuntime(working_directory=tmp_path),
        backends={"claude-code": backend},
    )

    assert backend.working_directory == tmp_path


def test_claude_code_backend_resumes_and_rejects_bad_output(tmp_path, monkeypatch):
    args_path = tmp_path / "args.json"
    monkeypatch.setenv("CLAUDE_ARGS", str(args_path))
    script = _claude_script(tmp_path, "print('not json')")
    g, root = agent_graph("continue", config=AgentConfig(backend="claude-code"))
    action = ops.append(g, root, BackendRun(backend="claude-code"))
    backend = ClaudeCodeBackend(command=str(script), working_directory=tmp_path)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        asyncio.run(_run(backend, g, root, action, resume=True))

    argv = json.loads(args_path.read_text())
    assert argv[argv.index("--resume") + 1] == action.run_id
    assert "--session-id" not in argv
    assert "Resume this interrupted run" in argv[1]


def test_claude_code_backend_reports_nonzero_exit(tmp_path, monkeypatch):
    args_path = tmp_path / "args.json"
    monkeypatch.setenv("CLAUDE_ARGS", str(args_path))
    script = _claude_script(
        tmp_path,
        "sys.stderr.write('permission denied')\nsys.exit(3)",
    )
    g, root = agent_graph("task", config=AgentConfig(backend="claude-code"))
    action = ops.append(g, root, BackendRun(backend="claude-code"))
    backend = ClaudeCodeBackend(command=str(script), working_directory=tmp_path)

    with pytest.raises(RuntimeError, match="permission denied"):
        asyncio.run(_run(backend, g, root, action, resume=False))


def test_claude_code_backend_terminates_on_cancellation(tmp_path, monkeypatch):
    args_path = tmp_path / "args.json"
    monkeypatch.setenv("CLAUDE_ARGS", str(args_path))
    script = _claude_script(tmp_path, "time.sleep(30)")
    g, root = agent_graph("task", config=AgentConfig(backend="claude-code"))
    action = ops.append(g, root, BackendRun(backend="claude-code"))
    backend = ClaudeCodeBackend(command=str(script), working_directory=tmp_path)

    async def cancel():
        task = asyncio.create_task(_run(backend, g, root, action, resume=False))
        while not args_path.exists():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())


def test_claude_code_backend_reports_missing_executable():
    g, root = agent_graph("task", config=AgentConfig(backend="claude-code"))
    action = ops.append(g, root, BackendRun(backend="claude-code"))
    backend = ClaudeCodeBackend(command="definitely-not-a-real-claude-command")

    with pytest.raises(RuntimeError, match="was not found"):
        asyncio.run(_run(backend, g, root, action, resume=False))
