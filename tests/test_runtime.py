import asyncio
import sys

import pytest
from helpers import StubLLM, agent_graph, first_user, run_result
from helpers import repl as block

from rlmflow import (
    DockerRuntime,
    Flow,
    LocalRuntime,
    ModalRuntime,
    ReplStatus,
    SubprocessRuntime,
    WorkerRepl,
    ops,
)
from rlmflow.graph.nodes import AgentStart, ExecAction
from rlmflow.runtime import DEFAULT_PREIMPORTS, base_namespace, build_docker_argv
from rlmflow.runtime.repl_client import invoke_tool_call
from rlmflow.tools import get_tool_metadata, tool

WRITE_A_NOTE = block(
    "from pathlib import Path\n"
    'Path("note.txt").write_text("hello")\n'
    'finish(Path("note.txt").read_text())'
)


def test_tool_restore_metadata_and_saved_calls_are_generic():
    g, root = agent_graph("root")
    action = ops.append(g, root, ExecAction(code=""))
    calls: list[tuple[str, str]] = []

    @tool("First tool.", proxy=True, restore="result")
    def first(value):
        calls.append(("first", value))
        return {"value": value}

    @tool("Second tool.", proxy=True, restore="result")
    def second(value):
        calls.append(("second", value))
        return value.upper()

    @tool("Custom restore.", proxy=True, restore=lambda saved, prefix: (prefix, saved))
    def custom(prefix):
        return prefix.lower()

    assert get_tool_metadata(first).restore == "result"
    assert invoke_tool_call(first, action, 0, False, ("a",), {}) == {"value": "a"}
    assert invoke_tool_call(second, action, 0, False, ("b",), {}) == "B"
    assert invoke_tool_call(first, action, 0, True, ("ignored",), {}) == {"value": "a"}
    assert invoke_tool_call(custom, action, 0, False, ("VALUE",), {}) == "value"
    assert invoke_tool_call(custom, action, 0, True, ("restored",), {}) == (
        "restored",
        "value",
    )
    assert action.saved_tool_calls == {
        "first": {0: {"value": "a"}},
        "second": {0: "B"},
        "custom": {0: "value"},
    }
    assert calls == [("first", "a"), ("second", "b")]

    with pytest.raises(TypeError, match="restore"):
        tool("Invalid.", restore="again")
    with pytest.raises(ValueError, match="proxy=True"):
        tool("Local.", restore="result")


class RecordingWorkerSession:
    def __init__(self, *_args, **kwargs):
        self.timeout = kwargs["timeout"]
        self.execution_timeout = kwargs["execution_timeout"]
        self.closed = False
        self.tenants = {}

    def add_tenant(self, tenant):
        self.tenants[tenant.tenant_id] = tenant

    def release(self, tenant):
        self.tenants.pop(tenant.tenant_id, None)

    def close(self, **_kwargs):
        self.closed = True


@pytest.mark.parametrize(
    "runtime",
    [
        LocalRuntime(repl_timeout=19, execution_timeout=17),
        SubprocessRuntime(repl_timeout=19, execution_timeout=17),
    ],
)
def test_process_runtimes_propagate_execution_timeout(runtime):
    repl = runtime.open(AgentStart(content="q"))
    try:
        assert isinstance(repl, WorkerRepl)
        assert repl.session.execution_timeout == 17
        assert repl.session.timeout == 19
    finally:
        repl.close()


@pytest.mark.parametrize(
    "runtime",
    [
        DockerRuntime("rlmflow:test", repl_timeout=19, execution_timeout=17),
        ModalRuntime(repl_timeout=19, execution_timeout=17),
    ],
)
def test_sandbox_runtimes_propagate_execution_timeout(monkeypatch, runtime):
    monkeypatch.setattr("rlmflow.runtime.repl_client.WorkerSession", RecordingWorkerSession)

    repl = runtime.open(AgentStart(content="q"))
    try:
        assert isinstance(repl, WorkerRepl)
        assert repl.session.execution_timeout == 17
        assert repl.session.timeout == 19
    finally:
        repl.close()


def test_runtime_execution_timeout_returns_dead_and_removes_worker():
    runtime = LocalRuntime(execution_timeout=0.05)
    flow = Flow(StubLLM(lambda _messages: ""), runtime=runtime)
    root = AgentStart(content="q")
    try:
        result = asyncio.run(
            runtime.execute(
                root,
                "import asyncio\nawait asyncio.sleep(0.15)",
            )
        )

        assert result.status is ReplStatus.DEAD
        assert isinstance(result.error, TimeoutError)
        assert result.output == "REPL execution failed: TimeoutError: REPL execution exceeded 0.05s"
        assert runtime.get(root) is None
    finally:
        asyncio.run(flow.aclose())


def test_local_runtime_uses_worker_and_working_directory(tmp_path):
    flow = Flow(
        StubLLM(lambda _messages: WRITE_A_NOTE),
        runtime=LocalRuntime(working_directory=tmp_path),
    )
    g, root = agent_graph("q")
    try:
        assert run_result(flow, g, root) == "hello"
        assert isinstance(flow.runtime.repl_for(root), WorkerRepl)
        assert (tmp_path / "note.txt").read_text() == "hello"
    finally:
        asyncio.run(flow.aclose())


def test_subprocess_runtime_uses_selected_python_and_working_directory(tmp_path):
    flow = Flow(
        StubLLM(lambda _messages: WRITE_A_NOTE),
        runtime=SubprocessRuntime(
            working_directory=tmp_path,
            python=sys.executable,
        ),
    )
    try:
        assert flow.run("q") == "hello"
        assert (tmp_path / "note.txt").read_text() == "hello"
    finally:
        asyncio.run(flow.aclose())


def test_subprocess_runtime_exposes_opt_in_agents_tree():
    code = block(
        'finish(AGENTS.get().path + "|" + str(len(AGENTS.get_siblings())) + "|" '
        "+ AGENTS.render_graph())"
    )
    flow = Flow(
        StubLLM(lambda _messages: code),
        runtime=SubprocessRuntime(),
        use_agent_tree=True,
    )
    try:
        assert flow.run("q") == "root|0|root [running] (you)"
    finally:
        asyncio.run(flow.aclose())


def test_subprocess_runtime_supports_subagent_handles():
    def reply(messages):
        if first_user(messages) == "parent":
            return block('child = launch_subagent(task="child")\nfinish(await child)')
        return block('finish("child-done")')

    flow = Flow(StubLLM(reply), runtime=SubprocessRuntime())
    try:
        assert run_result(flow, *agent_graph("parent", max_depth=1)) == "child-done"
    finally:
        asyncio.run(flow.aclose())


def test_worker_env_channel_round_trips():
    code = block('ENV["solved"] = ENV["RLMFLOW_IS_ROOT"] == "1"\nfinish(ENV["RLMFLOW_AGENT_ID"])')
    flow = Flow(StubLLM(lambda _messages: code))
    g, root = agent_graph("q")
    try:
        assert run_result(flow, g, root) == "root"
        published = flow.runtime.repl_for(root).env
        assert published["solved"] is True
        assert published["RLMFLOW_AGENT_ID"] == "root"
    finally:
        asyncio.run(flow.aclose())


def test_worker_executes_await_in_comprehensions(tmp_path):
    runtime = LocalRuntime(working_directory=tmp_path)
    repl = runtime.repl_for(AgentStart(content="q"))

    async def double(value):
        await asyncio.sleep(0)
        return value * 2

    async def run():
        repl.seed({"double": double}, {})
        return await repl.run(
            "by_key = {str(i): await double(i) for i in range(3)}\n"
            "values = [await double(i) for i in range(3)]"
        )

    try:
        result = asyncio.run(run())
        assert result.status is ReplStatus.OK
        assert repl.get_var("by_key") == {"0": 0, "1": 2, "2": 4}
        assert repl.get_var("values") == [0, 2, 4]
    finally:
        runtime.close()


def test_empty_iterator_reports_stopiteration_not_coroutine_wrapper():
    runtime = LocalRuntime()
    repl = runtime.repl_for(AgentStart(content="q"))

    async def run():
        repl.seed({}, {})
        return await repl.run("print(next(iter(())))")

    try:
        result = asyncio.run(run())
        assert result.status is ReplStatus.ERROR
        assert result.output == "StopIteration"
        assert "coroutine raised StopIteration" not in result.output
    finally:
        runtime.close()


def test_get_var_reads_a_variable_out_of_the_worker():
    flow = Flow(StubLLM(lambda _messages: block('result = {"n": 42}\nfinish("ok")')))
    g, root = agent_graph("q")
    try:
        assert run_result(flow, g, root) == "ok"
        assert flow.runtime.get_var(root, "result") == {"n": 42}
    finally:
        asyncio.run(flow.aclose())


def test_worker_inject_ships_a_live_object_by_value(tmp_path):
    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self, amount=1):
            self.n += amount

    runtime = LocalRuntime(working_directory=tmp_path)
    repl = runtime.repl_for(AgentStart(content="q"))
    counter = Counter()

    async def run():
        repl.seed({}, {})
        repl.inject("counter", counter)
        await repl.run("counter.bump(6)\ncount = counter.n")

    try:
        asyncio.run(run())
        assert repl.get_var("count") == 6
        with pytest.raises(KeyError, match="valid JSON"):
            repl.get_var("counter")
        assert counter.n == 0
    finally:
        runtime.close()


def test_worker_preimports_core_modules(tmp_path):
    # Models reach for these without importing; a NameError costs a turn to recover.
    runtime = LocalRuntime(working_directory=tmp_path)
    repl = runtime.repl_for(AgentStart(content="q"))

    async def run():
        repl.seed({}, {})
        return await repl.run(
            "sizes = json.loads('{\"a\": 1}')\n"
            "digits = re.findall(r'\\d+', 'a1b22')\n"
            "here = os.path.basename(os.getcwd())\n"
            "rounded = math.floor(2.7)\n"
            "rows = list(csv.reader(['x,y']))\n"
            "counted = collections.Counter('aab')['a']"
        )

    try:
        assert asyncio.run(run()).status is ReplStatus.OK
        assert repl.get_var("sizes") == {"a": 1}
        assert repl.get_var("digits") == ["1", "22"]
        assert repl.get_var("here") == tmp_path.name
        assert repl.get_var("rounded") == 2
        assert repl.get_var("rows") == [["x", "y"]]
        assert repl.get_var("counted") == 2
    finally:
        runtime.close()


def test_preimports_are_configurable_per_runtime(tmp_path):
    runtime = LocalRuntime(working_directory=tmp_path, preimports=["json"])
    repl = runtime.repl_for(AgentStart(content="q"))

    async def run(code):
        repl.seed({}, {})
        return await repl.run(code)

    try:
        assert "--preimport" in runtime.worker_command()
        assert asyncio.run(run("kept = json.dumps([1])")).status is ReplStatus.OK
        assert repl.get_var("kept") == "[1]"

        dropped = asyncio.run(run("re.findall('a', 'a')"))
        assert dropped.status is ReplStatus.ERROR
        assert "NameError" in dropped.output
    finally:
        runtime.close()


def test_preimports_default_to_absent_flag_and_empty_binds_nothing():
    assert "--preimport" not in LocalRuntime().worker_command()
    assert SubprocessRuntime(preimports=()).worker_command()[-2:] == ["--preimport", ""]
    assert base_namespace(()) == {"__builtins__": __builtins__}
    assert set(base_namespace()) == {"__builtins__", *DEFAULT_PREIMPORTS}
    with pytest.raises(ModuleNotFoundError):
        base_namespace(["definitely_not_a_real_module"])


def test_docker_argv_carries_configured_preimports(tmp_path):
    argv = build_docker_argv("rlmflow:minimal", preimports=["json", "re"])

    assert argv[-2:] == ["--preimport", "json,re"]
    assert "rlmflow.runtime.repl_server" in argv


def test_docker_argv_runs_worker(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
    )

    assert argv[-4:] == ["python", "-u", "-m", "rlmflow.runtime.repl_server"]
    assert argv[:4] == ["docker", "run", "-i", "--rm"]
    assert not any(":50001" in part for part in argv)
    assert ["--network", "none"] == argv[argv.index("--network") : argv.index("--network") + 2]
    assert ["--user", "1000:1000"] == argv[argv.index("--user") : argv.index("--user") + 2]
    assert "--read-only" in argv
    assert ["--cap-drop", "ALL"] == argv[argv.index("--cap-drop") : argv.index("--cap-drop") + 2]
