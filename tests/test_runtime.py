import asyncio
import sys

import pytest
from helpers import StubLLM, first_user

from rlmflow import (
    DockerRuntime,
    Flow,
    LocalRuntime,
    ModalRuntime,
    ReplStatus,
    SubprocessRuntime,
    WorkerRepl,
    start,
)
from rlmflow.runtime import build_docker_argv


def block(code):
    return f"```repl\n{code}\n```"


WRITE_A_NOTE = block(
    "from pathlib import Path\n"
    'Path("note.txt").write_text("hello")\n'
    'finish(Path("note.txt").read_text())'
)


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
    repl = runtime.open(start("q"))
    try:
        assert isinstance(repl, WorkerRepl)
        assert repl.session.execution_timeout == 17
        assert repl.session.timeout == 19
    finally:
        repl.close()


@pytest.mark.parametrize(
    ("session_class", "runtime"),
    [
        (
            "rlmflow.runtime.docker.WorkerSession",
            DockerRuntime("rlmflow:test", repl_timeout=19, execution_timeout=17),
        ),
        (
            "rlmflow.runtime.modal.WorkerSession",
            ModalRuntime(repl_timeout=19, execution_timeout=17),
        ),
    ],
)
def test_sandbox_runtimes_propagate_execution_timeout(monkeypatch, session_class, runtime):
    monkeypatch.setattr(session_class, RecordingWorkerSession)

    repl = runtime.open(start("q"))
    try:
        assert isinstance(repl, WorkerRepl)
        assert repl.session.execution_timeout == 17
        assert repl.session.timeout == 19
    finally:
        repl.close()


def test_runtime_execution_timeout_returns_dead_and_removes_worker():
    runtime = LocalRuntime(execution_timeout=0.05)
    flow = Flow(StubLLM(lambda _messages: ""), runtime=runtime)
    root = flow.start("q")
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
    root = start("q")
    try:
        assert flow.run(root) == "hello"
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
        assert flow.run(start("q")) == "hello"
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
        assert flow.run(start("q")) == "root|0|root [running] (you)"
    finally:
        asyncio.run(flow.aclose())


def test_subprocess_runtime_supports_subagent_handles():
    def reply(messages):
        if first_user(messages) == "parent":
            return block(
                "child = await launch_subagent("
                '"child", model="default", name="c")\n'
                "finish(await child.wait_for_result())"
            )
        return block('finish("child-done")')

    flow = Flow(StubLLM(reply), runtime=SubprocessRuntime())
    try:
        assert flow.run(start("parent", max_depth=1)) == "child-done"
    finally:
        asyncio.run(flow.aclose())


def test_worker_env_channel_round_trips():
    code = block('ENV["solved"] = ENV["RLMFLOW_IS_ROOT"] == "1"\nfinish(ENV["RLMFLOW_AGENT_ID"])')
    flow = Flow(StubLLM(lambda _messages: code))
    root = start("q")
    try:
        assert flow.run(root) == "root"
        published = flow.runtime.repl_for(root).env
        assert published["solved"] is True
        assert published["RLMFLOW_AGENT_ID"] == "root"
    finally:
        asyncio.run(flow.aclose())


def test_worker_executes_await_in_comprehensions(tmp_path):
    runtime = LocalRuntime(working_directory=tmp_path)
    repl = runtime.repl_for(start("q"))

    async def double(value):
        await asyncio.sleep(0)
        return value * 2

    async def run():
        repl.seed({"double": double}, {})
        return await repl.run(
            "by_key = {i: await double(i) for i in range(3)}\n"
            "values = [await double(i) for i in range(3)]"
        )

    try:
        result = asyncio.run(run())
        assert result.status is ReplStatus.OK
        assert repl.get_var("by_key") == {0: 0, 1: 2, 2: 4}
        assert repl.get_var("values") == [0, 2, 4]
    finally:
        runtime.close()


def test_get_var_reads_a_variable_out_of_the_worker():
    flow = Flow(StubLLM(lambda _messages: block('result = {"n": 42}\nfinish("ok")')))
    root = start("q")
    try:
        assert flow.run(root) == "ok"
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
    repl = runtime.repl_for(start("q"))
    counter = Counter()

    async def run():
        repl.seed({}, {})
        repl.inject("counter", counter)
        await repl.run("counter.bump(6)")

    try:
        asyncio.run(run())
        got = repl.get_var("counter")
        assert got.n == 6
        assert counter.n == 0
    finally:
        runtime.close()


def test_docker_argv_runs_worker(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
    )

    assert argv[-4:] == ["python", "-u", "-m", "rlmflow.runtime.repl_server"]
    assert argv[:4] == ["docker", "run", "-i", "--rm"]
    assert not any(":50001" in part for part in argv)
