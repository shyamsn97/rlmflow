import ast
import asyncio
import sys

import pytest
from helpers import StubLLM, first_user

from rlmflow import Flow, LocalRuntime, ReplStatus, SubprocessRuntime, start
from rlmflow.runtime import PopenConnection, build_docker_argv
from rlmflow.runtime.protocol import (
    CapabilitiesRequest,
    CapabilityMap,
    PingRequest,
    ProxyCall,
    ReplResponse,
    RunRequest,
    parse_client_message,
    parse_request,
)
from rlmflow.runtime.repl import DoneSignal, LocalRepl, has_top_level_await
from rlmflow.runtime.repl_client import RemoteRepl


def block(code):
    return f"```repl\n{code}\n```"


def remote_repl():
    """A client stub wired to a repl_server running in its own process."""
    return RemoteRepl(
        PopenConnection(
            [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
            label="test remote REPL",
            repl_timeout=5,
        )
    )


WRITE_A_NOTE = block(
    "from pathlib import Path\n"
    'Path("note.txt").write_text("hello")\n'
    'done(Path("note.txt").read_text())'
)


def test_local_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(lambda _messages: WRITE_A_NOTE),
        runtime=LocalRuntime(working_directory=tmp_path),
    )

    assert flow.run(start("q")) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_subprocess_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(lambda _messages: WRITE_A_NOTE),
        runtime=SubprocessRuntime(working_directory=tmp_path),
    )

    assert flow.run(start("q")) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_subprocess_runtime_executes_agent_code():
    flow = Flow(
        StubLLM(lambda _messages: block('print("subproc")\ndone("ok")')),
        runtime=SubprocessRuntime(),
    )
    root = start("q")

    assert flow.run(root) == "ok"
    assert "subproc" in root.frontier.content


def test_subprocess_runtime_supports_awaited_launch_subagents():
    def reply(messages):
        if first_user(messages) == "parent":
            return block(
                'r = await launch_subagents([{"name": "c", "query": "child"}])\ndone(r[0])'
            )
        return block('done("child-done")')

    flow = Flow(StubLLM(reply), runtime=SubprocessRuntime())

    assert flow.run(start("parent", max_depth=1)) == "child-done"


def test_local_repl_env_channel_round_trips():
    # Host seeds the env; agent code reads it via ENV and publishes new state;
    # the host reads that back off ``repl.env`` (distinct from ``namespace``).
    code = block('ENV["solved"] = ENV["RLMFLOW_IS_ROOT"] == "1"\ndone(ENV["RLMFLOW_AGENT_ID"])')
    flow = Flow(StubLLM(lambda _messages: code))
    root = start("q")

    assert flow.run(root) == "root"
    published = flow.runtime.repl_for(root).env
    assert published["solved"] is True
    assert published["RLMFLOW_AGENT_ID"] == "root"


@pytest.mark.parametrize(
    "expression",
    [
        "[await fetch(i) for i in items]",
        "{await fetch(i) for i in items}",
        "{i: await fetch(i) for i in items}",
        "(await fetch(i) for i in items)",
    ],
)
def test_top_level_await_detector_traverses_comprehensions(expression):
    assert has_top_level_await(ast.parse(f"result = {expression}"))


def test_top_level_await_detector_ignores_nested_function_scope():
    tree = ast.parse("async def fetch():\n    return await other()\n")
    assert not has_top_level_await(tree)


def test_local_repl_executes_await_in_comprehensions():
    repl = LocalRepl()

    async def double(value):
        await asyncio.sleep(0)
        return value * 2

    async def run():
        try:
            repl.seed({"double": double}, {})
            return await repl.run(
                "by_key = {i: await double(i) for i in range(3)}\n"
                "values = [await double(i) for i in range(3)]"
            )
        finally:
            repl.close()

    result = asyncio.run(run())
    assert result.status is ReplStatus.OK
    assert repl.get_var("by_key") == {0: 0, 1: 2, 2: 4}
    assert repl.get_var("values") == [0, 2, 4]


def test_subprocess_runtime_exposes_env_metadata():
    code = block('done(ENV["RLMFLOW_AGENT_ID"] + "|" + ENV["RLMFLOW_IS_ROOT"])')
    flow = Flow(StubLLM(lambda _messages: code), runtime=SubprocessRuntime())

    assert flow.run(start("q")) == "root|1"


def test_get_var_reads_a_variable_out_of_the_local_repl():
    flow = Flow(StubLLM(lambda _messages: block('result = {"n": 42}\ndone("ok")')))
    root = start("q")

    assert flow.run(root) == "ok"
    assert flow.runtime.get_var(root, "result") == {"n": 42}


def test_remote_repl_runs_code_in_the_repl_server():
    repl = remote_repl()

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    async def run():
        try:
            repl.seed({"done": done}, {"x": "remote"})
            return await repl.run('print(INPUTS["x"])\ndone("ok")')
        finally:
            repl.close()

    result = asyncio.run(run())

    assert "remote" in result.output
    assert (result.status, result.answer) == (ReplStatus.DONE, "ok")
    assert not repl.errored


def test_remote_repl_reads_back_published_env():
    repl = remote_repl()

    async def run():
        try:
            repl.seed({}, {})
            repl.update_env({"RLMFLOW_AGENT_ID": "root.child"})
            await repl.run('assert ENV["RLMFLOW_AGENT_ID"] == "root.child"\nENV["solved"] = True')
            return dict(repl.env)
        finally:
            repl.close()

    env = asyncio.run(run())

    assert env["solved"] is True
    assert env["RLMFLOW_AGENT_ID"] == "root.child"


def test_remote_inject_ships_a_live_object_by_value():
    pytest.importorskip("cloudpickle")

    # Defined locally so cloudpickle ships the CLASS by value too (the sandbox
    # process can't import this test module).
    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self, k=1):
            self.n += k
            return self.n

    repl = remote_repl()
    counter = Counter()

    async def run():
        try:
            repl.seed({}, {})
            assert repl.capabilities().cloudpickle is True
            repl.inject("counter", counter)
            await repl.run("counter.bump(5)\ncounter.bump()")
            return repl.get_var("counter")
        finally:
            repl.close()

    got = asyncio.run(run())

    assert got.n == 6  # sandbox mutations round-tripped back by value
    assert counter.n == 0  # by value: the host's own object was never touched


def test_remote_inject_refuses_an_unpicklable_value_without_the_capability():
    # A JSON-unsafe value + a sandbox that can't cloudpickle -> a clear error,
    # not a silent JSON coercion failure.
    class Sandboxless(RemoteRepl):
        def __init__(self):
            self.namespace = {}
            self._request_id = 0
            self._capabilities = CapabilityMap(cloudpickle=False)

    class Obj:
        pass

    with pytest.raises(RuntimeError, match="cloudpickle"):
        Sandboxless().inject("x", Obj())


def test_remote_protocol_models_round_trip():
    run = parse_request(RunRequest(id="r1", code="print(1)").model_dump())
    response = parse_client_message(ReplResponse(id="r1", output="1").model_dump())
    proxy = parse_client_message(ProxyCall(id="p1", proxy="done", args=["ok"]).model_dump())

    assert isinstance(run, RunRequest)
    assert run.code == "print(1)"
    assert isinstance(response, ReplResponse)
    assert response.output == "1"
    assert isinstance(proxy, ProxyCall)
    assert proxy.proxy == "done"


def test_repl_server_supports_ping_and_capabilities():
    repl = remote_repl()
    try:
        ping = repl.call(PingRequest(id="ping-test"))
        capabilities = repl.call(CapabilitiesRequest(id="cap-test"))
    finally:
        repl.close()

    assert ping.ok
    assert capabilities.capabilities is not None
    assert not hasattr(capabilities.capabilities, "fork")


def test_docker_argv_runs_the_repl_server(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
        network="none",
    )

    assert "rlmflow.runtime.repl_server" in argv
    assert argv[:4] == ["docker", "run", "-i", "--rm"]
