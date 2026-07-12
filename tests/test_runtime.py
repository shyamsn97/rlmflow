import asyncio
import sys

from rflow import (
    Flow,
    Graph,
    LocalRuntime,
    SubprocessRuntime,
    tool,
)
from rflow.runtime import PopenConnection, build_docker_argv
from rflow.runtime.protocol import (
    CapabilitiesRequest,
    PingRequest,
    ProxyCall,
    ReplResponse,
    RunRequest,
    parse_client_message,
    parse_request,
)
from rflow.runtime.repl import DoneSignal
from rflow.runtime.repl_client import ReplClient

from helpers import (
    StubLLM,
    first_user,
)


def test_minimal_runtime_register_tools_are_prompted_and_executable():
    seen = {}

    @tool("Double a number.")
    def double(x: int) -> int:
        return x * 2

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\nprint(double(3))\ndone("ok")\n```'

    runtime = LocalRuntime()
    runtime.register_tools([double])
    flow = Flow(StubLLM(reply), runtime=runtime)
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    assert "`double" in seen["system"]
    assert "6" in graph.nodes[-1].output



def test_minimal_local_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(
            lambda _messages: (
                "```repl\n"
                "from pathlib import Path\n"
                'Path("note.txt").write_text("hello")\n'
                'done(Path("note.txt").read_text())\n'
                "```"
            )
        ),
        runtime=LocalRuntime(working_directory=tmp_path),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"



def test_minimal_repl_client_uses_minimal_repl_server():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    async def run():
        try:
            repl.seed({"done": done}, {"x": "remote"})
            return await repl.run('print(INPUTS["x"])\ndone("ok")')
        finally:
            repl.close()

    output = asyncio.run(run())

    assert "remote" in output
    assert repl.done_result == "ok"
    assert not repl.errored



def test_minimal_remote_protocol_models_round_trip():
    run = parse_request(RunRequest(id="r1", code="print(1)").model_dump())
    response = parse_client_message(ReplResponse(id="r1", output="1").model_dump())
    proxy = parse_client_message(
        ProxyCall(id="p1", proxy="done", args=["ok"]).model_dump()
    )

    assert isinstance(run, RunRequest)
    assert run.code == "print(1)"
    assert isinstance(response, ReplResponse)
    assert response.output == "1"
    assert isinstance(proxy, ProxyCall)
    assert proxy.proxy == "done"



def test_minimal_repl_server_supports_ping_and_capabilities():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )
    try:
        ping = repl.call(PingRequest(id="ping-test"))
        capabilities = repl.call(CapabilitiesRequest(id="cap-test"))
    finally:
        repl.close()

    assert ping.ok
    assert capabilities.capabilities is not None
    assert capabilities.capabilities.fork == "none"



def test_minimal_subprocess_runtime_executes_agent_code():
    flow = Flow(
        StubLLM(lambda _messages: '```repl\nprint("subproc")\ndone("ok")\n```'),
        runtime=SubprocessRuntime(),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    assert "subproc" in graph.nodes[-1].output



def test_minimal_subprocess_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(
            lambda _messages: (
                "```repl\n"
                "from pathlib import Path\n"
                'Path("note.txt").write_text("hello")\n'
                'done(Path("note.txt").read_text())\n'
                "```"
            )
        ),
        runtime=SubprocessRuntime(working_directory=tmp_path),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"



def test_minimal_docker_argv_uses_minimal_repl_server(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
        network="none",
    )

    assert "rflow.runtime.repl_server" in argv
    assert argv[:4] == ["docker", "run", "-i", "--rm"]



def test_minimal_subprocess_runtime_supports_awaited_launch_subagents():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                'r = await launch_subagents([{"name": "c", "query": "child"}])\n'
                "done(r[0])\n"
                "```"
            )
        return '```repl\ndone("child-done")\n```'

    flow = Flow(StubLLM(reply), runtime=SubprocessRuntime(), max_depth=1)

    assert flow.run(Graph(query="parent")) == "child-done"

