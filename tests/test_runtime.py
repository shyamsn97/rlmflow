"""Phase 4 — remote REPL backends.

Covers the backend seam without requiring Docker/Modal/E2B:

* JSON (de)serialization of the two control objects that cross the wire;
* ``build_argv`` for the Docker transport;
* :class:`Flow` backend selection (``make_repl`` / ``Runtime.open``);
* the full JSON-over-stdio protocol, end to end, via an in-process loopback
  backend wired to the real :class:`ReplServer` — exercising seeding, host
  proxies, suspension, and resume.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from rflow import Flow, Graph
from rflow.graph import ChildHandle, UserQuery, WaitRequest
from rflow.runtime import (
    DockerRepl,
    RemoteRepl,
    ReplBackend,
    Runtime,
    SubprocessRepl,
    SubprocessRuntime,
    build_argv,
    deserialize,
    serialize,
)
from rflow.runtime.repl_server import ReplServer
from rflow.runtime.context import EngineContext
from rflow.runtime.runtime import parse_response
from rflow.runtime.env import (
    RFLOW_AGENT_ID,
    RFLOW_DEPTH,
    RFLOW_IS_ROOT,
    RFLOW_MAX_DEPTH,
    RFLOW_PARENT_AGENT_ID,
)
from rflow.tools.builtins import make_history
from rflow.tools.filesystem import read_file, write_file

from .fakes.sandbox import FakeE2BSandboxFactory, NoopLLM
from .helpers import ScriptedLLM, StubLLM, first_user_text, run_to_completion

_RFLOW_ENV_KEYS = [
    RFLOW_AGENT_ID,
    RFLOW_DEPTH,
    RFLOW_PARENT_AGENT_ID,
    RFLOW_MAX_DEPTH,
    RFLOW_IS_ROOT,
]


def _clear_rflow_env(monkeypatch):
    for key in _RFLOW_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ── serde ─────────────────────────────────────────────────────────────


def test_child_handle_roundtrip():
    h = ChildHandle("root.kid")
    assert h.to_dict() == {"child_handle": "root.kid"}
    back = ChildHandle.from_dict(h.to_dict())
    assert isinstance(back, ChildHandle) and back.agent_id == "root.kid"


def test_wait_request_roundtrip():
    w = WaitRequest(
        ["a", "b"],
        launch_id="launch-1",
        launch_specs=[{"name": "a", "query": "qa"}],
        launch_names=["a"],
    )
    assert w.to_dict() == {
        "wait_request": ["a", "b"],
        "launch_id": "launch-1",
        "launch_specs": [{"name": "a", "query": "qa"}],
        "launch_names": ["a"],
    }
    back = WaitRequest.from_dict(w.to_dict())
    assert isinstance(back, WaitRequest) and back.agent_ids == ["a", "b"]
    assert back.launch_id == "launch-1"
    assert back.launch_specs == [{"name": "a", "query": "qa"}]
    assert back.launch_names == ["a"]


def test_serialize_recurses_through_containers():
    value = {"handles": [ChildHandle("x"), ChildHandle("y")], "n": 1, "s": "str"}
    wire = serialize(value)
    assert wire == {
        "handles": [{"child_handle": "x"}, {"child_handle": "y"}],
        "n": 1,
        "s": "str",
    }
    # JSON-safe and reversible.
    back = deserialize(json.loads(json.dumps(wire)))
    assert [h.agent_id for h in back["handles"]] == ["x", "y"]


def test_serialize_passes_through_plain_json():
    assert serialize([1, "a", {"k": 2}]) == [1, "a", {"k": 2}]
    assert deserialize({"k": [1, 2]}) == {"k": [1, 2]}


def test_parse_response_suspended_and_done():
    suspended, payload = parse_response(
        {"suspended": True, "agent_ids": ["a"], "pre_output": "hi"}
    )
    assert suspended is True
    req, pre = payload
    assert isinstance(req, WaitRequest) and req.agent_ids == ["a"] and pre == "hi"

    suspended, out = parse_response({"suspended": False, "output": "done"})
    assert suspended is False and out == "done"


def test_local_runtime_exposes_public_agent_env(monkeypatch):
    _clear_rflow_env(monkeypatch)

    reply = (
        "```repl\n"
        "import os\n"
        f"print(os.environ[{RFLOW_AGENT_ID!r}])\n"
        f"print(os.environ[{RFLOW_DEPTH!r}])\n"
        f"print(os.environ[{RFLOW_PARENT_AGENT_ID!r}])\n"
        f"print(os.environ[{RFLOW_MAX_DEPTH!r}])\n"
        f"done(os.environ[{RFLOW_IS_ROOT!r}])\n"
        "```"
    )
    flow = Flow(StubLLM(reply), max_depth=3)
    assert flow.run("q") == "1"
    for key in _RFLOW_ENV_KEYS:
        assert key not in os.environ


# ── local subprocess runtime ──────────────────────────────────────────


def test_subprocess_runtime_exports_are_public():
    import rflow
    from rflow.runtime.local_process import SubprocessRuntime as Direct

    assert rflow.SubprocessRuntime is Direct
    assert SubprocessRuntime is Direct


def test_subprocess_repl_runs_in_working_directory(tmp_path):
    repl = SubprocessRepl(working_directory=tmp_path)
    try:
        suspended, out = repl.start(
            "from pathlib import Path\n"
            "Path('note.txt').write_text('hello')\n"
            "print(Path.cwd().name)\n"
            "print(Path('note.txt').read_text())"
        )
        assert suspended is False and not repl.errored
        assert out.splitlines() == [tmp_path.name, "hello"]
        assert (tmp_path / "note.txt").read_text() == "hello"
    finally:
        repl.close()


def test_subprocess_repl_timeout_closes_hung_code(tmp_path):
    repl = SubprocessRepl(working_directory=tmp_path, repl_timeout=0.2)
    started_at = time.perf_counter()
    with pytest.raises(TimeoutError, match="did not respond"):
        repl.start("import time\ntime.sleep(5)")
    assert time.perf_counter() - started_at < 2
    assert repl.proc is None


def test_flow_records_error_output_when_subprocess_repl_times_out(tmp_path):
    flow = Flow(
        StubLLM("```repl\nimport time\ntime.sleep(5)\n```"),
        runtime=SubprocessRuntime(working_directory=tmp_path, repl_timeout=0.2),
        max_iters=2,
    )
    try:
        graph = flow.start("hang")
        graph = flow.step(graph)
        graph = flow.step(graph)
    finally:
        flow.close()

    latest = graph.current()
    assert latest is not None
    assert latest.type == "error_output"
    assert "TimeoutError" in latest.content


def test_subprocess_repl_closes_when_agent_finishes(tmp_path):
    flow = Flow(
        StubLLM('```repl\nprint("done soon")\ndone("ok")\n```'),
        runtime=SubprocessRuntime(working_directory=tmp_path),
        max_iters=2,
    )
    try:
        graph = run_to_completion(flow, "finish")
        assert graph.result() == "ok"
        assert flow.repls == {}
        assert flow.runtime.repl_env_cache == {}
        assert flow.runtime.repl_inputs_cache == {}
    finally:
        flow.close()


def test_subprocess_seed_exposes_env_inputs_and_file_tools(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    flow = Flow(
        NoopLLM(), runtime=SubprocessRuntime(working_directory=tmp_path), max_depth=2
    )
    flow.runtime.register_tools([read_file, write_file])
    agent = flow.start("q", inputs={"DOC": "hello"})
    repl = flow.repl_for(agent)
    try:
        suspended, out = repl.start(
            "import os\n"
            "write_file('doc.txt', INPUTS['DOC'])\n"
            "print(read_file('doc.txt'))\n"
            f"print(os.environ[{RFLOW_AGENT_ID!r}])\n"
            f"print(os.environ[{RFLOW_DEPTH!r}])\n"
            f"print(os.environ[{RFLOW_IS_ROOT!r}])"
        )
        assert suspended is False and not repl.errored
        assert out.splitlines() == ["hello", "root", "0", "1"]
        assert (tmp_path / "doc.txt").read_text() == "hello"
    finally:
        flow.close()
        _clear_rflow_env(monkeypatch)


class ParallelSleepLLM(ScriptedLLM):
    thread_safe = True


def test_subprocess_runtime_runs_sibling_repl_blocks_concurrently(tmp_path):
    events_path = tmp_path / "events.log"

    def reply_for(messages):
        task = first_user_text(messages).lower()
        if "child a sleep" in task:
            return (
                "```repl\n"
                "from pathlib import Path\n"
                "import time\n"
                "with Path('events.log').open('a') as fh:\n"
                "    fh.write(f'a start {time.time()}\\n')\n"
                "time.sleep(0.8)\n"
                "with Path('events.log').open('a') as fh:\n"
                "    fh.write(f'a finish {time.time()}\\n')\n"
                'done("A")\n'
                "```"
            )
        if "child b sleep" in task:
            return (
                "```repl\n"
                "from pathlib import Path\n"
                "import time\n"
                "with Path('events.log').open('a') as fh:\n"
                "    fh.write(f'b start {time.time()}\\n')\n"
                "time.sleep(0.8)\n"
                "with Path('events.log').open('a') as fh:\n"
                "    fh.write(f'b finish {time.time()}\\n')\n"
                'done("B")\n'
                "```"
            )
        return (
            "```repl\n"
            "results = await launch_subagents([\n"
            '    {"name": "a", "query": "Child A sleep"},\n'
            '    {"name": "b", "query": "Child B sleep"},\n'
            "])\n"
            'done("|".join(results))\n'
            "```"
        )

    flow = Flow(
        ParallelSleepLLM(reply_for),
        runtime=SubprocessRuntime(working_directory=tmp_path),
        max_concurrency=2,
        max_depth=1,
        max_iters=6,
    )
    started_at = time.perf_counter()
    try:
        graph = run_to_completion(flow, "launch sleepers")
    finally:
        flow.close()
    elapsed = time.perf_counter() - started_at

    assert graph.result() == "A|B"
    rows = [line.split() for line in events_path.read_text().splitlines()]
    starts = [float(ts) for _name, event, ts in rows if event == "start"]
    finishes = [float(ts) for _name, event, ts in rows if event == "finish"]
    assert len(starts) == 2 and len(finishes) == 2
    assert max(starts) < min(finishes)
    assert elapsed < 1.8


def test_subprocess_runtime_eager_refills_ready_child_repl_work(tmp_path):
    events_path = tmp_path / "eager-events.log"

    def reply_for(messages):
        task = first_user_text(messages).lower()
        convo = "\n".join(m.get("content", "").lower() for m in messages)
        if "child a slow" in task:
            return (
                "```repl\n"
                "from pathlib import Path\n"
                "import time\n"
                "with Path('eager-events.log').open('a') as fh:\n"
                "    fh.write(f'a start {time.time()}\\n')\n"
                "time.sleep(0.8)\n"
                "with Path('eager-events.log').open('a') as fh:\n"
                "    fh.write(f'a finish {time.time()}\\n')\n"
                'done("A")\n'
                "```"
            )
        if "child b two step" in task:
            if "b marker" not in convo:
                return (
                    "```repl\n"
                    "from pathlib import Path\n"
                    "import time\n"
                    "with Path('eager-events.log').open('a') as fh:\n"
                    "    fh.write(f'b1 {time.time()}\\n')\n"
                    "print('b marker')\n"
                    "```"
                )
            return (
                "```repl\n"
                "from pathlib import Path\n"
                "import time\n"
                "with Path('eager-events.log').open('a') as fh:\n"
                "    fh.write(f'b2 {time.time()}\\n')\n"
                'done("B")\n'
                "```"
            )
        return (
            "```repl\n"
            "results = await launch_subagents([\n"
            '    {"name": "a", "query": "Child A slow"},\n'
            '    {"name": "b", "query": "Child B two step"},\n'
            "])\n"
            'done("|".join(results))\n'
            "```"
        )

    flow = Flow(
        ParallelSleepLLM(reply_for),
        runtime=SubprocessRuntime(working_directory=tmp_path),
        eager_children=True,
        max_concurrency=2,
        max_depth=1,
        max_iters=6,
    )
    try:
        graph = run_to_completion(flow, "launch eager sleepers")
    finally:
        flow.close()

    assert graph.result() == "A|B"
    rows = [line.split() for line in events_path.read_text().splitlines()]
    b2 = next(float(row[1]) for row in rows if row[0] == "b2")
    a_finish = next(float(row[2]) for row in rows if row[:2] == ["a", "finish"])
    assert b2 < a_finish


# ── docker argv ───────────────────────────────────────────────────────


def test_build_argv_minimal():
    argv = build_argv("img")
    assert argv == [
        "docker",
        "run",
        "-i",
        "--rm",
        "img",
        "python",
        "-m",
        "rflow.runtime.repl_server",
    ]


def test_build_argv_full_options():
    argv = build_argv(
        "img",
        mounts={"/host": "/workspace"},
        env={"OPENAI_API_KEY": "x"},
        network="none",
        cpus=2.0,
        memory="512m",
        user="1000",
        workdir="/workspace",
        extra_args=["--gpus", "all"],
        docker_bin="podman",
        entrypoint_argv=["python3", "-m", "rflow.runtime.repl_server"],
    )
    assert argv[0] == "podman"
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--cpus" in argv and argv[argv.index("--cpus") + 1] == "2.0"
    assert "--memory" in argv and argv[argv.index("--memory") + 1] == "512m"
    assert "--user" in argv and argv[argv.index("--user") + 1] == "1000"
    assert "--gpus" in argv and "all" in argv
    assert "-e" in argv and "OPENAI_API_KEY=x" in argv
    assert argv[-3:] == ["python3", "-m", "rflow.runtime.repl_server"]
    # the bind mount resolves the host path to an absolute path
    mount = argv[argv.index("-v") + 1]
    assert mount.endswith(":/workspace") and mount.startswith("/")


def test_docker_repl_is_repl_backend_without_booting():
    repl = DockerRepl("rlmflow:local", network="none")
    assert isinstance(repl, ReplBackend)
    assert repl.proc is None  # constructing does not boot a container
    assert repl.argv[:4] == ["docker", "run", "-i", "--rm"]


# ── Flow backend selection ────────────────────────────────────────────


class _BackendRuntime(Runtime):
    """A :class:`Runtime` that mints whatever ``factory(agent)`` returns.

    The supported way to plug a custom backend in: subclass ``Runtime`` and
    implement ``open`` (here over a callable, for terse tests).
    """

    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def open(self, agent):
        return self._factory(agent)


class _CountingRepl:
    def __init__(self) -> None:
        self.namespace = {}
        self.engine_context = EngineContext()
        self.process_env = {}
        self.errored = False
        self.closed = 0

    def start(self, code: str):
        return False, ""

    def resume(self, send_value: object):
        return False, ""

    def close(self) -> None:
        self.closed += 1


def test_make_repl_defaults_to_in_process():
    from rflow.runtime.repl import REPL

    flow = Flow(StubLLM(), max_depth=1)
    agent = flow.start("q")
    assert isinstance(flow.make_repl(agent), REPL)


def test_runtime_open_is_used_by_make_repl():
    created: list[Graph] = []

    def factory(agent: Graph):
        created.append(agent)
        return _LoopbackRepl()

    flow = Flow(StubLLM(), max_depth=1, runtime=_BackendRuntime(factory))
    agent = flow.start("q")
    repl = flow.make_repl(agent)
    assert created == [agent]
    repl.close()


def test_close_tears_down_backends():
    closed = {"n": 0}

    class _Counting(_LoopbackRepl):
        def close(self):
            closed["n"] += 1
            super().close()

    flow = Flow(StubLLM(), max_depth=1, runtime=_BackendRuntime(lambda agent: _Counting()))
    run_to_completion(flow, "q")
    flow.close()
    assert closed["n"] == 1


def test_set_graph_discards_stale_repl_without_advancing():
    created: list[_CountingRepl] = []

    def factory(agent):
        repl = _CountingRepl()
        created.append(repl)
        return repl

    flow = Flow(StubLLM(), max_depth=1, runtime=_BackendRuntime(factory))
    old = flow.start("old", {"doc": "old"})
    stale = flow.repl_for(old)
    flow.runtime.repl_env_cache["root"] = {"stale": "1"}
    flow.runtime.repl_inputs_cache["root"] = {"query": "old", "doc": "old"}

    incoming = Graph(
        agent_id="root",
        query="new",
        nodes=[UserQuery(content="incoming")],
    )
    before = incoming.to_dict()

    current = flow.set_graph(incoming)

    assert stale.closed == 1
    assert flow.repls == {}
    assert flow.runtime.repl_env_cache == {}
    assert flow.runtime.repl_inputs_cache == {}
    assert current is flow.graph and current is not incoming
    assert incoming.to_dict() == before
    assert current.query == "new"
    assert current.inputs == {}
    assert current.nodes[0].content == "incoming"
    assert [node.type for node in current.nodes] == ["user_query"]


def test_step_returns_live_graph_and_snapshot_is_explicit():
    flow = Flow(StubLLM(), max_depth=0)
    graph = flow.start("q")

    advanced = flow.step(graph)

    assert advanced is flow.graph
    snapshot = advanced.snapshot()
    assert snapshot is not flow.graph
    assert snapshot.to_dict() == flow.graph.to_dict()
    flow.graph.nodes.append(UserQuery(content="mutated live graph"))
    assert snapshot.to_dict() != flow.graph.to_dict()


def test_step_preserves_repl_for_same_graph_history():
    created: list[_CountingRepl] = []

    def factory(agent):
        repl = _CountingRepl()
        created.append(repl)
        return repl

    flow = Flow(StubLLM(), max_depth=1, runtime=_BackendRuntime(factory))
    graph = flow.start("same")
    live = flow.repl_for(graph)

    flow.step(graph.copy(deep=True))

    assert live.closed == 0
    assert flow.repls["root"] is live


# ── loopback backend: full protocol, no container ─────────────────────


class _LoopbackRepl(RemoteRepl):
    """A :class:`RemoteRepl` wired to a real :class:`ReplServer` over OS pipes.

    Exercises the exact JSON-over-stdio protocol a container would, in-process:
    no Docker, no SDKs.
    """

    def __init__(self) -> None:
        super().__init__()
        to_srv_r, to_srv_w = os.pipe()
        from_srv_r, from_srv_w = os.pipe()
        self._w = os.fdopen(to_srv_w, "w")
        self._r = os.fdopen(from_srv_r, "r")
        server = ReplServer(
            protocol_in=os.fdopen(to_srv_r, "r"),
            protocol_out=os.fdopen(from_srv_w, "w"),
        )
        self._thread = threading.Thread(target=server.serve, daemon=True)
        self._thread.start()

    def send(self, msg: dict) -> None:
        self._w.write(json.dumps(msg) + "\n")
        self._w.flush()

    def recv(self) -> dict:
        return json.loads(self._r.readline())

    def close(self) -> None:
        try:
            self._w.close()  # EOF ends the server's serve() loop
        except Exception:
            pass


def _loopback_flow(llm, **kwargs) -> Flow:
    kwargs.setdefault("max_iters", 5)
    return Flow(llm, runtime=_BackendRuntime(lambda agent: _LoopbackRepl()), **kwargs)


def test_loopback_single_agent_done():
    flow = _loopback_flow(StubLLM('```repl\ndone("ok")\n```'), max_depth=1)
    g = run_to_completion(flow, "q")
    assert g.result() == "ok"
    flow.close()


def test_loopback_inputs_injected_into_remote_repl():
    flow = _loopback_flow(StubLLM('```repl\ndone(INPUTS["DOC"])\n```'), max_depth=1)
    g = run_to_completion(flow, "q", {"DOC": "hello-input"})
    assert g.result() == "hello-input"
    flow.close()


def test_loopback_errored_block_is_recorded_not_fatal():
    # Two turns: a NameError (errored), then a valid done — the engine should
    # surface the traceback to the agent and let it recover.
    replies = iter(["```repl\nprint(missing_name)\n```", '```repl\ndone("recovered")\n```'])
    flow = _loopback_flow(ScriptedLLM(lambda _m: next(replies)), max_depth=1)
    g = run_to_completion(flow, "q")
    assert g.result() == "recovered"
    assert any(n.type == "error_output" or "NameError" in getattr(n, "output", "") for n in g.nodes)
    flow.close()


def test_loopback_structured_done_validates_on_host():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    flow = _loopback_flow(StubLLM('```repl\ndone({"x": 7})\n```'), max_depth=1)
    g = flow.start("q", output_schema=schema)
    while not g.finished:
        flow.step()
    assert json.loads(g.result()) == {"x": 7}
    flow.close()


def test_loopback_launch_subagents_delegates_and_resumes():
    def reply_for(messages):
        task = first_user_text(messages)
        if "depth 1" in task:
            return '```repl\ndone("child-answer")\n```'
        return (
            "```repl\n"
            'rs = await launch_subagents([{"name": "kid", "query": "child task"}])\n'
            'done("root saw " + rs[0])\n'
            "```"
        )

    flow = _loopback_flow(ScriptedLLM(reply_for), max_depth=2)
    g = run_to_completion(flow, "root")
    assert g.result() == "root saw child-answer"
    assert "root.kid" in {n_id for n_id in g.children}
    flow.close()


def test_loopback_launch_subagents_parallel_in_order():
    def reply_for(messages):
        task = first_user_text(messages)
        if "task a" in task:
            return '```repl\ndone("A")\n```'
        if "task bb" in task:
            return '```repl\ndone("B")\n```'
        return (
            "```repl\n"
            'rs = await launch_subagents([{"name": "a", "query": "task a"}, '
            '{"name": "b", "query": "task bb"}])\n'
            'done("|".join(rs))\n'
            "```"
        )

    flow = _loopback_flow(ScriptedLLM(reply_for), max_depth=2)
    g = run_to_completion(flow, "root")
    assert g.result() == "A|B"
    flow.close()


# ── E2B / RemoteFileRuntime (fake provider over the file bridge) ───────


def _use_fake_e2b(monkeypatch):
    monkeypatch.setitem(sys.modules, "e2b", SimpleNamespace(Sandbox=FakeE2BSandboxFactory))
    from rflow.runtime.sandbox.e2b import E2BRepl

    return E2BRepl


def test_remote_file_runtime_exports_and_hierarchy():
    from rflow.runtime import RemoteFileRuntime
    from rflow.runtime.sandbox.e2b import E2BRepl
    from rflow.runtime.sandbox.remote import RemoteFileRuntime as Direct

    assert RemoteFileRuntime is Direct
    assert issubclass(E2BRepl, RemoteFileRuntime)
    assert issubclass(E2BRepl, RemoteRepl)


def test_e2b_executes_repl_protocol_over_bridge(monkeypatch, tmp_path):
    E2BRepl = _use_fake_e2b(monkeypatch)
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    try:
        suspended, out = repl.start("print('hello from e2b')")
        assert (suspended, out) == (False, "hello from e2b")
    finally:
        repl.close()


def test_e2b_seed_routes_tools_by_kind(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)
    flow = Flow(NoopLLM(), max_depth=1, include_llm_query=False)
    agent = flow.start("q")
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    flow.seed_agent_context(repl, agent)
    try:
        repl.seed(flow.build_tools(repl.engine_context), {"DOC": "hi"})
        proxied = set(repl.proxied)
        # host-bound callables (proxy=True) are function proxies
        assert {"done", "_rflow_spawn_child", "get_subagent_result"} <= proxied
        assert "llm_query_batched" not in proxied
        # HISTORY is no longer part of the default tool namespace.
        assert not any(name.startswith("HISTORY") for name in proxied)
        # launchers are composed in the sandbox, never proxied
        assert not (proxied & {"launch_subagents"})
        # inputs land in the namespace; public agent metadata lands in os.environ.
        assert repl.start('print(INPUTS["DOC"])') == (False, "hi")
        suspended, out = repl.start(
            "import os\n"
            f"print(os.environ[{RFLOW_AGENT_ID!r}])\n"
            f"print(os.environ[{RFLOW_DEPTH!r}])\n"
            f"print(os.environ[{RFLOW_IS_ROOT!r}])"
        )
        assert suspended is False and out.splitlines() == ["root", "0", "1"]
    finally:
        repl.close()
        flow.close()


def test_e2b_seed_can_opt_into_llm_query_proxy(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)
    flow = Flow(NoopLLM(), max_depth=1, include_llm_query=True)
    agent = flow.start("q")
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    flow.seed_agent_context(repl, agent)
    try:
        repl.seed(flow.build_tools(repl.engine_context), {})
        assert "llm_query_batched" in set(repl.proxied)
    finally:
        repl.close()
        flow.close()
        _clear_rflow_env(monkeypatch)


def test_e2b_done_proxies_back_to_host_over_bridge(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)
    flow = Flow(NoopLLM(), max_depth=1)
    agent = flow.start("q")
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    flow.seed_agent_context(repl, agent)
    try:
        repl.seed(flow.build_tools(repl.engine_context), {})
        repl.engine_context.done_result = None
        suspended, _ = repl.start('done("answer")')
        assert suspended is False and not repl.errored
        # the host-side done() ran (over the wire) and stashed the result
        assert repl.engine_context.done_result == "answer"
    finally:
        repl.close()
        flow.close()
        _clear_rflow_env(monkeypatch)


def test_e2b_history_object_can_be_proxied_when_opted_in(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)
    flow = Flow(NoopLLM(), max_depth=1)
    agent = flow.start("remember this")
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    flow.seed_agent_context(repl, agent)
    try:
        tools = flow.build_tools(repl.engine_context) | {
            "HISTORY": make_history(flow, repl.engine_context)
        }
        repl.seed(tools, {})
        # Opted-in HISTORY forwards each call to the host, which slices the live
        # host graph and ships only the result back over the wire.
        suspended, out = repl.start(
            "print(HISTORY.messages()[0]['role']); print(len(HISTORY.last(5)))"
        )
        assert suspended is False and not repl.errored
        assert out.splitlines() == ["user", "1"]
    finally:
        repl.close()
        flow.close()
        _clear_rflow_env(monkeypatch)


def test_e2b_registered_tool_runs_in_sandbox_not_proxied(monkeypatch, tmp_path):
    from rflow.tools.filesystem import read_file

    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)
    flow = Flow(NoopLLM(), max_depth=1)
    flow.runtime.register_tools([read_file])
    agent = flow.start("q")
    repl = E2BRepl(remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5)
    flow.seed_agent_context(repl, agent)
    note = tmp_path / "note.txt"
    note.write_text("hello\nworld\n")
    try:
        repl.seed(flow.build_tools(repl.engine_context), {})
        # A registered tool is shipped into the sandbox to run there — it is NOT a
        # host proxy (calling it never crosses the wire).
        assert "read_file" not in repl.proxied
        suspended, out = repl.start(f"print(read_file({str(note)!r}))")
        assert suspended is False and not repl.errored
        assert "hello" in out and "world" in out
    finally:
        repl.close()
        flow.close()
        _clear_rflow_env(monkeypatch)


def test_e2b_full_flow_over_bridge(monkeypatch, tmp_path):
    _clear_rflow_env(monkeypatch)
    E2BRepl = _use_fake_e2b(monkeypatch)

    def factory(agent):
        return E2BRepl(
            remote_workdir=str(tmp_path / "remote"), setup_commands=[], repl_timeout=5
        )

    flow = Flow(
        StubLLM('```repl\ndone("ok")\n```'),
        max_depth=1,
        max_iters=5,
        runtime=_BackendRuntime(factory),
    )
    g = run_to_completion(flow, "q")
    assert g.result() == "ok"
    flow.close()
    _clear_rflow_env(monkeypatch)


def test_remote_file_runtime_setup_commands_resolution(monkeypatch, tmp_path):
    E2BRepl = _use_fake_e2b(monkeypatch)
    # None → default pip install; explicit [] → run nothing.
    default = E2BRepl(remote_workdir=str(tmp_path / "a"))
    assert default.setup_commands == list(E2BRepl.DEFAULT_SETUP_COMMANDS)
    explicit = E2BRepl(remote_workdir=str(tmp_path / "b"), setup_commands=[])
    assert explicit.setup_commands == []


def test_remote_file_runtime_close_ignores_failing_exec(tmp_path):
    from rflow.runtime.sandbox.remote import RemoteFileRuntime

    class _GoneRuntime(RemoteFileRuntime):
        def __init__(self):
            super().__init__(remote_workdir="/workspace")
            self._started = True
            self.closed = False

        def exec(self, command, *, timeout=None):
            raise RuntimeError("sandbox already shut down")

        def _close_sandbox(self):
            self.closed = True

    runtime = _GoneRuntime()
    runtime.close()  # must not raise even though the kill command fails
    assert runtime.closed and not runtime._started


# ── Modal (fake sandbox streams) ───────────────────────────────────────


def test_modal_starts_repl_server_as_entrypoint(monkeypatch, tmp_path):
    class FakeStdin:
        def __init__(self):
            self.writes: list[str] = []

        def write(self, data):
            self.writes.append(data)

        def drain(self):
            pass

    class FakeSandbox:
        created_args = None
        created_kwargs = None
        instance = None

        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = iter(())
            self.stderr = iter(())

        @classmethod
        def create(cls, *args, **kwargs):
            cls.created_args = args
            cls.created_kwargs = kwargs
            cls.instance = cls()
            return cls.instance

    class FakeApp:
        @staticmethod
        def lookup(name, create_if_missing=False):
            return {"name": name, "create": create_if_missing}

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(App=FakeApp, Sandbox=FakeSandbox))
    from rflow.runtime.sandbox.modal import ModalRepl

    image = object()
    repl = ModalRepl(app_name="test-app", remote_workdir="/workspace", image=image)
    repl.send({"cmd": "ping"})

    assert FakeSandbox.created_args[:2] == ("sh", "-lc")
    assert "rflow.runtime.repl_server" in FakeSandbox.created_args[2]
    assert "--workdir /workspace" in FakeSandbox.created_args[2]
    assert FakeSandbox.created_kwargs["app"] == {"name": "test-app", "create": True}
    assert FakeSandbox.created_kwargs["image"] is image
    assert FakeSandbox.instance.stdin.writes == ['{"cmd": "ping"}\n']


def test_modal_uses_direct_repl_streams(tmp_path):
    import queue

    from rflow.runtime.sandbox.modal import ModalRepl

    class FakeStdin:
        def __init__(self):
            self.writes: list[str] = []
            self.drained = False

        def write(self, data):
            self.writes.append(data)

        def drain(self):
            self.drained = True

    class FakeSandbox:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = iter(())
            self.stderr = iter(())

    repl = ModalRepl()
    repl.container = FakeSandbox()
    repl._stdout_queue = queue.Queue()
    repl._stdout_queue.put('{"ok": true}')

    repl.send({"cmd": "ping"})
    assert repl.container.stdin.writes == ['{"cmd": "ping"}\n']
    assert repl.container.stdin.drained
    assert repl.recv() == {"ok": True}


def test_modal_recv_reads_stdout_stream_without_background_thread():
    from rflow.runtime.sandbox.modal import ModalRepl

    repl = ModalRepl()
    repl.container = object()
    repl._stdout_iter = iter(['{"ok": true}\n'])

    assert repl.recv() == {"ok": True}


def test_modal_splits_stdout_chunks_into_json_lines():
    import queue

    from rflow.runtime.sandbox.modal import ModalRepl

    repl = ModalRepl()
    repl._stdout_queue = queue.Queue()
    repl._start_reader(
        iter(['{"ok": true}\n{"suspended": false, "output": ""}\n']),
        repl._stdout_queue,
    )
    assert repl._stdout_queue.get(timeout=1) == '{"ok": true}'
    assert repl._stdout_queue.get(timeout=1) == '{"suspended": false, "output": ""}'


def test_modal_stdout_reader_ignores_client_closed_during_close():
    import queue

    from rflow.runtime.sandbox.modal import ModalRepl

    class ClientClosed(Exception):
        pass

    class ClosingStream:
        def __iter__(self):
            return self

        def __next__(self):
            raise ClientClosed("client closed")

    repl = ModalRepl()
    repl._stdout_queue = queue.Queue()
    repl._closing.set()
    repl._start_reader(ClosingStream(), repl._stdout_queue)

    assert repl._stdout_queue.get(timeout=1) is None


def test_modal_requires_optional_dependency(monkeypatch):
    # Simulate `modal` not installed: importing it inside _ensure_sandbox fails.
    monkeypatch.setitem(sys.modules, "modal", None)
    from rflow.runtime.sandbox.modal import ModalRepl

    repl = ModalRepl()
    with pytest.raises(ModuleNotFoundError, match="modal"):
        repl.send({"cmd": "ping"})
