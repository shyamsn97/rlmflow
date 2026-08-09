"""Transport correctness: one owner per stream, and per-agent scoping of state.

Covers the defects in docs/research/repl_plan.md stage 0, plus the refcounted
working directory (D14) and the frontier guard (D13).
"""

import asyncio
import contextvars
import sys
import threading
import time

import pytest

from rlmflow.graph.nodes import (
    AgentBusyError,
    AgentStart,
    ExecAction,
    UserQuery,
    running_step,
)
from rlmflow.runtime import PopenConnection
from rlmflow.runtime.connections import DEFAULT_REPL_TIMEOUT
from rlmflow.runtime.protocol import ProxyCall, ReplResponse, RunRequest
from rlmflow.runtime.repl import LocalRepl, WorkingDirectoryConflict
from rlmflow.runtime.repl_client import ProtocolDesyncError, RemoteRepl
from rlmflow.runtime.runtime import SubprocessRuntime


def agent(name="q"):
    return AgentStart(content=name)


def sandbox(**kwargs):
    return SubprocessRuntime(**kwargs)


# --------------------------------------------------------------------------
# D1/D2 - stderr has an owner, and a silent sandbox cannot hang the host
# --------------------------------------------------------------------------


def test_flooding_stderr_does_not_wedge_the_sandbox():
    """Past the pipe buffer, an undrained stderr would block the run forever."""
    runtime = sandbox()
    a = agent()
    try:
        run = asyncio.run(
            runtime.execute(
                a,
                "import sys\nsys.stderr.write('x' * 300_000)\nsys.stderr.flush()\n"
                "print('survived')",
            )
        )
        assert "survived" in (run.output or "")
    finally:
        runtime.close_repl(a)


def test_stderr_is_captured_for_diagnosis():
    connection = PopenConnection(
        [sys.executable, "-u", "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"],
        label="dying sandbox",
        repl_timeout=5,
    )
    repl = RemoteRepl(connection)
    with pytest.raises(RuntimeError, match="boom"):
        repl.call(RunRequest(id="run-1", code="pass"))
    connection.close()


def test_repl_timeout_defaults_to_finite():
    """``None`` means wait forever, and ``RemoteRepl.run`` is uncancellable."""
    assert DEFAULT_REPL_TIMEOUT is not None
    assert PopenConnection(["true"]).repl_timeout == DEFAULT_REPL_TIMEOUT
    assert sandbox().repl_timeout == DEFAULT_REPL_TIMEOUT


def test_a_silent_sandbox_becomes_a_dead_outcome_not_a_hang():
    runtime = sandbox(repl_timeout=1)
    a = agent()
    try:
        run = asyncio.run(runtime.execute(a, "import time\ntime.sleep(30)"))
        assert run.status == "dead"
        assert "did not respond" in (run.output or "")
    finally:
        runtime.close_repl(a)


# --------------------------------------------------------------------------
# D3 - the wire has exactly one writer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "code"),
    [
        ("os.write", "import os\nos.write(1, b'RAW\\n')"),
        ("os.system", "import os\nos.system('echo SHELLOUT')"),
        (
            "subprocess",
            "import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(\"SUBPROC\")'])",
        ),
    ],
)
def test_writing_to_fd1_does_not_destroy_the_repl(label, code):
    """Any of these used to corrupt the protocol and take the namespace with it."""
    runtime = sandbox()
    a = agent()
    try:
        asyncio.run(runtime.execute(a, "kept = 'state'"))
        run = asyncio.run(runtime.execute(a, code))
        assert run.status != "dead", f"{label} killed the REPL: {run.output}"

        after = asyncio.run(runtime.execute(a, "print('kept is', kept)"))
        assert "kept is state" in (after.output or ""), f"{label} lost the namespace"
    finally:
        runtime.close_repl(a)


def test_escaped_fd1_output_is_reported_to_the_agent():
    runtime = sandbox()
    a = agent()
    try:
        run = asyncio.run(
            runtime.execute(
                a,
                "import subprocess, sys\n"
                "subprocess.run([sys.executable, '-c', 'print(\"FROM CHILD\")'])\n"
                "print('from agent')",
            )
        )
        assert "from agent" in (run.output or "")
        assert "FROM CHILD" in (run.output or "")
    finally:
        runtime.close_repl(a)


def test_captured_subprocess_output_still_works():
    runtime = sandbox()
    a = agent()
    try:
        run = asyncio.run(
            runtime.execute(
                a,
                "import subprocess, sys\n"
                "out = subprocess.run([sys.executable, '-c', 'print(\"CAP\")'],"
                " capture_output=True, text=True).stdout\n"
                "print('got', out.strip())",
            )
        )
        assert "got CAP" in (run.output or "")
    finally:
        runtime.close_repl(a)


# --------------------------------------------------------------------------
# D5/D6 - one reader, routing by id
# --------------------------------------------------------------------------


class FakeConnection:
    """Hands back canned messages, so ordering can be forced."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.lock = threading.Lock()

    def send(self, msg):
        with self.lock:
            self.sent.append(msg)

    def recv(self):
        if not self.script:
            raise RuntimeError("connection exhausted")
        item = self.script.pop(0)
        return item() if callable(item) else item

    def close(self):
        pass


def test_a_response_for_no_outstanding_request_raises():
    repl = RemoteRepl(FakeConnection([ReplResponse(id="run-99", output="not yours")]))
    with pytest.raises(ProtocolDesyncError, match="matches no outstanding request"):
        repl.call(RunRequest(id="run-1", code="pass"))


def test_a_caller_gets_its_own_response_when_they_arrive_out_of_order():
    """Two callers, one connection: neither may receive the other's answer."""
    connection = FakeConnection([])
    repl = RemoteRepl(connection)
    both_sent = threading.Event()

    # Answer the second request first, so ordering alone would cross them.
    def script():
        both_sent.wait(5)
        return [
            ReplResponse(id="run-2", output="two"),
            ReplResponse(id="run-1", output="one"),
        ]

    pending = []

    def recv():
        if not pending:
            pending.extend(script())
        return pending.pop(0)

    connection.recv = recv
    results = {}

    def caller(request_id, key):
        results[key] = repl.call(RunRequest(id=request_id, code="pass")).output

    threads = [
        threading.Thread(target=caller, args=("run-1", "first")),
        threading.Thread(target=caller, args=("run-2", "second")),
    ]
    for t in threads:
        t.start()
    time.sleep(0.2)  # let both register as outstanding
    both_sent.set()
    for t in threads:
        t.join(10)

    assert results == {"first": "one", "second": "two"}


def test_sends_are_serialized():
    """Interleaved bytes from two senders are unparseable, so send holds a lock."""
    connection = FakeConnection([])
    repl = RemoteRepl(connection)
    order = []

    def slow_send(msg):
        order.append(("start", msg.id))
        time.sleep(0.05)
        order.append(("end", msg.id))

    connection.send = slow_send
    threads = [
        threading.Thread(target=repl._send, args=(RunRequest(id=f"run-{i}", code="x"),))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    # Every send completes before the next begins.
    for i in range(0, len(order), 2):
        assert order[i][0] == "start"
        assert order[i + 1] == ("end", order[i][1])


def test_a_proxy_call_is_serviced_while_waiting_for_a_response():
    repl = RemoteRepl(
        FakeConnection(
            [
                ProxyCall(id="proxy-1", proxy="ping", args=["hi"]),
                ReplResponse(id="run-1", output="done"),
            ]
        )
    )
    repl.proxied["ping"] = lambda value: f"pong {value}"

    assert repl.call(RunRequest(id="run-1", code="pass")).output == "done"
    assert repl.connection.sent[-1].value == "pong hi"


# --------------------------------------------------------------------------
# D14 - the working directory is refcounted, not mutually exclusive
# --------------------------------------------------------------------------


def test_a_co_tenant_runs_while_another_is_parked(tmp_path):
    """A mutex held across a run would deadlock delegation."""
    repl = LocalRepl(working_directory=tmp_path)
    parked = threading.Event()
    child_done = threading.Event()
    outcome = {}

    def wait_for_child():
        parked.set()
        outcome["child_finished_in_time"] = child_done.wait(5)

    repl.namespace["wait_for_child"] = wait_for_child

    def parent():
        asyncio.run(repl.run("wait_for_child()"))

    def child():
        parked.wait(5)
        asyncio.run(repl.run("pass"))
        child_done.set()

    threads = [threading.Thread(target=parent), threading.Thread(target=child)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert outcome.get("child_finished_in_time") is True
    assert time.time() - started < 3


def test_the_working_directory_is_applied_and_restored(tmp_path):
    import os

    before = os.getcwd()
    repl = LocalRepl(working_directory=tmp_path)
    run = asyncio.run(repl.run("import os\nprint(os.getcwd())"))

    assert str(tmp_path.resolve()) in run.output
    assert os.getcwd() == before


def test_two_different_working_directories_raise_rather_than_silently_diverge(tmp_path):
    """cwd is per-process, so two live targets cannot both be honoured."""
    first, second = tmp_path / "a", tmp_path / "b"
    # LocalRepl creates its own directory; make both up front for clarity.
    first.mkdir()
    second.mkdir()
    one, two = LocalRepl(working_directory=first), LocalRepl(working_directory=second)
    parked = threading.Event()
    release = threading.Event()
    errors = []

    def hold():
        one.namespace["park"] = lambda: (parked.set(), release.wait(5))
        asyncio.run(one.run("park()"))

    holder = threading.Thread(target=hold)
    holder.start()
    parked.wait(5)
    try:
        run = asyncio.run(two.run("print('should not get here')"))
        errors.append(run.output)
    finally:
        release.set()
        holder.join(5)

    assert any("WorkingDirectoryConflict" in text for text in errors), errors


def test_conflict_is_raised_directly_by_the_scope(tmp_path):
    from rlmflow.runtime.repl import _CWD

    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()

    with _CWD.held(one):
        with pytest.raises(WorkingDirectoryConflict, match="per-process"):
            with _CWD.held(two):
                pass


def test_the_same_directory_nests_without_complaint(tmp_path):
    """Co-tenants share a directory, which is the case that must not raise."""
    import os

    from rlmflow.runtime.repl import _CWD

    with _CWD.held(tmp_path):
        with _CWD.held(tmp_path):
            assert os.getcwd() == str(tmp_path.resolve())
        # Still held by the outer scope, not restored early.
        assert os.getcwd() == str(tmp_path.resolve())


# --------------------------------------------------------------------------
# D13 - an agent with a step in flight cannot be appended to
# --------------------------------------------------------------------------


def elsewhere(fn):
    """Run ``fn`` in a fresh context, the way another task or step would.

    The guard turns on who is appending, so the test has to actually be someone
    else — a fresh ``Context`` sees ``active_step()`` as unset, exactly like a
    concurrent injection would.
    """
    return contextvars.Context().run(fn)


def test_appending_to_a_busy_agent_is_refused():
    a = agent()
    action = a.append(ExecAction(code="x = 1"))

    with running_step(action):
        with pytest.raises(AgentBusyError, match="still in flight"):
            elsewhere(lambda: a.frontier.append(UserQuery(content="injected")))


def test_the_busy_error_names_the_agent_and_the_step():
    a = agent()
    action = a.append(ExecAction(code="x = 1"))

    with running_step(action):
        with pytest.raises(AgentBusyError) as caught:
            elsewhere(lambda: a.frontier.append(UserQuery(content="injected")))

    assert "root" in str(caught.value)
    assert action.id in str(caught.value)


def test_a_different_agents_step_does_not_grant_access():
    """Being inside *some* step is not the same as being inside this agent's."""
    a, b = agent(), agent()
    mine = a.append(ExecAction(code="x = 1"))
    theirs = b.append(ExecAction(code="y = 2"))

    with running_step(mine):
        with running_step(theirs):
            with pytest.raises(AgentBusyError):
                a.frontier.append(UserQuery(content="from another agent's step"))


def test_an_agents_own_step_may_extend_its_transcript():
    """``prepare_turn`` appends from inside the step; that must keep working."""
    a = agent()
    action = a.append(ExecAction(code="x = 1"))

    with running_step(action):
        appended = a.frontier.append(UserQuery(content="a nudge from my own step"))

    assert a.frontier is appended


def test_busy_clears_when_the_step_finishes():
    a = agent()
    action = a.append(ExecAction(code="x = 1"))

    with running_step(action):
        assert a.in_flight is action

    assert a.in_flight is None
    assert a.frontier.append(UserQuery(content="now fine")) is a.frontier


def test_the_frontier_error_names_what_it_found():
    a = agent()
    stale = a.append(ExecAction(code="x = 1"))
    a.frontier.append(UserQuery(content="moves the frontier"))

    with pytest.raises(ValueError, match="not the frontier"):
        stale.append(UserQuery(content="too late"))
