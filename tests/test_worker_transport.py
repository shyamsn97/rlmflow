import asyncio
import os
import queue
import sys
import threading
import time

import pytest

from rlmflow import ReplStatus, SubprocessRuntime, WorkerRepl, WorkerSession, start
from rlmflow.runtime.connections import PopenConnection
from rlmflow.runtime.protocol import PingRequest, ReplResponse
from rlmflow.runtime.repl_client import ProtocolDesyncError
from rlmflow.tools import tool


def make_repl(tmp_path, *, session=None):
    if session is None:
        session = WorkerSession(
            PopenConnection(
                [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
                cwd=tmp_path,
            ),
            timeout=10,
        )
    return WorkerRepl(session)


class FakeConnection:
    def __init__(self):
        self.incoming = queue.Queue()
        self.sent = []
        self.closed = False

    def start(self):
        pass

    def send(self, message):
        self.sent.append(message)
        if isinstance(message, PingRequest) and message.tenant_id == "__session__":
            self.incoming.put(ReplResponse(id=message.id))

    def recv(self):
        item = self.incoming.get(timeout=5)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self, *, force=False):
        del force
        self.closed = True
        self.incoming.put(RuntimeError("closed"))


def test_concurrent_worker_outputs_route_by_execution(tmp_path):
    first = make_repl(tmp_path)
    second = make_repl(tmp_path, session=first.session)
    first.seed({}, {})
    second.seed({}, {})

    async def run():
        return await asyncio.gather(
            first.run("import asyncio\nawait asyncio.sleep(0.05)\nprint('first')"),
            second.run("import asyncio\nawait asyncio.sleep(0.05)\nprint('second')"),
        )

    try:
        left, right = asyncio.run(run())
        assert left.output == "first"
        assert right.output == "second"
    finally:
        first.close()
        second.close()


def test_concurrent_host_rpc_routes_to_each_tenant(tmp_path):
    calls = []

    @tool("Record an agent value.", proxy=True)
    async def record(value):
        await asyncio.sleep(0.01)
        calls.append(value)
        return value.upper()

    first = make_repl(tmp_path)
    second = make_repl(tmp_path, session=first.session)
    first.seed({"record": record}, {})
    second.seed({"record": record}, {})

    async def run():
        return await asyncio.gather(
            first.run("print(await record('first'))"),
            second.run("print(await record('second'))"),
        )

    try:
        left, right = asyncio.run(run())
        assert left.output == "FIRST"
        assert right.output == "SECOND"
        assert sorted(calls) == ["first", "second"]
    finally:
        first.close()
        second.close()


def test_gather_overlaps_async_host_tools_in_one_tenant(tmp_path):
    barrier = asyncio.Barrier(2)

    @tool("Meet another host call.", proxy=True)
    async def rendezvous(value):
        await barrier.wait()
        return value

    repl = make_repl(tmp_path)
    repl.seed({"rendezvous": rendezvous}, {})

    async def run():
        return await repl.run(
            "import asyncio\n"
            "values = await asyncio.gather("
            "rendezvous('first'), rendezvous('second'))\n"
            "print(','.join(values))"
        )

    try:
        result = asyncio.run(run())
        assert result.status is ReplStatus.OK
        assert result.output == "first,second"
    finally:
        repl.close()


def test_worker_exception_does_not_poison_next_execution(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})

    async def run():
        return await repl.run("raise ValueError('boom')"), await repl.run("print('alive')")

    try:
        failed, recovered = asyncio.run(run())
        assert failed.status is ReplStatus.ERROR
        assert "ValueError: boom" in failed.output
        assert recovered.status is ReplStatus.OK
        assert recovered.output == "alive"
    finally:
        repl.close()


def test_raw_fd_output_cannot_corrupt_protocol(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})

    async def run():
        first = await repl.run("import os\nos.write(1, b'raw-output\\n')")
        second = await repl.run("print('still framed')")
        return first, second

    try:
        first, second = asyncio.run(run())
        assert first.status is ReplStatus.OK
        assert second.output == "still framed"
        assert "raw-output" in repl.session.connection.stderr_tail()
    finally:
        repl.close()


def test_subprocess_output_cannot_corrupt_protocol(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})
    code = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', \"print('child-output')\"], check=True)"
    )
    try:
        assert asyncio.run(repl.run(code)).status is ReplStatus.OK
        assert asyncio.run(repl.run("print('still alive')")).output == "still alive"
    finally:
        repl.close()


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.write(1, b'raw\\n')",
        "import os\nos.system('echo shell-output')",
        (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', \"print('child-output')\"], check=True)"
        ),
    ],
)
def test_native_stdout_preserves_worker_and_state(tmp_path, code):
    repl = make_repl(tmp_path)
    repl.seed({}, {})
    try:
        asyncio.run(repl.run("kept = 'state'"))
        assert asyncio.run(repl.run(code)).status is ReplStatus.OK
        assert asyncio.run(repl.run("print(kept)")).output == "state"
    finally:
        repl.close()


def test_captured_subprocess_output_is_returned_normally(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})
    code = (
        "import subprocess, sys\n"
        "out = subprocess.run([sys.executable, '-c', \"print('captured')\"], "
        "capture_output=True, text=True, check=True).stdout\n"
        "print(out.strip())"
    )
    try:
        assert asyncio.run(repl.run(code)).output == "captured"
    finally:
        repl.close()


def test_native_stderr_flood_does_not_block_worker(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})
    try:
        result = asyncio.run(
            repl.run("import os\nos.write(2, b'x' * 300_000)\nprint('survived')")
        )
        assert result.output == "survived"
    finally:
        repl.close()


def test_user_stdin_cannot_consume_protocol_frames(tmp_path):
    repl = make_repl(tmp_path)
    repl.seed({}, {})
    try:
        result = asyncio.run(repl.run("import sys\nprint(repr(sys.stdin.read()))"))
        assert result.status is ReplStatus.OK
        assert result.output == "''"
        assert asyncio.run(repl.run("print('still framed')")).output == "still framed"
    finally:
        repl.close()


def test_responses_are_routed_by_id_not_arrival_order():
    connection = FakeConnection()
    session = WorkerSession(connection, timeout=5)
    session.ensure_started()
    results = {}

    def request(request_id):
        results[request_id] = session.request(
            PingRequest(id=request_id, tenant_id="tenant"),
            timeout=5,
        ).output

    threads = [
        threading.Thread(target=request, args=("first",)),
        threading.Thread(target=request, args=("second",)),
    ]
    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 5
        while len(connection.sent) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        connection.incoming.put(ReplResponse(id="second", output="two"))
        connection.incoming.put(ReplResponse(id="first", output="one"))
        for thread in threads:
            thread.join(timeout=5)
        assert results == {"first": "one", "second": "two"}
    finally:
        session.close()


def test_unsolicited_response_fails_the_session():
    connection = FakeConnection()
    session = WorkerSession(connection, timeout=5)
    try:
        session.ensure_started()
        connection.incoming.put(ReplResponse(id="unknown"))
        deadline = time.monotonic() + 5
        while session._failure is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert isinstance(session._failure, ProtocolDesyncError)
        with pytest.raises(RuntimeError, match="matches no outstanding request"):
            session.ensure_started()
    finally:
        session.close()


def test_concurrent_sends_are_serialized():
    connection = FakeConnection()
    active = 0
    max_active = 0
    guard = threading.Lock()
    original_send = connection.send

    def slow_send(message):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        original_send(message)
        with guard:
            active -= 1

    connection.send = slow_send
    session = WorkerSession(connection, timeout=5)
    session.ensure_started()

    def request(index):
        session.request(
            PingRequest(id=f"request-{index}", tenant_id="__session__"),
            timeout=5,
        )

    threads = [threading.Thread(target=request, args=(index,)) for index in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert max_active == 1
    finally:
        session.close()


def test_configured_execution_timeout_becomes_dead_outcome(tmp_path):
    runtime = SubprocessRuntime(
        working_directory=tmp_path,
        repl_timeout=5,
        execution_timeout=0.05,
    )
    agent = start("q")
    try:
        result = asyncio.run(runtime.execute(agent, "import time\ntime.sleep(30)"))
        assert result.status is ReplStatus.DEAD
        assert "REPL execution exceeded 0.05s" in result.output
    finally:
        runtime.close()


def test_cancelling_shared_execution_kills_worker_group(tmp_path):
    first = make_repl(tmp_path)
    second = make_repl(tmp_path, session=first.session)
    second.seed({}, {})

    async def cancel():
        started = asyncio.Event()

        @tool("Block until this execution is cancelled.", proxy=True)
        async def wait_forever():
            started.set()
            await asyncio.Event().wait()

        first.seed({"wait_forever": wait_forever}, {})
        task = asyncio.create_task(first.run("await wait_forever()"))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(cancel())
        with pytest.raises(RuntimeError, match="closed"):
            asyncio.run(second.run("print('unreachable')"))
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(not hasattr(os, "listdir"), reason="requires descriptor inspection")
def test_worker_uses_few_host_descriptors(tmp_path):
    before = len(os.listdir("/dev/fd"))
    repl = make_repl(tmp_path)
    try:
        repl.seed({}, {})
        asyncio.run(repl.run("x = 1"))
        assert len(os.listdir("/dev/fd")) - before < 10
    finally:
        repl.close()
