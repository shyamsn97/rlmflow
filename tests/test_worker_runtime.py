import asyncio
import sys
import threading
from pathlib import Path

import pytest

from rlmflow import AgentStart, ExecAction, Flow, LocalRuntime, WorkerRepl, WorkerSession
from rlmflow.runtime import DoneSignal, ReplStatus
from rlmflow.runtime.connections import PopenConnection
from rlmflow.runtime.repl import MISSING_REPL_NOTE
from rlmflow.tools import tool


def make_repl(
    path: Path,
    *,
    session: WorkerSession | None = None,
    execution_timeout: float | None = None,
) -> WorkerRepl:
    if session is None:
        connection = PopenConnection(
            [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
            cwd=path,
        )
        session = WorkerSession(
            connection,
            timeout=10,
            execution_timeout=execution_timeout,
        )
    return WorkerRepl(session)


@pytest.fixture
def repl(tmp_path: Path):
    instance = make_repl(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def run(repl: WorkerRepl, code: str):
    return asyncio.run(repl.run(code))


def test_worker_preserves_state_and_supports_top_level_await(repl):
    repl.seed({}, {"question": "hello"})
    first = run(
        repl,
        "import asyncio\n"
        "await asyncio.sleep(0)\n"
        "x = 40\n"
        "ENV['seen'] = INPUTS['question']\n"
        "print(x + 2)",
    )
    second = run(repl, "print(x)")

    assert first.status is ReplStatus.OK
    assert first.output == "42"
    assert second.output == "40"
    assert repl.get_var("x") == 40
    assert repl.get_env_var("seen") == "hello"


def test_context_mappings_support_standard_dict_operations(repl):
    repl.seed({}, {"first": "one", "second": "two"})
    result = run(
        repl,
        "print(sorted(INPUTS.keys()))\n"
        "print(INPUTS.get('first'))\n"
        "print(sorted(INPUTS.items()))\n"
        "assert INPUTS.copy() == {'first': 'one', 'second': 'two'}\n"
        "ENV.update({'copied': INPUTS['second']})",
    )

    assert result.status is ReplStatus.OK
    assert result.output.splitlines() == [
        "['first', 'second']",
        "one",
        "[('first', 'one'), ('second', 'two')]",
    ]
    assert repl.get_env_var("copied") == "two"


def test_worker_execution_is_unbounded_by_default(tmp_path):
    instance = make_repl(tmp_path)
    try:
        instance.seed({}, {})
        instance.session.ensure_started()
        instance.session.timeout = 0.05
        result = run(instance, "import asyncio\nawait asyncio.sleep(0.15)\nprint('done')")
        assert result.status is ReplStatus.OK
        assert result.output == "done"
    finally:
        instance.close()


def test_worker_execution_timeout_can_be_configured(tmp_path):
    instance = make_repl(tmp_path, execution_timeout=0.05)
    try:
        instance.seed({}, {})
        with pytest.raises(TimeoutError, match="REPL execution exceeded 0.05s"):
            run(instance, "import asyncio\nawait asyncio.sleep(0.15)")
        assert instance.session.closed
    finally:
        instance.close()


def test_cancellation_during_worker_startup_closes_session(tmp_path, monkeypatch):
    instance = make_repl(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    start = instance.session.connection.start

    def delayed_start():
        entered.set()
        assert release.wait(timeout=5)
        start()

    monkeypatch.setattr(instance.session.connection, "start", delayed_start)

    async def cancel_during_startup():
        task = asyncio.create_task(instance.run("print('unreachable')"))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(cancel_during_startup())
        assert instance.session.closed
    finally:
        release.set()
        instance.close()


def test_worker_reports_errors_and_missing_blocks(repl):
    repl.seed({}, {})
    error = run(repl, "1 / 0")
    missing = run(repl, "   ")

    assert error.status is ReplStatus.ERROR
    assert error.output == "ZeroDivisionError: division by zero"
    assert missing.status is ReplStatus.ERROR
    assert missing.output == MISSING_REPL_NOTE
    assert "one fenced ```repl``` block" in missing.output


def test_worker_calls_async_host_tools(repl):
    @tool("Double a value.", proxy=True)
    async def double(value):
        await asyncio.sleep(0)
        return value * 2

    repl.seed({"double": double}, {})
    result = run(repl, "print(await double(21))")
    assert result.status is ReplStatus.OK
    assert result.output == "42"


def test_worker_completion_returns_answer(repl):
    def finish(answer):
        raise DoneSignal(answer)

    repl.seed({"finish": finish, "done": finish}, {})
    result = run(repl, "finish({'value': 42})")
    assert result.status is ReplStatus.DONE
    assert result.answer == {"value": 42}


def test_worker_runtime_integrates_with_flow():
    class ScriptedLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return "```python\nx = 40\nprint(x + 2)\n```"
            return "```python\nfinish(x)\n```"

    flow = Flow(ScriptedLLM(), runtime=LocalRuntime())
    try:
        assert flow.run("count") == "40"
    finally:
        asyncio.run(flow.aclose())


def test_runtime_delegates_without_sharing_workers():
    class ScriptedLLM:
        def chat(self, messages):
            last = messages[-1]["content"]
            if "child answer" in last:
                return "```python\nfinish('parent answer')\n```"
            if "child task" in last:
                return "```python\nfinish('child answer')\n```"
            return (
                "```python\n"
                "child = await launch_subagent('child task', name='child')\n"
                "answer = await child.wait_for_result()\n"
                "print(answer)\n"
                "```"
            )

    runtime = LocalRuntime(repl_timeout=10)
    flow = Flow(ScriptedLLM(), runtime=runtime)
    try:
        root = flow.start("delegate", max_depth=1)
        assert flow.run(root) == "parent answer"
        # The child kept its own worker, on its own session.
        child = root.sub_agents[0]
        assert sorted(runtime.repls) == sorted([root.id, child.id])
        assert runtime.repls[child.id].session is not runtime.repls[root.id].session
    finally:
        asyncio.run(flow.aclose())


def test_workers_share_heap_but_keep_agent_bindings_separate(tmp_path):
    first = make_repl(tmp_path)
    second = make_repl(tmp_path, session=first.session)
    first.seed({}, {"agent": "first"})
    second.seed({}, {"agent": "second"})
    first.update_env({"owner": "first"})
    second.update_env({"owner": "second"})

    async def exercise():
        await first.run("shared = []")
        return await asyncio.gather(
            first.run("shared.append(INPUTS['agent']); ENV['seen'] = len(shared)"),
            second.run("shared.append(INPUTS['agent']); ENV['seen'] = len(shared)"),
        )

    try:
        left, right = asyncio.run(exercise())
        assert left.status is ReplStatus.OK
        assert right.status is ReplStatus.OK
        assert sorted(first.get_var("shared")) == ["first", "second"]
        assert first.get_env_var("owner") == "first"
        assert second.get_env_var("owner") == "second"
        assert {first.get_env_var("seen"), second.get_env_var("seen")} == {1, 2}
    finally:
        first.close()
        second.close()


def test_reuse_repl_places_child_in_parent_worker(tmp_path):
    class ScriptedLLM:
        def chat(self, messages):
            last = messages[-1]["content"]
            if "child task" in last:
                return "```python\nshared.append('child')\nfinish('child answer')\n```"
            if "child answer" in last:
                return "```python\nfinish(shared)\n```"
            return (
                "```python\n"
                "shared = ['parent']\n"
                "child = await launch_subagent("
                "'child task', name='child', reuse_repl=True)\n"
                "print(await child.wait_for_result())\n"
                "```"
            )

    runtime = LocalRuntime(repl_timeout=10, execution_timeout=17)
    flow = Flow(ScriptedLLM(), runtime=runtime)
    root = flow.start("delegate", max_depth=1)
    try:
        assert flow.run(root) == "['parent', 'child']"
        parent_repl = runtime.repls[root.id]
        assert parent_repl.session.execution_timeout == 17
        # Same worker, separate tenant: the child's REPL rides the parent's session.
        child_repl = runtime.repls[root.sub_agents[0].id]
        assert child_repl is not parent_repl
        assert child_repl.session is parent_repl.session
        orders = [
            node.repl_execution_order for node in root.walk() if isinstance(node, ExecAction)
        ]
        assert orders == [1, 2, 3]
        loaded = AgentStart.load(root.save(tmp_path / "run"))
        assert [
            node.repl_execution_order for node in loaded.walk() if isinstance(node, ExecAction)
        ] == orders
    finally:
        asyncio.run(flow.aclose())
