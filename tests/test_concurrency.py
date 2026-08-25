import asyncio
import threading

import pytest
from helpers import first_user

from rlmflow import (
    AppendChild,
    Flow,
    Node,
    Runtime,
    SequentialPool,
    SubprocessRuntime,
    UserQuery,
    start,
)
from rlmflow.prompts import PromptProfile


def block(code):
    return f"```python\n{code}\n```"


def fanout(*names):
    calls = "\n".join(
        f"    await launch_subagent({name!r}, model='default', name={name!r})," for name in names
    )
    return block(
        f"handles = [\n{calls}\n]\n"
        "answers = [await h.wait_for_result() for h in handles]\n"
        "done(','.join(answers))"
    )


class BarrierLLM:
    """A blocking client that holds each child until every sibling has called it."""

    def __init__(self, parties):
        self.barrier = threading.Barrier(parties)

    def chat(self, messages):
        if first_user(messages) == "parent":
            return fanout("a", "b")
        self.barrier.wait(timeout=5)
        return block("done('c')")


def test_blocking_children_get_a_thread_each():
    root = start("parent", max_depth=1)
    # A pool that ran the children in turn would never clear the barrier, and the
    # first child through would break it rather than hang the suite.
    assert Flow(BarrierLLM(2), workers=2).run(root) == "c,c"


def test_ten_subprocess_children_can_launch_concurrently():
    names = tuple(f"search-{index}" for index in range(10))

    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            return fanout(*names) if query == "parent" else block(f"done({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        root = start("parent", max_depth=1)
        # Finished children keep their workers, so ask for the cleanup here rather
        # than leaving eleven subprocesses behind.
        assert flow.run(root, close_repls=True) == ",".join(names)
        assert list(flow.runtime.repls) == []
    finally:
        asyncio.run(flow.aclose())


def test_a_finished_child_keeps_its_repl():
    """A child's namespace is the only record of what its code built, so a host
    (or a meta-agent) can still read it after the child has answered."""

    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            return block(f"kept = {query!r}\ndone({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        root = start("parent", max_depth=1)
        assert flow.run(root) == "a,b"
        children = root.sub_agents
        assert [flow.runtime.get_var(child, "kept") for child in children] == ["a", "b"]
        assert sorted(flow.runtime.repls) == sorted([root.id, *(c.id for c in children)])
    finally:
        asyncio.run(flow.aclose())
    assert list(flow.runtime.repls) == []


def test_prebuilt_subtrees_get_a_thread_each():
    root = start("parent", max_depth=1)
    a = start("a")
    b = start("b")
    root.append_child(a, name="a")
    root.append_child(b, name="b")
    action = root.frontier
    action.code = (
        "handles = [\n"
        "    await launch_subagent('', model='default', name='a'),\n"
        "    await launch_subagent('', model='default', name='b'),\n"
        "]\n"
        "answers = [await h.wait_for_result() for h in handles]\n"
        "done(','.join(answers))"
    )

    assert Flow(BarrierLLM(2), workers=2).run(root) == "c,c"
    assert isinstance(action, AppendChild)
    assert action.child_agents == [a, b]


def test_prebuilt_children_are_not_submitted_twice():
    calls = 0

    def render_board(_runtime: Runtime, node: Node) -> list[dict[str, str]]:
        return [
            *node.render(),
            {"role": "user", "content": "current board"},
        ]

    class LLM:
        async def chat(self, messages):
            nonlocal calls
            calls += 1
            assert messages[-1]["content"].endswith("current board")
            await asyncio.sleep(0.05)
            return block("done('ok')")

    root = start("parent", max_depth=1)
    first = start("first", prompt_profile="worker")
    second = start("second", prompt_profile="worker")
    first.append(UserQuery(content="recover first"))
    second.append(UserQuery(content="recover second"))
    root.append_child(first, name="first")
    root.append_child(second, name="second")
    root.frontier.code = (
        "handles = [\n"
        "    await launch_subagent('', model='default', name='first'),\n"
        "    await launch_subagent('', model='default', name='second'),\n"
        "]\n"
        "answers = [await h.wait_for_result() for h in handles]\n"
        "done(','.join(answers))"
    )
    flow = Flow(
        LLM(),
        prompt_profiles={
            "worker": PromptProfile(render_fn=render_board),
        },
    )

    async def run():
        try:
            return await asyncio.wait_for(flow.arun(root), timeout=5)
        finally:
            await flow.aclose()

    assert asyncio.run(run()) == "ok,ok"
    assert calls == 2
    for child in root.sub_agents:
        boards = [
            node
            for node in child.transcript()
            if isinstance(node, UserQuery) and node.content == "current board"
        ]
        assert boards == []


def test_sequential_pool_does_not_hold_its_slot_while_parent_awaits_children():
    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            return block(f"done({query!r})")

    root = start("parent", max_depth=1)
    assert Flow(LLM(), pool=SequentialPool()).run(root) == "a,b"


def test_a_failing_child_does_not_cancel_its_sibling():
    class HalfBrokenLLM:
        async def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("good", "bad")
            await asyncio.sleep(0)  # both children are in flight when one raises
            if query == "bad":
                raise RuntimeError("exploded")
            return block("done('good')")

    root = start("parent", max_depth=1)
    failed = "[child failed: RuntimeError: exploded]"
    assert Flow(HalfBrokenLLM()).run(root) == f"good,{failed}"
    assert [child.result() for child in root.sub_agents] == ["good", failed]


def test_cancelling_a_run_cancels_the_children_it_launched():
    class Probe:
        """Answers the parent, then parks every child until it is cancelled."""

        def __init__(self):
            self.started = set()
            self.cancelled = set()
            self.ready = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            self.started.add(query)
            if len(self.started) == 2:
                self.ready.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.add(query)
                raise
            return block("done('unexpected')")

    async def main():
        llm = Probe()
        flow = Flow(llm)
        run = asyncio.create_task(flow.arun(start("parent", max_depth=1)))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        return llm, flow

    llm, flow = asyncio.run(main())
    assert llm.cancelled == {"a", "b"}
    assert flow.queue is None
