import asyncio
import threading

import pytest
from helpers import agent_graph, first_user, run_result
from helpers import repl as block

from rlmflow import (
    AgentConfig,
    AppendChild,
    Flow,
    SequentialPool,
    SubprocessRuntime,
    UserQuery,
    ops,
    reads,
)
from rlmflow.prompts import PromptProfile


def fanout(*names):
    calls = "\n".join(f"    launch_subagent(task={name!r})," for name in names)
    return block(
        f"handles = [\n{calls}\n]\nanswers = [await h for h in handles]\nfinish(','.join(answers))"
    )


class BarrierLLM:
    """A blocking client that holds each child until every sibling has called it."""

    def __init__(self, parties):
        self.barrier = threading.Barrier(parties)

    def chat(self, messages):
        if first_user(messages) == "parent":
            return fanout("a", "b")
        self.barrier.wait(timeout=5)
        return block("finish('c')")


def test_blocking_children_get_a_thread_each():
    g, root = agent_graph("parent", max_depth=1)
    # A pool that ran the children in turn would never clear the barrier, and the
    # first child through would break it rather than hang the suite.
    assert run_result(Flow(BarrierLLM(2), workers=2), g, root) == "c,c"


def test_ten_subprocess_children_can_launch_concurrently():
    names = tuple(f"search-{index}" for index in range(10))

    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            return fanout(*names) if query == "parent" else block(f"finish({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        g, root = agent_graph("parent", max_depth=1)
        # Finished children keep their workers, so ask for the cleanup here rather
        # than leaving eleven subprocesses behind.
        assert run_result(flow, g, root, close_repls=True) == ",".join(names)
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
            return block(f"kept = {query!r}\nfinish({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        g, root = agent_graph("parent", max_depth=1)
        assert run_result(flow, g, root) == "a,b"
        children = reads.children(g, root)
        assert [flow.runtime.get_var(child, "kept") for child in children] == ["a", "b"]
        assert sorted(flow.runtime.repls) == sorted([root.id, *(c.id for c in children)])
    finally:
        asyncio.run(flow.aclose())
    assert list(flow.runtime.repls) == []


JOIN_ATTACHED = (
    "handles = [\n"
    "    launch_agent(task=''),\n"
    "    launch_agent(task=''),\n"
    "]\n"
    "answers = [await h for h in handles]\n"
    "finish(','.join(answers))"
)


def test_prebuilt_subtrees_get_a_thread_each():
    g, root = agent_graph("parent", max_depth=1)
    a = ops.spawn(g, "a")
    b = ops.spawn(g, "b")
    action = ops.attach(g, root, a, b, names=["a", "b"])
    action.code = JOIN_ATTACHED

    assert run_result(Flow(BarrierLLM(2), workers=2), g, root) == "c,c"
    assert isinstance(action, AppendChild)
    assert reads.children(g, root) == [a, b]


def test_prebuilt_children_are_not_submitted_twice():
    calls = 0

    def add_board(user_prompt, _flow, _node):
        user_prompt.add_section("turn.board", "current board", order=1050)

    class LLM:
        async def chat(self, messages):
            nonlocal calls
            calls += 1
            assert messages[-1]["content"].endswith("current board")
            await asyncio.sleep(0.05)
            return block("finish('ok')")

    g, root = agent_graph("parent", max_depth=1)
    first = ops.spawn(g, "first", config=AgentConfig(prompt_profile="worker"))
    second = ops.spawn(g, "second", config=AgentConfig(prompt_profile="worker"))
    ops.append(g, first, UserQuery(content="recover first"))
    ops.append(g, second, UserQuery(content="recover second"))
    ops.attach(g, root, first, second, names=["first", "second"]).code = JOIN_ATTACHED
    flow = Flow(
        LLM(),
        prompt_profiles={
            "worker": PromptProfile(user_modifiers=add_board),
        },
    )

    async def run():
        try:
            await asyncio.wait_for(flow.arun(g), timeout=5)
            return reads.result(g, root)
        finally:
            await flow.aclose()

    assert asyncio.run(run()) == "ok,ok"
    assert calls == 2
    for child in reads.children(g, root):
        boards = [
            node
            for node in reads.history(g, child)
            if isinstance(node, UserQuery) and node.content == "current board"
        ]
        assert boards == []


def test_sequential_pool_does_not_hold_its_slot_while_parent_awaits_children():
    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            return block(f"finish({query!r})")

    g, root = agent_graph("parent", max_depth=1)
    assert run_result(Flow(LLM(), pool=SequentialPool()), g, root) == "a,b"


def test_a_failing_child_does_not_cancel_its_sibling():
    class HalfBrokenLLM:
        async def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("good", "bad")
            await asyncio.sleep(0)  # both children are in flight when one raises
            if query == "bad":
                raise RuntimeError("exploded")
            return block("finish('good')")

    g, root = agent_graph("parent", max_depth=1)
    failed = "[child failed: RuntimeError: exploded]"
    assert run_result(Flow(HalfBrokenLLM()), g, root) == f"good,{failed}"
    assert [reads.result(g, child) for child in reads.children(g, root)] == ["good", failed]


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
            return block("finish('unexpected')")

    async def main():
        llm = Probe()
        flow = Flow(llm)
        run = asyncio.create_task(flow.arun(agent_graph("parent", max_depth=1)[0]))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        return llm, flow

    llm, flow = asyncio.run(main())
    assert llm.cancelled == {"a", "b"}
    assert flow.queue is None
