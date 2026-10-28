import asyncio
import threading

import pytest
from helpers import agent_graph, first_user, run_result
from helpers import repl as block

from rlmflow import Flow, SequentialPool, SubprocessRuntime


def fanout(*names):
    calls = "\n".join(f"    launch_agent({name!r})," for name in names)
    return block(
        f"handles = [\n{calls}\n]\nanswers = [await h for h in handles]\nsubmit(','.join(answers))"
    )


JOIN_ATTACHED = block(
    "answers = [await c for c in AGENTS.get_children()]\nsubmit(','.join(answers))"
)


class BarrierLLM:
    """A blocking client that holds each child until every sibling has called it."""

    def __init__(self, parties, parent=None):
        self.barrier = threading.Barrier(parties)
        self.parent = parent or fanout("a", "b")

    def chat(self, messages):
        if first_user(messages) == "parent":
            return self.parent
        self.barrier.wait(timeout=5)
        return block("submit('c')")


def test_blocking_children_get_a_thread_each():
    g, root = agent_graph("parent")
    # A pool that ran the children in turn would never clear the barrier, and the
    # first child through would break it rather than hang the suite.
    assert run_result(Flow(BarrierLLM(2), workers=2), g, root) == "c,c"


def test_ten_subprocess_children_can_launch_concurrently():
    names = tuple(f"search-{index}" for index in range(10))

    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            return fanout(*names) if query == "parent" else block(f"submit({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        g, root = agent_graph("parent")
        assert run_result(flow, g, root) == ",".join(names)
    finally:
        asyncio.run(flow.aclose())
    assert list(flow.runtime.repls) == []


def test_a_finished_child_keeps_its_repl():
    """A child's namespace is the only record of what its code built, so a host
    (or a meta-agent) can still read it after the child has answered."""

    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            return block(f"kept = {query!r}\nsubmit({query!r})")

    flow = Flow(LLM(), runtime=SubprocessRuntime())
    try:
        g, root = agent_graph("parent")
        assert run_result(flow, g, root) == "a,b"
        children = g.children(root)
        assert [flow.runtime.get_var(child, "kept") for child in children] == ["a", "b"]
        assert sorted(flow.runtime.repls) == sorted([root.id, *(c.id for c in children)])
    finally:
        asyncio.run(flow.aclose())
    assert list(flow.runtime.repls) == []


def test_attached_subtrees_get_a_thread_each():
    g, root = agent_graph("parent")
    a = g.spawn("a", name="a")
    b = g.spawn("b", name="b")
    g.attach(root, a, b)

    assert run_result(Flow(BarrierLLM(2, parent=JOIN_ATTACHED), workers=2), g, root) == "c,c"
    assert g.children(root) == [a, b]


def test_attached_children_with_queued_messages_step_once_each():
    calls = 0

    class LLM:
        async def chat(self, messages):
            nonlocal calls
            if first_user(messages) == "parent":
                return JOIN_ATTACHED
            calls += 1
            assert messages[-1]["content"].startswith("recover")
            await asyncio.sleep(0.05)
            return block("submit('ok')")

    g, root = agent_graph("parent")
    first = g.spawn("first", name="first")
    second = g.spawn("second", name="second")
    g.send(first, "recover first")
    g.send(second, "recover second")
    g.attach(root, first, second)
    flow = Flow(LLM())

    async def run():
        try:
            await asyncio.wait_for(flow.arun(g), timeout=5)
            return g.result(root)
        finally:
            await flow.aclose()

    assert asyncio.run(run()) == "ok,ok"
    assert calls == 2


def test_sequential_pool_does_not_hold_its_slot_while_parent_awaits_children():
    class LLM:
        def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("a", "b")
            return block(f"submit({query!r})")

    g, root = agent_graph("parent")
    assert run_result(Flow(LLM(), workers=SequentialPool()), g, root) == "a,b"


def test_a_failing_child_does_not_cancel_its_sibling():
    class HalfBrokenLLM:
        async def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return fanout("good", "bad")
            await asyncio.sleep(0)  # both children are in flight when one raises
            if query == "bad":
                raise RuntimeError("exploded")
            return block("submit('good')")

    g, root = agent_graph("parent")
    failed = "[child failed: RuntimeError: exploded]"
    assert run_result(Flow(HalfBrokenLLM()), g, root) == f"good,{failed}"
    assert [g.result(child) for child in g.children(root)] == ["good", failed]


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
            return block("submit('unexpected')")

    async def main():
        llm = Probe()
        flow = Flow(llm)
        run = asyncio.create_task(flow.arun(agent_graph("parent")[0]))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        return llm, flow

    llm, flow = asyncio.run(main())
    assert llm.cancelled == {"a", "b"}
    assert flow.queue is None
