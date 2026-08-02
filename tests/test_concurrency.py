import asyncio
import threading

import pytest
from helpers import first_user

from rlmflow import Flow, SequentialPool, start


def block(code):
    return f"```python\n{code}\n```"


def fanout(*names):
    specs = ", ".join(f"{{'name': {name!r}, 'query': {name!r}}}" for name in names)
    return block(f"answers = await launch_subagents([{specs}])\ndone(','.join(answers))")


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
