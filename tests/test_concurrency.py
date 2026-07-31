import asyncio
import threading

import pytest
from helpers import first_user

from rlmflow import Flow, start


class BlockingLLM:
    def __init__(self, reply, *, parties=0):
        self.reply = reply
        self.barrier = threading.Barrier(parties) if parties else None
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def chat(self, messages, **_kwargs):
        query = first_user(messages)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if query == "child" and self.barrier is not None:
                self.barrier.wait(timeout=5)
            return self.reply(messages)
        finally:
            with self.lock:
                self.active -= 1


def fanout_reply(messages):
    if first_user(messages) == "parent":
        return (
            "```repl\n"
            "r = await launch_subagents(["
            "{'name':'a','query':'child'},"
            "{'name':'b','query':'child'}])\n"
            "done(','.join(r))\n"
            "```"
        )
    return "```repl\ndone('c')\n```"


def test_children_execute_in_parallel_through_task_queue():
    llm = BlockingLLM(fanout_reply, parties=2)
    flow = Flow(llm, max_depth=1, workers=2)
    root = start("parent", max_depth=1)

    assert flow.run(root) == "c,c"
    assert llm.peak == 2


def test_child_failure_is_a_value_and_does_not_cancel_sibling():
    async def reply(messages):
        query = first_user(messages)
        if query == "parent":
            return (
                "```repl\n"
                "r = await launch_subagents(["
                "{'name':'good','query':'good'},"
                "{'name':'bad','query':'bad'}])\n"
                "done('|'.join(r))\n"
                "```"
            )
        await asyncio.sleep(0)
        if query == "bad":
            raise RuntimeError("exploded")
        return "```repl\ndone('good')\n```"

    flow = Flow(type("LLM", (), {"chat": staticmethod(reply)})(), max_depth=1)
    root = start("parent", max_depth=1)

    assert flow.run(root) == "good|[child failed: RuntimeError: exploded]"


def test_cancelling_parent_cancels_running_children():
    class Probe:
        def __init__(self):
            self.started = set()
            self.cancelled = set()
            self.ready = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages):
            query = first_user(messages)
            if query == "parent":
                return (
                    "```repl\n"
                    "r = await launch_subagents(["
                    "{'name':'a','query':'a'},"
                    "{'name':'b','query':'b'}])\n"
                    "done(','.join(r))\n"
                    "```"
                )
            self.started.add(query)
            if len(self.started) == 2:
                self.ready.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.add(query)
                raise
            return "```repl\ndone('unexpected')\n```"

    async def main():
        llm = Probe()
        flow = Flow(llm, max_depth=1)
        root = start("parent", max_depth=1)
        task = asyncio.create_task(flow.arun(root))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return llm, flow

    llm, flow = asyncio.run(main())

    assert llm.cancelled == {"a", "b"}
    assert not flow.tasks.tasks
