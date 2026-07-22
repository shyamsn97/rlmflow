"""Concurrency model: the worker pool bounds blocking leaf calls, agent
scheduling is unbounded async, and deep delegation never deadlocks.

Two kinds of LLM client exercise the two paths:

- ``SyncBlockingLLM`` — a blocking ``chat`` that runs on the pool's threads, so
  ``workers`` caps how many run at once.
- ``AsyncBlockingLLM`` — an ``async`` ``chat`` that stays on the event loop and
  bypasses the pool entirely, so ``workers`` does not bound it.
"""

import asyncio
import threading
import time

from rlmflow import Flow, Graph

from helpers import first_user


class SyncBlockingLLM:
    """Blocking client; records the peak number of concurrent ``chat`` threads."""

    def __init__(self, reply):
        self.reply = reply
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def chat(self, messages, **_kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.02)
            return self.reply(messages)
        finally:
            with self._lock:
                self.active -= 1


class AsyncBlockingLLM:
    """Async client; records the peak number of concurrent ``chat`` coroutines."""

    def __init__(self, reply):
        self.reply = reply
        self.active = 0
        self.peak = 0

    async def chat(self, messages, **_kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return self.reply(messages)
        finally:
            self.active -= 1


def _delegate(names, query, finish):
    specs = ", ".join(f"{{'name': {n!r}, 'query': {query!r}}}" for n in names)
    return (
        "```repl\n"
        f"results = await launch_subagents([{specs}])\n"
        f"done({finish})\n"
        "```"
    )


def _fanout_reply(child_names):
    def reply(messages):
        if first_user(messages) == "parent":
            return _delegate(child_names, "child", "','.join(results)")
        return "```repl\ndone('c')\n```"

    return reply


def test_children_execute_concurrently():
    llm = SyncBlockingLLM(_fanout_reply(["a", "b"]))
    flow = Flow(llm, max_depth=1, workers=4)
    graph = Graph(query="parent")

    assert asyncio.run(flow.arun(graph=graph)) == "c,c"
    # Both children reached their (blocking) LLM turn on separate threads.
    assert llm.peak >= 2


def test_workers_bounds_concurrent_blocking_calls():
    llm = SyncBlockingLLM(_fanout_reply(["a", "b", "c", "d"]))
    flow = Flow(llm, max_depth=1, workers=2)
    graph = Graph(query="parent")

    assert asyncio.run(flow.arun(graph=graph)) == "c,c,c,c"
    # Four children, but the pool only lets two blocking calls run at a time.
    assert llm.peak == 2


def test_async_client_is_not_bounded_by_workers():
    llm = AsyncBlockingLLM(_fanout_reply(["a", "b", "c", "d"]))
    # workers=1 would serialize *blocking* calls, but an async client bypasses the
    # pool, so all four children run their turns concurrently on the loop.
    flow = Flow(llm, max_depth=1, workers=1)
    graph = Graph(query="parent")

    assert asyncio.run(flow.arun(graph=graph)) == "c,c,c,c"
    assert llm.peak == 4


def test_deep_delegation_does_not_deadlock_under_bounded_pool():
    def reply(messages):
        task = first_user(messages)
        if task == "root":
            return _delegate(["m0", "m1"], "mid", "'|'.join(results)")
        if task == "mid":
            return _delegate(["l0", "l1"], "leaf", "'+'.join(results)")
        return "```repl\ndone('leaf')\n```"

    # A three-deep tree (root -> 2 mids -> 4 leaves = 7 agents) under a pool that
    # only allows 2 concurrent blocking calls. A parent's turn ends before its
    # children run, so no thread is held across delegation and nothing starves.
    llm = SyncBlockingLLM(reply)
    flow = Flow(llm, max_depth=3, workers=2)
    graph = Graph(query="root")

    result = asyncio.run(asyncio.wait_for(flow.arun(graph=graph), timeout=10))

    assert result == "leaf+leaf|leaf+leaf"
    assert llm.peak <= 2
    assert flow.runs == {}


def test_per_repl_env_is_isolated_across_concurrent_local_agents():
    # The parent reads its own RLMFLOW_AGENT_ID *after* awaiting delegation. With a
    # process-global env this would be clobbered by a concurrent child; the
    # per-REPL env keeps each agent's metadata isolated in one process.
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                "results = await launch_subagents(["
                "{'name': 'a', 'query': 'child'}, {'name': 'b', 'query': 'child'}])\n"
                'done(ENV["RLMFLOW_AGENT_ID"] + "::" + ",".join(results))\n'
                "```"
            )
        return '```repl\ndone(ENV["RLMFLOW_AGENT_ID"])\n```'

    flow = Flow(SyncBlockingLLM(reply), max_depth=1, workers=4)

    assert asyncio.run(flow.arun(graph=Graph(query="parent"))) == "root::root.a,root.b"


def test_launch_subagents_runs_children_without_an_active_run():
    # Called directly (no run_streaming loop): the children still run to completion
    # because each self-drives on a throwaway queue.
    def reply(messages):
        task = first_user(messages)
        return f'```repl\ndone("{task} done")\n```'

    flow = Flow(SyncBlockingLLM(reply), max_depth=1)
    graph = Graph(query="parent")
    repl = flow.repl_for(graph)
    launch_subagents = flow.launch_subagents(graph, repl)

    results = asyncio.run(
        launch_subagents(
            [{"name": "a", "query": "task a"}, {"name": "b", "query": "task b"}]
        )
    )

    assert results == ["task a done", "task b done"]
    assert flow.runs == {}
