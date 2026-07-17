import asyncio
import threading
import time

import pytest

from rflow import (
    Flow,
    Graph,
    SequentialPool,
)

from helpers import (
    StubLLM,
    first_user,
)


def test_sequential_pool_runs_blocking_calls_one_at_a_time():
    lock = threading.Lock()
    active = 0
    max_seen = 0

    def reply(messages):
        nonlocal active, max_seen
        task = first_user(messages)
        if task == "parent":
            specs = [{"name": f"c{i}", "query": f"child {i}"} for i in range(3)]
            return (
                f"```repl\nresults = await launch_subagents({specs!r})\n"
                "done(','.join(results))\n```"
            )
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        try:
            time.sleep(0.02)
            return f'```repl\ndone("{task}")\n```'
        finally:
            with lock:
                active -= 1

    flow = Flow(StubLLM(reply), max_depth=1, pool=SequentialPool())
    graph = Graph(query="parent")

    assert flow.run(graph=graph) == "child 0,child 1,child 2"
    assert max_seen == 1


def test_parallel_stream_merges_tagged_events_from_one_flow():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    a = Graph(query="a")
    b = Graph(query="b")

    async def collect():
        return [event async for event in flow.parallel_stream(a, b)]

    events = asyncio.run(collect())
    graph_ids = {event.graph_id for event in events}
    assert graph_ids == {a.graph_id, b.graph_id}
    assert a.result() == "ok"
    assert b.result() == "ok"
    assert flow.runs == {}


def test_parallel_stream_advances_graphs_concurrently():
    state = {"active": 0, "peak": 0}

    class ConcurrencyProbeLLM:
        async def chat(self, _messages):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return '```repl\ndone("ok")\n```'

    flow = Flow(ConcurrencyProbeLLM())

    async def main():
        graphs = [Graph(query=f"q{i}") for i in range(3)]
        async for _event in flow.parallel_stream(*graphs):
            pass
        return graphs

    graphs = asyncio.run(main())
    # All three graphs' turns overlapped on the one loop (async client, no bound).
    assert state["peak"] == 3
    assert all(graph.result() == "ok" for graph in graphs)


def test_parallel_run_returns_graphs_in_argument_order():
    def reply(messages):
        return f'```repl\ndone("{first_user(messages)}")\n```'

    flow = Flow(StubLLM(reply))

    graphs = asyncio.run(flow.parallel_run("qa", "qb", "qc"))

    assert [graph.result() for graph in graphs] == ["qa", "qb", "qc"]
    assert flow.runs == {}


def test_parallel_stream_rejects_duplicate_graph():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.parallel_stream(graph, graph)]

    with pytest.raises(RuntimeError, match="already being streamed"):
        asyncio.run(collect())
