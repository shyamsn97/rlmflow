import asyncio

from rflow import (
    AsyncPool,
    Flow,
    Graph,
    SequentialPool,
    group_flows,
)

from helpers import (
    StubLLM,
    first_user,
)


def test_minimal_async_pool_bounds_child_fanout():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        task = first_user(messages)
        if task == "parent":
            specs = [{"name": f"c{i}", "query": f"child {i}"} for i in range(4)]
            return f"```repl\nresults = await launch_subagents({specs!r})\ndone(','.join(results))\n```"
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return f'```repl\ndone("{task}")\n```'
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_depth=1, max_concurrency=2)
    graph = Graph(query="parent")

    assert flow.run(graph) == "child 0,child 1,child 2,child 3"
    assert max_seen == 2



def test_minimal_sequential_pool_runs_child_fanout_one_at_a_time():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        task = first_user(messages)
        if task == "parent":
            specs = [{"name": f"c{i}", "query": f"child {i}"} for i in range(3)]
            return f"```repl\nresults = await launch_subagents({specs!r})\ndone(','.join(results))\n```"
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return f'```repl\ndone("{task}")\n```'
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_depth=1, pool=SequentialPool())
    graph = Graph(query="parent")

    assert flow.run(graph) == "child 0,child 1,child 2"
    assert max_seen == 1



def test_minimal_async_pool_cancels_sibling_work_on_failure():
    cancelled = False

    async def slow():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def fail():
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def run():
        pool = AsyncPool(max_concurrency=2)
        try:
            await pool.gather(slow(), fail())
        except RuntimeError:
            return
        raise AssertionError("pool.gather should have raised")

    asyncio.run(run())

    assert cancelled



def test_minimal_group_flows_streams_tagged_events_and_returns_graphs():
    def reply(_messages):
        return '```repl\ndone("ok")\n```'

    a_flow = Flow(StubLLM(reply))
    b_flow = Flow(StubLLM(reply))
    a_graph = Graph(query="a")
    b_graph = Graph(query="b")

    group = group_flows(a=(a_flow, a_graph), b=(b_flow, b_graph))

    async def collect():
        events = [event async for event in group]
        return events

    events = asyncio.run(collect())
    graph_ids = {event.graph_id for event in events}
    assert graph_ids == {a_graph.graph_id, b_graph.graph_id}
    assert a_graph.result() == "ok"
    assert b_graph.result() == "ok"



def test_minimal_group_flows_pool_bounds_concurrent_flows():
    state = {"active": 0, "peak": 0}

    class ConcurrencyProbeLLM:
        async def chat(self, _messages):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0)
            state["active"] -= 1
            return '```repl\ndone("ok")\n```'

    entries = {f"f{i}": (Flow(ConcurrencyProbeLLM()), f"q{i}") for i in range(3)}

    async def peak_with(pool):
        state["active"] = state["peak"] = 0
        await group_flows(pool=pool, **entries).run()
        return state["peak"]

    assert asyncio.run(peak_with(SequentialPool())) == 1
    entries = {f"f{i}": (Flow(ConcurrencyProbeLLM()), f"q{i}") for i in range(3)}
    assert asyncio.run(peak_with(AsyncPool(max_concurrency=2))) == 2



def test_minimal_group_flows_run_returns_graphs_by_label():
    flow_a = Flow(StubLLM(lambda _messages: '```repl\ndone("A")\n```'))
    flow_b = Flow(StubLLM(lambda _messages: '```repl\ndone("B")\n```'))

    group = group_flows(a=(flow_a, "qa"), b=(flow_b, "qb"))
    graphs = asyncio.run(group.run())

    assert set(graphs) == {"a", "b"}
    assert graphs["a"].result() == "A"
    assert graphs["b"].result() == "B"

