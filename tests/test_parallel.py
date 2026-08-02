import asyncio

from helpers import StubLLM, first_user

from rlmflow import Flow, parallel_run, parallel_stream, start


def block(code):
    return f"```python\n{code}\n```"


def test_parallel_stream_merges_the_nodes_of_every_run():
    flow = Flow(StubLLM(lambda _messages: block("done('ok')")))
    left, right = start("left"), start("right")

    async def collect():
        return [node async for node in parallel_stream(flow, left, right)]

    nodes = asyncio.run(collect())
    assert {id(node.root) for node in nodes} == {id(left), id(right)}
    assert left.result() == right.result() == "ok"


def test_parallel_stream_overlaps_the_runs_it_drives():
    class BarrierLLM:
        """Blocks each caller until every run has asked for a reply."""

        def __init__(self, parties):
            self.barrier = asyncio.Barrier(parties)

        async def chat(self, _messages):
            await self.barrier.wait()
            return block("done('ok')")

    roots = [start(f"q{index}") for index in range(3)]
    flow = Flow(BarrierLLM(len(roots)))

    async def drain():
        async for _node in parallel_stream(flow, *roots):
            pass

    # Runs driven one after another would never clear the barrier, so this would hang.
    asyncio.run(asyncio.wait_for(drain(), timeout=5))
    assert [root.result() for root in roots] == ["ok"] * 3


def test_parallel_stream_uses_one_task_queue():
    class ProbeFlow(Flow):
        def __init__(self):
            super().__init__(StubLLM(lambda _messages: block("done('ok')")))
            self.queues = set()

        async def step(self, node):
            self.queues.add(id(self.queue))
            return await super().step(node)

    flow = ProbeFlow()
    asyncio.run(parallel_run(flow, "a", "b", "c"))
    assert len(flow.queues) == 1


def test_one_parallel_root_can_stop_without_cancelling_another():
    def reply(messages):
        return block(f"done({first_user(messages)!r})")

    flow = Flow(StubLLM(reply))
    short, long = start("short"), start("long")

    async def drain():
        return [
            node
            async for node in parallel_stream(
                flow,
                short,
                long,
                until=lambda _node, root: root is short,
            )
        ]

    asyncio.run(drain())
    assert not short.terminal
    assert long.result() == "long"


def test_parallel_run_returns_the_roots_in_argument_order():
    def reply(messages):
        return block(f"done({first_user(messages)!r})")

    roots = asyncio.run(parallel_run(Flow(StubLLM(reply)), "a", "b", "c"))
    assert [root.result() for root in roots] == ["a", "b", "c"]
