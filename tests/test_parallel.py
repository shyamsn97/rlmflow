import asyncio

from helpers import StubLLM, first_user

from rlmflow import Flow, parallel_run, parallel_stream, start


def test_parallel_stream_merges_events_from_multiple_agents():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    left = start("left")
    right = start("right")

    async def collect():
        return [node async for node in parallel_stream(flow, left, right)]

    events = asyncio.run(collect())

    assert {node.agent.config.id for node in events} == {
        left.config.id,
        right.config.id,
    }
    assert left.result() == right.result() == "ok"


def test_parallel_stream_runs_async_models_concurrently():
    state = {"active": 0, "peak": 0}

    class LLM:
        async def chat(self, _messages):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return '```repl\ndone("ok")\n```'

    flow = Flow(LLM())
    roots = [start(f"q{index}") for index in range(3)]

    async def run():
        async for _ in parallel_stream(flow, *roots):
            pass

    asyncio.run(run())

    assert state["peak"] == 3
    assert all(root.result() == "ok" for root in roots)


def test_parallel_run_preserves_input_order():
    def reply(messages):
        return f'```repl\ndone({first_user(messages)!r})\n```'

    roots = asyncio.run(parallel_run(Flow(StubLLM(reply)), "a", "b", "c"))

    assert [root.result() for root in roots] == ["a", "b", "c"]
