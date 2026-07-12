import asyncio

from rflow import (
    Flow,
    Graph,
)

from helpers import (
    StubLLM,
    UsageLLM,
)


def test_minimal_llm_query_batched_uses_shared_pool():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return messages[-1]["content"].upper()
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_concurrency=2)

    async def collect():
        return await flow.llm_query_batched(["a", "b", "c", "d"])

    assert asyncio.run(collect()) == ["A", "B", "C", "D"]
    assert max_seen == 2



def test_minimal_llm_query_batched_can_be_used_as_opt_in_tool():
    def reply(messages):
        if len(messages) == 1:
            return messages[0]["content"].upper()
        return (
            "```repl\n"
            'results = await llm_query_batched(["a", "b"])\n'
            'done("|".join(results))\n'
            "```"
        )

    flow = Flow(StubLLM(reply), use_llm_query=True)
    graph = Graph(query="q")

    assert flow.run(graph) == "A|B"



def test_minimal_llm_query_batched_routes_model():
    fast = UsageLLM(lambda messages: messages[-1]["content"].upper(), 5, 7)
    flow = Flow(StubLLM(lambda _messages: "unused"), llm_clients={"fast": fast})

    async def collect():
        return await flow.llm_query_batched(["a", "b"], model="fast")

    assert asyncio.run(collect()) == ["A", "B"]



def test_minimal_llm_query_batched_supports_output_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    flow = Flow(StubLLM(lambda _messages: '{"answer": 3}'))

    async def collect():
        return await flow.llm_query_batched(["q1", "q2"], output_schema=schema)

    assert asyncio.run(collect()) == [{"answer": 3}, {"answer": 3}]



def test_minimal_llm_query_batched_forwards_sampling_kwargs():
    class KwargsLLM:
        def __init__(self):
            self.calls = []
            self.last_usage = None

        def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    llm = KwargsLLM()
    flow = Flow(llm)

    async def collect():
        return await flow.llm_query_batched(
            ["a"], temperature=0.5, top_p=0.9, max_tokens=10, stop=["x"]
        )

    assert asyncio.run(collect()) == ["ok"]
    assert llm.calls == [
        {"temperature": 0.5, "top_p": 0.9, "max_tokens": 10, "stop": ["x"]}
    ]

