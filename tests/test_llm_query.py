import asyncio

from helpers import StubLLM

from rlmflow import Flow


def test_llm_query_batched_routes_to_a_named_client():
    fast = StubLLM(lambda messages: messages[-1]["content"].upper())
    flow = Flow(StubLLM(lambda _messages: "unused"), llm_clients={"fast": fast})

    assert asyncio.run(flow.llm_query_batched(["a", "b"], model="fast")) == ["A", "B"]


def test_llm_query_batched_parses_replies_against_an_output_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    flow = Flow(StubLLM(lambda _messages: '{"answer": 3}'))

    answers = asyncio.run(flow.llm_query_batched(["q1", "q2"], output_schema=schema))
    assert answers == [{"answer": 3}, {"answer": 3}]


def test_llm_query_batched_forwards_sampling_kwargs():
    class KwargsLLM:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    llm = KwargsLLM()
    kwargs = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 10, "stop": ["x"]}

    assert asyncio.run(Flow(llm).llm_query_batched(["a"], **kwargs)) == ["ok"]
    assert llm.calls == [kwargs]
