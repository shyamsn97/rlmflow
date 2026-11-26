"""LLM clients, streaming adapters, and ``complete``."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from helpers import ScriptedFlow, StubLLM, repl, turns

from rlmflow import Agent, Flow, Graph
from rlmflow.graph import LLMUsage
from rlmflow.llm import (
    LLMChunk,
    LLMClient,
    Reply,
    ToolCall,
    as_chunk,
    complete,
    join_chunks,
    stream_with_retry,
)
from rlmflow.llm.anthropic import AnthropicClient
from rlmflow.llm.openai import OpenAIClient


@pytest.fixture(autouse=True)
def _fast_stream_retry(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("rlmflow.llm.base._retry_sleep", no_sleep)


class _ChatClient(LLMClient):
    def chat(self, messages, *args, **kwargs):
        self.last_usage = LLMUsage(2, 3)
        return "hello"


def test_default_stream_is_one_chunk_from_chat():
    chunks = list(_ChatClient().stream([{"role": "user", "content": "hi"}]))
    assert [chunk.text for chunk in chunks] == ["hello"]
    assert chunks[-1].usage == LLMUsage(2, 3)


def test_complete_awaits_async_clients_with_kwargs_and_timeout():
    seen = {}

    class Client:
        async def completion(self, messages, *, marker):
            seen.update(messages=messages, marker=marker)
            return "async", LLMUsage(4, 5)

    messages = [{"role": "user", "content": "hi"}]

    assert asyncio.run(complete(Client(), messages, timeout=2, marker="sample")) == Reply(
        "async", usage=LLMUsage(4, 5)
    )
    assert seen == {"messages": messages, "marker": "sample"}


def test_complete_runs_blocking_clients_on_a_thread():
    main_thread = threading.get_ident()
    chat_threads = []

    class Client:
        last_usage = LLMUsage(1, 2)

        def chat(self, _messages):
            chat_threads.append(threading.get_ident())
            return "ab"

    assert asyncio.run(complete(Client(), [])) == Reply("ab", usage=LLMUsage(1, 2))
    assert chat_threads != [main_thread]


def test_complete_awaits_an_async_chat():
    class Client:
        async def chat(self, _messages):
            return "awaited"

    assert asyncio.run(complete(Client(), [])) == Reply("awaited")


def test_complete_times_out_a_hung_client():
    class Hung:
        async def completion(self, _messages):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError, match=r"model call to Hung exceeded 0.01s"):
        asyncio.run(complete(Hung(), [], timeout=0.01))


def test_built_in_clients_cap_each_call_by_default_and_take_an_override():
    assert OpenAIClient("gpt-6-sol", client=object()).request_timeout == 300
    assert OpenAIClient("gpt-6-sol", request_timeout=60, client=object()).request_timeout == 60
    assert OpenAIClient("gpt-6-sol", request_timeout=None, client=object()).request_timeout is None
    assert AnthropicClient(client=object()).request_timeout == 300


def test_join_matches_stream_chunks():
    class Pieces:
        def stream(self, messages, **_kwargs):
            yield LLMChunk(text="ab")
            yield LLMChunk(text="c", usage=LLMUsage(1, 2))

    reply = asyncio.run(join_chunks(Pieces().stream([])))
    assert reply == Reply("abc", usage=LLMUsage(1, 2))


def test_as_chunk_accepts_plain_strings():
    assert as_chunk("x") == LLMChunk(text="x")
    assert as_chunk(None) == LLMChunk()


class APIConnectionError(Exception):
    """Name-matched to is_retryable without importing openai."""


def _delta(text=None, usage=None):
    choice = SimpleNamespace(delta=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice] if text is not None else [], usage=usage)


class _AsyncEvents:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        item = self._events.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.script = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _openai(fake: _FakeCompletions) -> OpenAIClient:
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return OpenAIClient(model="gpt-4o", client=sdk)


def test_agent_reasoning_effort_is_configurable_and_defaults_gpt6_to_low():
    # Sol rejects "none" and "minimal"; "low" is the floor every GPT-6 tier accepts.
    assert Agent("gpt-6-sol").reasoning_effort == "low"
    assert Agent("gpt-6.1-terra").reasoning_effort == "low"
    assert Agent("gpt-6-luna").reasoning_effort == "low"
    assert Agent("gpt-4o").reasoning_effort is None
    assert Agent("gpt-6.1-terra", reasoning_effort="high").reasoning_effort == "high"
    assert Agent("gpt-6.1-terra", reasoning_effort=None).reasoning_effort is None


def test_openai_stream_yields_deltas_and_trailing_usage():
    fake = _FakeCompletions()
    fake.script.append(
        _AsyncEvents(
            [
                _delta("hel"),
                _delta("lo"),
                _delta(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2)),
            ]
        )
    )
    client = _openai(fake)

    reply = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert reply == Reply("hello", usage=LLMUsage(4, 2))
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["stream_options"] == {"include_usage": True}


def test_openai_reports_reasoning_tokens_without_double_counting_them():
    fake = _FakeCompletions()
    fake.script.append(
        _AsyncEvents(
            [
                _delta("hi"),
                _delta(
                    usage=SimpleNamespace(
                        prompt_tokens=4,
                        completion_tokens=100,
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=90),
                    )
                ),
            ]
        )
    )

    usage = asyncio.run(
        join_chunks(_openai(fake).stream([{"role": "user", "content": "hi"}]))
    ).usage
    assert usage == LLMUsage(4, 100, 90)
    # The provider already counted the 90 inside completion_tokens, so the budget
    # `total` feeds must not grow.
    assert usage.total == 104
    assert (usage + usage).reasoning_tokens == 180


def test_openai_usage_without_reasoning_details_reports_zero():
    fake = _FakeCompletions()
    fake.script.append(
        _AsyncEvents(
            [_delta("hi"), _delta(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2))]
        )
    )

    usage = asyncio.run(
        join_chunks(_openai(fake).stream([{"role": "user", "content": "hi"}]))
    ).usage
    assert usage.reasoning_tokens == 0


def test_openai_stream_forwards_sampling_kwargs():
    fake = _FakeCompletions()
    fake.script.append(_AsyncEvents([_delta("ok")]))
    client = _openai(fake)
    asyncio.run(
        join_chunks(
            client.stream(
                [{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=16,
                stop=["\n"],
            )
        )
    )
    sent = fake.calls[0]
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 16
    assert sent["stop"] == ["\n"]


def test_flow_passes_agent_reasoning_effort_only_to_openai_clients():
    fake = _FakeCompletions()
    fake.script.append(_AsyncEvents([_delta("ok")]))
    flow = Flow(llm_clients={"reasoning": _openai(fake)})

    asyncio.run(
        flow.complete(
            [{"role": "user", "content": "hi"}],
            "reasoning",
            reasoning_effort="none",
        )
    )

    assert fake.calls[0]["reasoning_effort"] == "none"


def test_openai_retries_before_first_content_token():
    fake = _FakeCompletions()
    fake.script.extend(
        [
            APIConnectionError("drop"),
            _AsyncEvents(
                [
                    _delta("ok"),
                    _delta(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
                ]
            ),
        ]
    )
    client = _openai(fake)
    reply = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert reply == Reply("ok", usage=LLMUsage(1, 1))
    assert len(fake.calls) == 2


def test_openai_does_not_retry_after_content():
    fake = _FakeCompletions()
    fake.script.append(_AsyncEvents([_delta("partial"), APIConnectionError("drop")]))
    client = _openai(fake)
    with pytest.raises(APIConnectionError):
        asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert len(fake.calls) == 1


class _AnthropicStream:
    def __init__(self, texts, usage, content=()):
        self._texts = list(texts)
        self._usage = usage
        self._content = list(content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for item in self._texts:
                if isinstance(item, Exception):
                    raise item
                yield item

        return gen()

    async def get_final_message(self):
        return SimpleNamespace(usage=self._usage, content=self._content)


class _FakeMessages:
    def __init__(self):
        self.calls = []
        self.script = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _anthropic(fake: _FakeMessages) -> AnthropicClient:
    sdk = SimpleNamespace(messages=fake)
    return AnthropicClient(model="claude-test", client=sdk)


def test_anthropic_stream_yields_text_then_usage():
    fake = _FakeMessages()
    fake.script.append(
        _AnthropicStream(["a", "b"], SimpleNamespace(input_tokens=3, output_tokens=5))
    )
    client = _anthropic(fake)
    reply = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert reply == Reply("ab", usage=LLMUsage(3, 5))
    assert fake.calls[0]["max_tokens"] == 8192


def test_anthropic_stream_forwards_sampling_kwargs():
    fake = _FakeMessages()
    fake.script.append(_AnthropicStream(["ok"], SimpleNamespace(input_tokens=1, output_tokens=1)))
    client = _anthropic(fake)
    asyncio.run(
        join_chunks(
            client.stream(
                [{"role": "user", "content": "hi"}],
                temperature=0.1,
                max_tokens=32,
                stop=["END"],
            )
        )
    )
    sent = fake.calls[0]
    assert sent["temperature"] == 0.1
    assert sent["max_tokens"] == 32
    assert sent["stop_sequences"] == ["END"]


def test_flow_consumes_a_clients_stream():
    class StreamLLM(LLMClient):
        def __init__(self):
            self.chat_calls = 0

        def stream(self, messages, *args, **kwargs):
            yield LLMChunk(calls=repl('submit("ok")').calls, usage=LLMUsage(2, 2))

        def chat(self, messages, *args, **kwargs):
            self.chat_calls += 1
            raise AssertionError("chat must not run when stream is available")

    llm = StreamLLM()
    g = Graph()
    root = g.start("go")
    flow = ScriptedFlow(llm)
    flow.run(g)
    assert root.final_result() == "ok"
    assert llm.chat_calls == 0
    [turn] = turns(g, root)
    assert LLMUsage.of(turn.data["usage"]) == LLMUsage(2, 2)


def test_chat_only_fake_still_runs():
    llm = StubLLM(lambda _messages: repl('submit("ok")'))
    assert ScriptedFlow(llm).run("default", "go") == "ok"


def _call_delta(index, *, id=None, name=None, arguments=None):
    part = SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )
    delta = SimpleNamespace(content=None, tool_calls=[part])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def test_openai_sends_tools_and_joins_streamed_tool_call_arguments():
    fake = _FakeCompletions()
    fake.script.append(
        _AsyncEvents(
            [
                _delta("checking"),
                _call_delta(0, id="call_1", name="repl", arguments='{"code": "pri'),
                _call_delta(0, arguments='nt(1)"}'),
                _delta(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
            ]
        )
    )
    tools = [{"type": "function", "function": {"name": "repl", "parameters": {}}}]

    reply = asyncio.run(complete(_openai(fake), [{"role": "user", "content": "hi"}], tools=tools))

    assert reply == Reply(
        "checking", (ToolCall("repl", '{"code": "print(1)"}', "call_1"),), LLMUsage(1, 1)
    )
    assert fake.calls[0]["tools"] == tools
    assert fake.calls[0]["parallel_tool_calls"] is False


class _FakeResponses:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _AsyncEvents(self.events)


def _event(type, **fields):
    return SimpleNamespace(type=type, **fields)


def test_openai_uses_responses_on_openais_endpoint_and_chat_elsewhere():
    openai = SimpleNamespace(base_url="https://api.openai.com/v1/")
    local = SimpleNamespace(base_url="http://localhost:8000/v1/")
    assert OpenAIClient("gpt-6-sol", client=openai).api == "responses"
    assert OpenAIClient("qwen3", client=local).api == "chat"
    assert OpenAIClient("gpt-6-sol", client=openai, api="chat").api == "chat"
    with pytest.raises(ValueError, match="api must be"):
        OpenAIClient("gpt-6-sol", client=openai, api="completions")


def test_openai_responses_translates_messages_and_streams_calls_and_usage():
    usage = SimpleNamespace(
        input_tokens=5,
        output_tokens=7,
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    call = SimpleNamespace(
        type="function_call", name="repl", arguments='{"code": "x = 1"}', call_id="call_9"
    )
    fake = _FakeResponses(
        [
            _event("response.output_text.delta", delta="look"),
            _event("response.output_item.done", item=call),
            _event("response.completed", response=SimpleNamespace(usage=usage)),
        ]
    )
    client = OpenAIClient(
        "gpt-6-sol",
        reasoning_effort="low",
        api="responses",
        client=SimpleNamespace(responses=fake),
    )
    tools = [{"type": "function", "function": {"name": "repl", "parameters": {}}}]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "node_1",
                    "type": "function",
                    "function": {"name": "repl", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "node_1", "content": "1"},
    ]

    reply = asyncio.run(complete(client, messages, tools=tools, max_tokens=50))

    assert reply == Reply(
        "look", (ToolCall("repl", '{"code": "x = 1"}', "call_9"),), LLMUsage(5, 7, 3)
    )
    [sent] = fake.calls
    assert sent["instructions"] == "sys"
    assert sent["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
        {"type": "function_call", "call_id": "node_1", "name": "repl", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "node_1", "output": "1"},
    ]
    assert sent["tools"] == [{"type": "function", "name": "repl", "parameters": {}}]
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["max_output_tokens"] == 50
    assert sent["parallel_tool_calls"] is False
    assert sent["store"] is False


def test_anthropic_translates_tools_calls_and_results():
    fake = _FakeMessages()
    tool_use = SimpleNamespace(type="tool_use", id="toolu_1", name="repl", input={"code": "x"})
    fake.script.append(
        _AnthropicStream([], SimpleNamespace(input_tokens=1, output_tokens=1), [tool_use])
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "repl", "description": "Run.", "parameters": {"type": "object"}},
        }
    ]
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "node_1",
                    "type": "function",
                    "function": {"name": "repl", "arguments": '{"code": "print(1)"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "node_1", "content": "1"},
        {"role": "user", "content": "Background agents: none"},
    ]

    reply = asyncio.run(complete(_anthropic(fake), messages, tools=tools))

    assert reply.calls == (ToolCall("repl", '{"code": "x"}', "toolu_1"),)
    sent = fake.calls[0]
    assert sent["system"] == "sys"
    assert sent["tools"] == [
        {"name": "repl", "description": "Run.", "input_schema": {"type": "object"}}
    ]
    assert sent["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "node_1", "name": "repl", "input": {"code": "print(1)"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "node_1", "content": "1"},
                {"type": "text", "text": "Background agents: none"},
            ],
        },
    ]


def test_complete_hands_tools_only_to_clients_that_take_them():
    class Plain:
        def chat(self, _messages):
            return repl("print(1)")

    tools = [{"type": "function", "function": {"name": "repl", "parameters": {}}}]
    assert asyncio.run(complete(Plain(), [], tools=tools)).calls == repl("print(1)").calls


def test_hung_client_hits_request_timeout():
    class Hung:
        request_timeout = 0.01

        async def completion(self, messages, **_kwargs):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        ScriptedFlow(Hung()).run("default", "hang")


def test_stream_with_retry_retries_then_gives_up():
    attempts = {"n": 0}

    def factory():
        attempts["n"] += 1

        async def gen():
            raise APIConnectionError("nope")
            yield LLMChunk()  # pragma: no cover — keeps this an async generator

        return gen()

    async def run():
        async for _ in stream_with_retry(factory):
            pass

    with pytest.raises(APIConnectionError):
        asyncio.run(run())
    assert attempts["n"] == 3
