"""LLM clients, streaming adapters, and ``complete``."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from helpers import ScriptedFlow, StubLLM, turns

from rlmflow import Graph, LLMChunk, LLMUsage
from rlmflow.llm import LLMClient, as_chunk, complete, join_chunks, stream_with_retry
from rlmflow.llm.anthropic import AnthropicClient
from rlmflow.llm.openai import OpenAIClient


@pytest.fixture(autouse=True)
def _fast_stream_retry(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("rlmflow.llm._retry_sleep", no_sleep)


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

    assert asyncio.run(complete(Client(), messages, timeout=2, marker="sample")) == (
        "async",
        LLMUsage(4, 5),
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

    assert asyncio.run(complete(Client(), [])) == ("ab", LLMUsage(1, 2))
    assert chat_threads != [main_thread]


def test_complete_awaits_an_async_chat():
    class Client:
        async def chat(self, _messages):
            return "awaited"

    assert asyncio.run(complete(Client(), [])) == ("awaited", LLMUsage())


def test_complete_times_out_a_hung_client():
    class Hung:
        async def completion(self, _messages):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(complete(Hung(), [], timeout=0.01))


def test_join_matches_stream_chunks():
    class Pieces:
        def stream(self, messages, **_kwargs):
            yield LLMChunk(text="ab")
            yield LLMChunk(text="c", usage=LLMUsage(1, 2))

    text, usage = asyncio.run(join_chunks(Pieces().stream([])))
    assert text == "abc"
    assert usage == LLMUsage(1, 2)


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

    text, usage = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert text == "hello"
    assert usage == LLMUsage(4, 2)
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

    _text, usage = asyncio.run(
        join_chunks(_openai(fake).stream([{"role": "user", "content": "hi"}]))
    )
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

    _text, usage = asyncio.run(
        join_chunks(_openai(fake).stream([{"role": "user", "content": "hi"}]))
    )
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
    text, usage = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert text == "ok"
    assert usage == LLMUsage(1, 1)
    assert len(fake.calls) == 2


def test_openai_does_not_retry_after_content():
    fake = _FakeCompletions()
    fake.script.append(_AsyncEvents([_delta("partial"), APIConnectionError("drop")]))
    client = _openai(fake)
    with pytest.raises(APIConnectionError):
        asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert len(fake.calls) == 1


class _AnthropicStream:
    def __init__(self, texts, usage):
        self._texts = list(texts)
        self._usage = usage

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
        return SimpleNamespace(usage=self._usage)


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
    text, got = asyncio.run(join_chunks(client.stream([{"role": "user", "content": "hi"}])))
    assert text == "ab"
    assert got == LLMUsage(3, 5)
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
            yield LLMChunk(text='```repl\nsubmit("ok")\n```', usage=LLMUsage(2, 2))

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
    llm = StubLLM(lambda _messages: '```repl\nsubmit("ok")\n```')
    assert ScriptedFlow(llm).run("go") == "ok"


def test_hung_client_hits_request_timeout():
    class Hung:
        request_timeout = 0.01

        async def completion(self, messages, **_kwargs):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        ScriptedFlow(Hung()).run("hang")


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
