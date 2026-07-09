"""Phase 2 — LLM power: client protocol, channel, structured output, batching."""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, TypeAdapter

from rflow import (
    AnthropicClient,
    DoneSignal,
    Flow,
    Graph,
    LLMChannel,
    LLMClient,
    LLMUsage,
    OpenAIClient,
)
from rflow.integrations.structured import (
    StructuredOutputError,
    StructuredOutputParser,
    json_schema_for,
)
from rflow.minimal.clients.llm import TinkerClient, is_retryable, retry_transient
from rflow.runtime.context import EngineContext

from rflow import is_errored

from .helpers import ScriptedLLM, StubLLM, make_flow, run_to_completion

DONE_OK = '```repl\ndone("ok")\n```'


# ── test doubles ──────────────────────────────────────────────────────


class PlainClient(LLMClient):
    """Chat-only client; relies on the default ``completion``/``stream``."""

    def __init__(self, reply: str = "abc") -> None:
        self.reply = reply

    def chat(self, messages, *args, **kwargs) -> str:
        return self.reply


class NameClient(LLMClient):
    """Returns its own name — used to assert channel routing."""

    thread_safe = True

    def __init__(self, name: str) -> None:
        self.name = name

    def chat(self, messages, *args, **kwargs) -> str:
        return self.name


class EchoClient(LLMClient):
    """Echoes the last user message — used to assert batch ordering."""

    thread_safe = True

    def chat(self, messages, *args, **kwargs) -> str:
        return messages[-1]["content"]


class UsageClient(LLMClient):
    """Returns a fixed reply and reports fixed token usage."""

    thread_safe = True

    def __init__(self, reply: str, usage: tuple[int, int]) -> None:
        self.reply = reply
        self._usage = usage

    def chat(self, messages, *args, **kwargs) -> str:
        text, _ = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        usage = LLMUsage(input_tokens=self._usage[0], output_tokens=self._usage[1])
        self.last_usage = usage
        return self.reply, usage


class LaneClient(LLMClient):
    """Returns scripted replies in sequence; records call count thread-safely."""

    thread_safe = True

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        with self._lock:
            i = min(self.calls, len(self.replies) - 1)
            self.calls += 1
            return self.replies[i]


class BusyClient(LLMClient):
    """Tracks peak concurrent in-flight calls."""

    thread_safe = True

    def __init__(self, hold: float = 0.05) -> None:
        self.hold = hold
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.hold)
        with self._lock:
            self.active -= 1
        return "ok"


class Out(BaseModel):
    answer: int
    note: str


JSON_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}


# ── 1. llm.py: protocol, retries, providers ────────────────────────────


def test_default_completion_joins_stream_and_returns_empty_usage():
    text, usage = PlainClient("hello").completion([{"role": "user", "content": "x"}])
    assert text == "hello"
    assert usage == LLMUsage()


def test_is_retryable_distinguishes_transient_from_timeout():
    def named(name: str) -> Exception:
        return type(name, (Exception,), {})()

    assert is_retryable(named("APIConnectionError")) is True
    assert is_retryable(named("RateLimitError")) is True
    assert is_retryable(named("APITimeoutError")) is False
    assert is_retryable(named("ValueError")) is False

    wrapped = ValueError("boom")
    wrapped.__cause__ = type("ConnectError", (Exception,), {})()
    assert is_retryable(wrapped) is True


def test_retry_transient_retries_connection_then_succeeds():
    attempts = {"n": 0}

    @retry_transient
    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise type("APIConnectionError", (Exception,), {})()
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 2


def test_retry_transient_does_not_retry_timeouts():
    attempts = {"n": 0}

    @retry_transient
    def slow() -> str:
        attempts["n"] += 1
        raise type("APITimeoutError", (Exception,), {})()

    with pytest.raises(Exception):
        slow()
    assert attempts["n"] == 1


def test_openai_client_completion_and_chat_agree():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )
    client = object.__new__(OpenAIClient)
    client.model = "gpt-test"
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: resp)
        )
    )
    text, usage = client.completion([{"role": "user", "content": "hi"}])
    assert text == "hi there"
    assert (usage.input_tokens, usage.output_tokens) == (3, 4)
    assert client.last_usage == usage
    assert client.chat([{"role": "user", "content": "hi"}]) == "hi there"


def test_anthropic_split_messages_extracts_system():
    client = object.__new__(AnthropicClient)
    system, chat = client.split_messages(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
    )
    assert system == "SYS"
    assert chat == [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]


# ── 2. LLMChannel ───────────────────────────────────────────────────────


def test_channel_call_routes_to_named_lane():
    channel = LLMChannel(
        {"default": NameClient("A"), "b": NameClient("B")}, max_concurrency=2
    )
    text, _ = channel.call("b", [{"role": "user", "content": "hi"}])
    assert text == "B"
    channel.shutdown()


def test_channel_unknown_model_raises_listing_keys():
    channel = LLMChannel({"default": EchoClient()}, max_concurrency=1)
    with pytest.raises(ValueError, match="unknown model"):
        channel.call("ghost", [{"role": "user", "content": "x"}])
    channel.shutdown()


def test_channel_closed_raises():
    channel = LLMChannel({"default": EchoClient()}, max_concurrency=1)
    channel.shutdown()
    with pytest.raises(RuntimeError):
        channel.call("default", [{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError):
        channel.batch("default", ["a"])


def test_channel_batch_preserves_order_parallel():
    channel = LLMChannel({"default": EchoClient()}, max_concurrency=4)
    pairs = channel.batch("default", ["p0", "p1", "p2", "p3"])
    assert [text for text, _ in pairs] == ["p0", "p1", "p2", "p3"]
    channel.shutdown()


def test_channel_batch_preserves_order_serial():
    channel = LLMChannel(
        {"default": EchoClient()}, max_concurrency=4, thread_safe={"default": False}
    )
    pairs = channel.batch("default", ["p0", "p1", "p2"])
    assert [text for text, _ in pairs] == ["p0", "p1", "p2"]
    channel.shutdown()


def test_channel_bounds_concurrency():
    client = BusyClient()
    channel = LLMChannel({"default": client}, max_concurrency=1)
    channel.batch("default", ["a", "b", "c", "d"])
    assert client.peak == 1
    channel.shutdown()


def test_channel_runs_in_parallel_when_allowed():
    client = BusyClient()
    channel = LLMChannel({"default": client}, max_concurrency=4)
    channel.batch("default", ["a", "b", "c", "d"])
    assert client.peak >= 2
    channel.shutdown()


def test_channel_timeout_propagates():
    class Slow(LLMClient):
        thread_safe = True

        def chat(self, messages, *args, **kwargs) -> str:
            time.sleep(0.5)
            return "x"

    channel = LLMChannel({"default": Slow()}, max_concurrency=1, request_timeout=0.05)
    with pytest.raises(TimeoutError):
        channel.call("default", [{"role": "user", "content": "x"}])
    channel.shutdown()


# ── 3. Flow wiring ──────────────────────────────────────────────────────


def test_single_client_flow_unchanged():
    flow = make_flow(DONE_OK)
    assert flow.run("q") == "ok"


def test_step_llm_records_usage_and_tokens():
    flow = Flow(UsageClient(DONE_OK, usage=(10, 5)), max_iters=5)
    assert flow.run("q") == "ok"
    assert flow.last_usage == LLMUsage(input_tokens=10, output_tokens=5)
    outs = flow.graph.all_nodes.llm_outputs()
    assert outs and outs[0].input_tokens == 10 and outs[0].output_tokens == 5


def test_per_agent_model_selects_lane():
    launch = (
        "```repl\n"
        'await launch_subagents([{"query": "sub", "model": "fast"}])\n'
        "```"
    )
    default = LaneClient([launch, '```repl\ndone("root")\n```'])
    fast = LaneClient(['```repl\ndone("child")\n```'])
    flow = Flow(default, llm_clients={"fast": fast}, max_iters=5, max_depth=2)
    assert flow.run("q") == "root"
    assert fast.calls >= 1
    child = flow.graph.children["root.subagent"]
    assert child.model == "fast"
    assert child.result() == "child"


def test_close_shuts_down_channel():
    flow = make_flow(DONE_OK)
    flow.close()
    with pytest.raises(RuntimeError):
        flow.call_llm([{"role": "user", "content": "x"}])


# ── 4. structured output ────────────────────────────────────────────────


def test_parser_pydantic_and_json_schema_paths():
    parser = StructuredOutputParser()
    parsed = parser('{"answer": 1, "note": "x"}', Out)
    assert parsed.answer == 1 and parsed.note == "x"
    assert parser('{"x": 2}', JSON_SCHEMA) == {"x": 2}


def test_parser_invalid_json_raises_structured_error():
    parser = StructuredOutputParser()
    with pytest.raises(StructuredOutputError) as info:
        parser("not json at all", Out)
    message = str(info.value)
    assert "Expected JSON Schema" in message
    assert "Validation error" in message


def test_build_messages_includes_schema_hint():
    flow = make_flow()
    graph = flow.start("q", output_schema=Out)
    system = flow.build_messages(graph, force_final=False)[0]["content"]
    assert "JSON Schema" in system
    assert "answer" in system and "note" in system


def test_done_validates_against_schema():
    flow = make_flow()
    graph = flow.start("q", output_schema=Out)
    context = EngineContext(output_schema=graph.output_schema)
    tools = flow.build_tools(context)
    with pytest.raises(DoneSignal):
        tools["done"]({"answer": 7, "note": "hi"})
    assert json.loads(context.done_result or "") == {"answer": 7, "note": "hi"}

    with pytest.raises(StructuredOutputError):
        tools["done"]("not structured")


def test_done_json_schema_dict():
    flow = make_flow()
    graph = flow.start("q", output_schema=JSON_SCHEMA)
    context = EngineContext(output_schema=graph.output_schema)
    tools = flow.build_tools(context)
    with pytest.raises(DoneSignal):
        tools["done"]({"x": 1})
    assert context.done_result == '{"x": 1}'
    with pytest.raises(StructuredOutputError):
        tools["done"]({"x": "not an int"})


def test_structured_run_end_to_end():
    flow = Flow(StubLLM('```repl\ndone({"answer": 42, "note": "hi"})\n```'), max_iters=5)
    result = flow.run("q", output_schema=Out)
    assert json.loads(result) == {"answer": 42, "note": "hi"}


def test_step_applies_output_schema_to_existing_graph():
    def reply_for(messages):
        system = messages[0]["content"]
        if "answer" in system and "note" in system:
            return '```repl\ndone({"answer": 7, "note": "updated"})\n```'
        return '```repl\ndone("plain")\n```'

    flow = Flow(ScriptedLLM(reply_for), max_iters=5)
    graph = flow.start("q")
    graph = flow.step(graph, output_schema=Out)
    while not graph.finished:
        graph = flow.step(graph)

    assert graph.output_schema == json_schema_for(Out)
    assert json.loads(graph.result()) == {"answer": 7, "note": "updated"}


def test_step_followup_can_change_output_schema():
    def reply_for(messages):
        system = messages[0]["content"]
        if "answer" in system and "note" in system:
            return '```repl\ndone({"answer": 9, "note": "followup"})\n```'
        return '```repl\ndone("plain")\n```'

    flow = Flow(ScriptedLLM(reply_for), max_iters=5)
    graph = run_to_completion(flow, "plain task")
    assert graph.result() == "plain"

    graph = flow.step(query="now answer with structure", output_schema=Out)
    while not graph.finished:
        graph = flow.step(graph)

    assert graph.output_schema == json_schema_for(Out)
    assert json.loads(graph.result()) == {"answer": 9, "note": "followup"}


def test_direct_graph_output_schema_mutation_syncs_live_repl():
    replies = iter(
        [
            '```repl\nprint("created repl")\n```',
            '```repl\ndone({"answer": 11, "note": "direct"})\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=5)
    graph = flow.start("plain first")
    graph = flow.step(graph)  # LLMOutput
    graph = flow.step(graph)  # ExecOutput, root REPL now exists
    assert "root" in flow.repls

    graph.output_schema = json_schema_for(Out)
    graph = flow.step(graph)  # LLMOutput sees refreshed schema prompt
    graph = flow.step(graph)  # done(...) validates against graph.output_schema

    assert json.loads(graph.result()) == {"answer": 11, "note": "direct"}


def test_direct_graph_output_schema_removal_syncs_live_repl():
    replies = iter(
        [
            '```repl\ndone({"answer": 1, "note": "structured"})\n```',
            '```repl\ndone("plain after schema removal")\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=5)
    graph = run_to_completion(flow, "structured", output_schema=Out)
    assert "root" not in flow.repls
    assert json.loads(graph.result()) == {"answer": 1, "note": "structured"}

    graph.output_schema = None
    graph = flow.step(graph, query="plain follow-up")
    while not graph.finished:
        graph = flow.step(graph)

    assert graph.output_schema is None
    assert graph.result() == "plain after schema removal"


def test_done_result_does_not_persist_into_followup_execution():
    replies = iter(
        [
            '```repl\ndone({"answer": 1, "note": "structured"})\n```',
            '```repl\nprint("not done yet")\n```',
            '```repl\ndone("plain final")\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=6)
    graph = run_to_completion(flow, "structured", output_schema=Out)
    assert json.loads(graph.result()) == {"answer": 1, "note": "structured"}
    assert "root" not in flow.repls

    graph.output_schema = None
    graph = flow.step(graph, query="plain follow-up")  # LLMOutput with print only
    graph = flow.step(graph)  # ExecOutput; stale done_result must not terminate
    assert graph.current().type == "exec_output"
    assert not graph.finished
    assert "not done yet" in graph.current().output

    while not graph.finished:
        graph = flow.step(graph)
    assert graph.result() == "plain final"


def test_direct_graph_input_mutation_syncs_live_repl_inputs():
    replies = iter(
        [
            '```repl\nprint("created repl")\n```',
            '```repl\nprint(INPUTS["extra"])\ndone("ok")\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=5)
    graph = flow.start("q")
    graph = flow.step(graph)  # LLMOutput
    graph = flow.step(graph)  # ExecOutput, root REPL now exists
    assert "root" in flow.repls

    graph.inputs["extra"] = "synced-from-graph"
    graph = flow.step(graph)  # LLMOutput
    graph = flow.step(graph)  # ExecOutput/DoneOutput

    assert graph.result() == "ok"
    assert "synced-from-graph" in graph.nodes[-1].output


def test_followup_query_updates_graph_query_and_appends_message():
    replies = iter(
        [
            '```repl\ndone("first")\n```',
            '```repl\ndone("second")\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=5)
    graph = run_to_completion(flow, "first task")
    assert graph.result() == "first"

    graph = flow.step(graph, query="second task")
    assert graph.query == "second task"
    # the follow-up query is delivered as a new user message, not via INPUTS
    user_text = " ".join(m["content"] for m in graph.messages() if m["role"] == "user")
    assert "second task" in user_text
    while not graph.finished:
        graph = flow.step(graph)

    assert graph.result() == "second"


def test_step_existing_finished_graph_with_new_query_calls_llm_on_followup():
    seen_messages: list[list[dict[str, str]]] = []
    replies = iter(
        [
            '```repl\ndone("first")\n```',
            '```repl\ndone("second")\n```',
        ]
    )

    def reply_for(messages: list[dict[str, str]]) -> str:
        seen_messages.append(messages)
        return next(replies)

    flow = Flow(ScriptedLLM(reply_for), max_iters=1)
    graph = run_to_completion(flow, "first task")
    assert graph.finished
    assert graph.result() == "first"

    original = graph.copy()
    graph = flow.step(graph, query="second task")

    assert original.finished
    assert len(graph.nodes) > len(original.nodes)
    assert graph.nodes[len(original.nodes)].type == "user_query"
    assert "New user task:\nsecond task" in graph.nodes[len(original.nodes)].content
    assert "You have not interacted with the REPL environment" not in graph.nodes[
        len(original.nodes)
    ].content
    assert not graph.finished
    assert graph.current().type == "llm_output"
    assert seen_messages[-1][-1]["role"] == "user"
    assert "New user task:\nsecond task" in seen_messages[-1][-1]["content"]
    assert "delegate them with `await launch_subagents" in seen_messages[-1][-1]["content"]

    graph = flow.step(graph)
    assert graph.result() == "second"


def test_step_existing_finished_graph_with_new_query_and_inputs_updates_repl_inputs():
    seen_messages: list[list[dict[str, str]]] = []
    replies = iter(
        [
            '```repl\ndone("first")\n```',
            '```repl\nprint(INPUTS["context"])\ndone(INPUTS["context"])\n```',
        ]
    )

    def reply_for(messages: list[dict[str, str]]) -> str:
        seen_messages.append(messages)
        return next(replies)

    flow = Flow(ScriptedLLM(reply_for), max_iters=3)
    graph = run_to_completion(flow, "first task", inputs={"context": "old context"})
    assert graph.finished
    assert graph.result() == "first"

    original = graph.copy()
    graph = flow.step(
        graph,
        query="second task",
        inputs={"context": "fresh context"},
    )

    assert original.inputs["context"] == "old context"
    assert graph.query == "second task"
    assert graph.inputs["context"] == "fresh context"
    assert graph.nodes[len(original.nodes)].type == "user_query"
    assert "New user task:\nsecond task" in graph.nodes[len(original.nodes)].content
    assert "fresh context" not in graph.nodes[len(original.nodes)].content
    assert seen_messages[-1][-1]["role"] == "user"
    assert "New user task:\nsecond task" in seen_messages[-1][-1]["content"]

    while not graph.finished:
        graph = flow.step(graph)

    assert graph.result() == "fresh context"
    assert "fresh context" in graph.nodes[-1].output


def test_query_is_not_mirrored_into_repl_inputs():
    replies = iter(
        [
            '```repl\nprint("query" in INPUTS, list(INPUTS))\ndone("ok")\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _messages: next(replies)), max_iters=5)
    graph = flow.start("some query", {"doc": "d"})
    while not graph.finished:
        graph = flow.step(graph)

    assert graph.result() == "ok"
    # INPUTS carries only caller inputs; the query is the message, not an input.
    assert "False" in graph.nodes[-1].output
    assert "'doc'" in graph.nodes[-1].output


def test_removed_agent_discards_repl_and_runtime_sync_cache():
    flow = make_flow(max_depth=1)
    flow.start("root")
    handle = flow.spawn_child("root", "child", "child task")
    child_id = handle.agent_id
    child = flow.graph[child_id]
    flow.repl_for(child)
    assert child_id in flow.repls
    assert child_id in flow.runtime.repl_env_cache
    assert child_id in flow.runtime.repl_inputs_cache

    flow.graph.remove_child(child_id)
    flow.sync_graph_state()

    assert child_id not in flow.repls
    assert child_id not in flow.runtime.repl_env_cache
    assert child_id not in flow.runtime.repl_inputs_cache


def test_graph_output_schema_round_trips():
    graph = Graph(agent_id="root", output_schema=json_schema_for(Out))
    restored = Graph.from_dict(graph.to_dict())
    assert restored.output_schema == graph.output_schema


# ── 5. llm_query_batched REPL tool ──────────────────────────────────────


def test_llm_query_batched_returns_texts_in_order():
    flow = Flow(EchoClient())
    assert flow.llm_query_batched(["a", "b", "c"]) == ["a", "b", "c"]


def test_llm_query_batched_validates_input():
    flow = Flow(EchoClient())
    with pytest.raises(TypeError):
        flow.llm_query_batched("not a list")
    with pytest.raises(TypeError):
        flow.llm_query_batched([1, 2])
    with pytest.raises(ValueError, match="unknown model"):
        flow.llm_query_batched(["a"], model="ghost")


def test_llm_query_batched_parses_structured():
    class JsonClient(LLMClient):
        thread_safe = True

        def chat(self, messages, *args, **kwargs) -> str:
            return '{"x": 7}'

    flow = Flow(JsonClient())
    out = flow.llm_query_batched(["a", "b"], output_schema=JSON_SCHEMA)
    assert out == [{"x": 7}, {"x": 7}]


def test_llm_query_batched_registered_and_reserved():
    # Disabled by default (keeps the prompt smaller).
    flow = make_flow()
    flow.start("q")
    assert "llm_query_batched" not in flow.build_tools()
    # Can be turned on explicitly.
    on = make_flow(include_llm_query=True)
    on.start("q")
    assert "llm_query_batched" in on.build_tools()
    # Reserved regardless of whether the tool is exposed.
    assert "llm_query_batched" in Flow._RESERVED
    with pytest.raises(ValueError, match="reserved"):
        flow.start("q", {"llm_query_batched": "x"})


# ── 6. DSPy integration (optional extra) ────────────────────────────────


def test_integrations_package_imports_without_dspy():
    module = importlib.import_module("rflow.integrations")
    assert hasattr(module, "StructuredOutputParser")


def test_dspy_adapter():
    import importlib.util

    if importlib.util.find_spec("dspy") is None:
        with pytest.raises(ModuleNotFoundError, match="dspy"):
            importlib.import_module("rflow.integrations.dspy")
    else:  # pragma: no cover - exercised only when the extra is installed
        from rflow.integrations.dspy import RecursiveFlowLM

        lm = RecursiveFlowLM(NameClient("X"))
        resp = lm.forward(prompt="hi")
        assert resp.choices[0].message.content == "X"


# ── 7. ported coverage: retries, channel, parser, structured, tinker ────


def test_is_retryable_treats_all_timeout_variants_as_non_retryable():
    def named(name: str) -> Exception:
        return type(name, (Exception,), {})()

    assert is_retryable(named("ReadTimeout")) is False
    assert is_retryable(named("TimeoutError")) is False
    assert is_retryable(named("ConnectTimeout")) is False
    # a transient error wrapping a timeout cause is still non-retryable
    wrapped = ValueError("boom")
    wrapped.__cause__ = named("ReadTimeout")
    assert is_retryable(wrapped) is False


def test_retry_transient_stops_after_three_attempts():
    attempts = {"n": 0}

    @retry_transient
    def always_flaky() -> str:
        attempts["n"] += 1
        raise type("APIConnectionError", (Exception,), {})()

    with pytest.raises(Exception):
        always_flaky()
    assert attempts["n"] == 3


class RecordingClient(LLMClient):
    """Thread-safe client that records the kwargs each completion receives."""

    thread_safe = True

    def __init__(self) -> None:
        self.kwargs: list[dict] = []
        self._lock = threading.Lock()

    def chat(self, messages, *args, **kwargs) -> str:
        text, _ = self.completion(messages, *args, **kwargs)
        return text

    def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
        with self._lock:
            self.kwargs.append(kwargs)
        return messages[-1]["content"], LLMUsage()


def test_channel_batch_forwards_sampling_kwargs_and_timeout():
    client = RecordingClient()
    channel = LLMChannel({"default": client}, max_concurrency=4, request_timeout=5)
    channel.batch("default", ["a", "b"], temperature=0.2, max_tokens=128, stop=["X"])
    channel.shutdown()
    assert len(client.kwargs) == 2
    for kw in client.kwargs:
        assert kw["temperature"] == 0.2
        assert kw["max_tokens"] == 128
        assert kw["stop"] == ["X"]
        assert kw["timeout"] == 5  # request_timeout injected by the channel


def test_channel_serializes_unsafe_client_to_one_in_flight():
    client = BusyClient()
    channel = LLMChannel(
        {"default": client}, max_concurrency=4, thread_safe={"default": False}
    )
    channel.batch("default", ["a", "b", "c", "d"])
    channel.shutdown()
    assert client.peak == 1


def test_channel_global_cap_holds_across_concurrent_callers():
    client = BusyClient(hold=0.03)
    channel = LLMChannel({"default": client}, max_concurrency=2)

    def worker():
        channel.batch("default", ["a", "b", "c"])

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    channel.shutdown()
    assert client.peak <= 2


def test_channel_batch_returns_per_request_usage():
    class IncUsageClient(LLMClient):
        thread_safe = True

        def __init__(self) -> None:
            self.n = 0
            self._lock = threading.Lock()

        def chat(self, messages, *args, **kwargs) -> str:
            text, _ = self.completion(messages, *args, **kwargs)
            return text

        def completion(self, messages, *args, **kwargs) -> tuple[str, LLMUsage]:
            with self._lock:
                self.n += 1
                tokens = self.n
            usage = LLMUsage(input_tokens=tokens, output_tokens=tokens * 2)
            self.last_usage = usage
            return "ok", usage

    channel = LLMChannel({"default": IncUsageClient()}, max_concurrency=1)
    pairs = channel.batch("default", ["p0", "p1", "p2"])
    channel.shutdown()
    inputs = sorted(usage.input_tokens for _text, usage in pairs)
    assert inputs == [1, 2, 3]  # distinct per-request usage, not a shared last_usage


def test_channel_batch_times_out_stuck_request():
    class Slow(LLMClient):
        thread_safe = True

        def chat(self, messages, *args, **kwargs) -> str:
            time.sleep(0.5)
            return "x"

    channel = LLMChannel({"default": Slow()}, max_concurrency=2, request_timeout=0.05)
    with pytest.raises(TimeoutError, match="timed out"):
        channel.batch("default", ["slow"])
    channel.shutdown()


def test_channel_batch_empty_returns_empty():
    channel = LLMChannel({"default": EchoClient()}, max_concurrency=2)
    assert channel.batch("default", []) == []
    channel.shutdown()


def test_parser_accepts_type_adapter_schema():
    parser = StructuredOutputParser()
    out = parser('{"a": ["x", "y"]}', TypeAdapter(dict[str, list[str]]))
    assert out == {"a": ["x", "y"]}


def test_parser_accepts_json_schema_string():
    parser = StructuredOutputParser()
    assert parser('{"x": 5}', json.dumps(JSON_SCHEMA)) == {"x": 5}


def test_parser_rejects_markdown_fenced_json_with_hint():
    parser = StructuredOutputParser()
    with pytest.raises(StructuredOutputError) as info:
        parser('```json\n{"x": 1}\n```', JSON_SCHEMA)
    message = str(info.value)
    assert "Markdown fences" in message
    assert "JSONDecodeError" in message


def test_parser_rejects_schema_validation_failure():
    parser = StructuredOutputParser()
    with pytest.raises(StructuredOutputError) as info:
        parser('{"x": "not an int"}', JSON_SCHEMA)
    assert '"not an int"' in str(info.value)


def test_structured_run_repairs_after_invalid_done():
    replies = iter(
        [
            '```repl\ndone({"answer": "NaN", "note": "bad"})\n```',  # fails int validation
            '```repl\ndone({"answer": 7, "note": "good"})\n```',
        ]
    )
    flow = Flow(ScriptedLLM(lambda _m: next(replies)), max_iters=5)
    g = run_to_completion(flow, "q", output_schema=Out)

    errored = [n for n in g.nodes if is_errored(n)]
    # the category lives on `.error`; the recovery hint is in `.output`.
    assert errored and "StructuredOutputError" in errored[0].output
    assert json.loads(g.result()) == {"answer": 7, "note": "good"}


def test_reusing_flow_for_plain_run_clears_root_schema():
    flow = make_flow()
    structured = flow.start("q", output_schema=Out)
    assert structured.output_schema is not None
    plain = flow.start("q2")
    assert plain.output_schema is None


def test_child_does_not_inherit_parent_schema_but_can_have_its_own():
    flow = make_flow(max_depth=2)
    flow.start("q", output_schema=Out)
    plain_child = flow.spawn_child("root", "plain", "do work")
    assert flow.graph[plain_child.agent_id].output_schema is None

    class ChildOut(BaseModel):
        score: int

    structured_child = flow.spawn_child("root", "scored", "do work", output_schema=ChildOut)
    assert flow.graph[structured_child.agent_id].output_schema == json_schema_for(ChildOut)


# ── TinkerClient (fake SDK) ─────────────────────────────────────────────


class _FakeRenderer:
    def build_generation_prompt(self, messages):
        return [1, 2, 3, 4]  # 4 input tokens

    def get_stop_sequences(self):
        return ["<|end|>"]

    def parse_response(self, tokens):
        return {"content": "hello-tinker"}


class _FakeFuture:
    def __init__(self, output):
        self._output = output

    def result(self, timeout=None):
        return self._output


class _FakeSampling:
    def __init__(self):
        self.calls = []

    def sample(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt, num_samples, sampling_params))
        return _FakeFuture(SimpleNamespace(sequences=[{"tokens": [7, 8, 9]}]))


def test_tinker_client_samples_and_tracks_usage(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tinker",
        SimpleNamespace(
            types=SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw))
        ),
    )
    sampling = _FakeSampling()
    client = TinkerClient(sampling_client=sampling, renderer_obj=_FakeRenderer())

    text, usage = client.completion(
        [{"role": "user", "content": "hi"}], temperature=0.2, timeout=5
    )
    assert text == "hello-tinker"
    assert (usage.input_tokens, usage.output_tokens) == (4, 3)
    assert client.last_usage == usage
    (_prompt, num_samples, params) = sampling.calls[0]
    assert num_samples == 1
    assert params.max_tokens == 8192 and params.temperature == 0.2
    assert params.stop == ["<|end|>"]
