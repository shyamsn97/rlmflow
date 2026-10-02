import asyncio
from types import SimpleNamespace

import pytest
from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import (
    AgentConfig,
    ConfidenceGate,
    FallbackEngine,
    Flow,
    InstructorEngine,
    JevEngine,
    JsonSchemaCodec,
    LowConfidenceError,
    RetryingEngine,
    StructuredOutputError,
    StructuredRequest,
    StructuredResult,
    UnsupportedStructuredOutputError,
    get_tool_metadata,
)

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["label"],
    "additionalProperties": False,
}


class Label(BaseModel):
    label: str


class RecordingCodec(JsonSchemaCodec):
    def __init__(self):
        self.validated = []

    def render_contract(self, schema, *, source):
        return f"CUSTOM {source}: {super().render_contract(schema, source=source)}"

    def validate(self, value, schema):
        self.validated.append(value)
        return super().validate(value, schema)


class FakeEngine:
    def __init__(self, result=None, error=None, reasons=()):
        self.result = result
        self.error = error
        self.reasons = reasons
        self.calls = 0

    def supports(self, request):
        return self.reasons

    async def generate(self, request, codec):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_flow_normalizes_root_contract_and_codec_owns_prompt_and_finish():
    codec = RecordingCodec()
    flow = Flow(
        StubLLM(lambda _messages: "```repl\nfinish({'label': 'yes'})\n```"),
        output_codec=codec,
    )
    root = flow.start("classify", config=AgentConfig(output_schema=Label))

    assert root.config.output_schema["type"] == "object"
    assert "CUSTOM finish" in flow.build_system_prompt(root).render()
    assert flow.run(root) == {"label": "yes"}
    assert codec.validated == [{"label": "yes"}]


def test_chat_structured_query_returns_value_and_usage_metadata():
    flow = Flow(StubLLM(lambda _messages: '{"label":"yes"}'))

    result = asyncio.run(
        flow.structured_query("choose", output_schema=SCHEMA, model="default")
    )

    assert result.value == {"label": "yes"}
    assert result.metadata["engine"] == "chat"
    assert result.metadata["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_custom_engine_is_explicit_and_locally_revalidated():
    engine = FakeEngine(
        StructuredResult({"label": "yes"}, {"engine": "fake", "confidence": {"label": 0.9}})
    )
    flow = Flow(StubLLM(lambda _messages: ""), structured_engine=engine)

    result = asyncio.run(flow.structured_query("state", output_schema=SCHEMA))

    assert result.value == {"label": "yes"}
    assert result.metadata["confidence"] == {"label": 0.9}

    unsupported = FakeEngine(reasons=("label is unsupported",))
    unsupported_flow = Flow(
        StubLLM(lambda _messages: ""),
        structured_engine=unsupported,
    )
    with pytest.raises(UnsupportedStructuredOutputError, match="label is unsupported"):
        asyncio.run(unsupported_flow.structured_query("state", output_schema=SCHEMA))
    assert unsupported.calls == 0

    invalid_flow = Flow(
        StubLLM(lambda _messages: ""),
        structured_engine=FakeEngine(StructuredResult({"label": "maybe"})),
    )
    with pytest.raises(StructuredOutputError):
        asyncio.run(invalid_flow.structured_query("state", output_schema=SCHEMA))


def test_engine_wrappers_make_retry_fallback_and_confidence_explicit():
    flaky = FakeEngine(error=RuntimeError("temporary"))
    good = FakeEngine(
        StructuredResult({"label": "yes"}, {"confidence": {"label": 0.95}})
    )
    fallback = FallbackEngine(RetryingEngine(flaky, attempts=2), good)
    gated = ConfidenceGate(fallback, minimum=0.9)

    result = asyncio.run(
        gated.generate(
            StructuredRequest("state", SCHEMA),
            JsonSchemaCodec(),
        )
    )

    assert result.value == {"label": "yes"}
    assert flaky.calls == 2
    assert good.calls == 1

    non_retryable = RuntimeError("auth")
    non_retryable.retryable = False
    refused = FakeEngine(error=non_retryable)
    with pytest.raises(RuntimeError, match="auth"):
        asyncio.run(
            RetryingEngine(refused, attempts=3).generate(
                StructuredRequest("state", SCHEMA),
                JsonSchemaCodec(),
            )
        )
    assert refused.calls == 1

    with pytest.raises(LowConfidenceError):
        asyncio.run(
            ConfidenceGate(
                FakeEngine(
                    StructuredResult(
                        {"label": "yes"},
                        {"confidence": {"label": 0.2}},
                    )
                ),
                minimum=0.9,
            ).generate(StructuredRequest("state", SCHEMA), JsonSchemaCodec())
        )


class FakeJevClient:
    def __init__(self):
        self.calls = []

    def convert(self, schema):
        return ("plan", schema)

    async def ask(self, plan, state, *, model):
        self.calls.append((plan, state, model))
        return SimpleNamespace(
            output={"label": "yes"},
            probabilities={"label": {"yes": 0.9, "no": 0.1}},
            confidence={"label": 0.9},
            usage=SimpleNamespace(input_tokens=12, output_tokens=1),
        )


def test_jev_engine_preserves_decision_metadata_and_revalidates():
    client = FakeJevClient()
    flow = Flow(
        StubLLM(lambda _messages: ""),
        structured_engine=JevEngine(client),
    )

    result = asyncio.run(flow.structured_query("evidence", output_schema=SCHEMA))

    assert result.value == {"label": "yes"}
    assert result.metadata["probabilities"]["label"]["yes"] == 0.9
    assert result.metadata["confidence"] == {"label": 0.9}
    assert result.metadata["usage"] == {"input_tokens": 12, "output_tokens": 1}
    assert client.calls[0][1:] == ("evidence", "jev-latest")

    assert asyncio.run(flow.llm_query("more evidence", schema=SCHEMA)) == {
        "label": "yes"
    }
    assert client.calls[1][1:] == ("more evidence", "jev-latest")


class FakeInstructorClient:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return Label(label="yes")


def test_instructor_engine_uses_pydantic_contract_and_rejects_schema_only():
    client = FakeInstructorClient()
    flow = Flow(
        StubLLM(lambda _messages: ""),
        structured_engine=InstructorEngine(client, max_retries=2),
    )

    result = asyncio.run(flow.structured_query("evidence", output_schema=Label))

    assert result.value == {"label": "yes"}
    assert result.metadata["engine"] == "instructor"
    assert client.calls[0]["response_model"] is Label
    assert client.calls[0]["max_retries"] == 2
    assert "model" not in client.calls[0]

    with pytest.raises(UnsupportedStructuredOutputError, match="Pydantic"):
        asyncio.run(flow.structured_query("evidence", output_schema=SCHEMA))


def test_llm_query_schema_delegates_to_chat_structured_engine():
    flow = Flow(StubLLM(lambda _messages: '{"label":"yes"}'))

    value = asyncio.run(flow.llm_query("classify", schema=SCHEMA))

    assert value == {"label": "yes"}


def test_query_builtins_are_ordinary_node_bound_tools():
    calls = []
    flow = Flow(
        StubLLM(lambda _messages: calls.append("default") or "default"),
        llm_clients={
            "worker": StubLLM(lambda _messages: calls.append("worker") or "worker")
        },
        use_llm_query=True,
    )
    bound = flow.build_tools(flow.start("q", config=AgentConfig(model="worker")))

    query_meta = get_tool_metadata(bound["llm_query"])
    assert query_meta is not None and "Omitting `model` uses `worker`" in query_meta.description
    assert bound["llm_query"] is not flow.llm_query
    assert asyncio.run(bound["llm_query"]("inspect")) == "worker"
    assert calls == ["worker"]
