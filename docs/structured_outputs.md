# Structured outputs

Structured output has two separate jobs:

1. an **output codec** defines and validates the contract;
1. a **structured engine** produces a candidate value from supplied state.

Keeping them separate means agent completion remains deterministic while one-shot extraction and classification can use chat models, Jev, or explicit compositions of those engines.

## Agent completion

`output_schema` is a durable JSON Schema contract. The model still completes through the normal strict REPL protocol:

```repl
submit({"label": "relevant"})
```

`submit(value)` validates locally through the Flow's `OutputCodec`. It never calls another model. A valid candidate lands as a durable `Submission`; the normal graph transition then accepts it as a `DoneOutput`. `DoneOutput.result`, child handoffs, and persisted results remain strict JSON values.

Pydantic models and `TypeAdapter` instances are accepted at live API boundaries and normalized to JSON Schema before entering `AgentConfig`.

## Custom codecs

`JsonSchemaCodec` is the default. A codec owns schema normalization, contract prompt text, text parsing, and Python-value validation:

```python
from rlmflow import Flow, JsonSchemaCodec

class DomainCodec(JsonSchemaCodec):
    def validate(self, value, schema):
        value = super().validate(value, schema)
        if isinstance(value, dict) and value.get("score", 0) < 0:
            raise ValueError("score must be non-negative")
        return value

flow = Flow(model, output_codec=DomainCodec())
```

Wrapping a codec is the extension point for semantic validation, custom contract wording, auditing, or domain-specific errors. The graph persists the canonical JSON Schema, not the live codec object.

## Structured queries

Use `Flow.structured_query(...)` when a service should produce a structured decision from state:

```python
result = await flow.structured_query(
    review_text,
    output_schema=review_schema,
    instruction="Classify this review.",
)

result.value
result.metadata
```

Agent REPLs use the one structured model-call path, `llm_query(..., schema=...)`:

```python
decision = await llm_query(
    review,
    schema=review_schema,
)
print(decision)
```

`llm_query` also remains the free-form one-shot Q&A and transformation tool. When `schema=` is provided, it delegates to the Flow's configured structured engine. A Flow has one structured engine; the default is `ChatStructuredEngine`, backed by the Flow's normal LLM.

## Jev

Install the optional integration:

```bash
pip install "rlmflow[jev]"
```

Then select Jev for the Flow:

```python
from rlmflow import Flow, JevEngine

flow = Flow(
    model,
    structured_engine=JevEngine(),
)

result = await flow.structured_query(
    ticket_text,
    output_schema=ticket_decision_schema,
)
```

Jev is a decision engine, not a chat model or validator. It supports booleans, 0–1 probabilities, enums, and small bounded integer ranges. It rejects free-form strings, arrays, nested objects, references, unions, and other unsupported schemas before network access. Field descriptions strongly affect decision quality.

Jev metadata preserves field probabilities, confidence, model, and usage. Its value is validated again through the configured codec.

## Instructor

Install the optional integration:

```bash
pip install "rlmflow[instructor]"
```

Instructor uses the original Pydantic model as its response contract:

```python
from pydantic import BaseModel
from rlmflow import Flow, InstructorEngine

class Decision(BaseModel):
    label: str

flow = Flow(
    model,
    structured_engine=InstructorEngine(provider="openai/gpt-4o-mini"),
)
result = await flow.structured_query(text, output_schema=Decision)
```

`InstructorEngine` intentionally rejects plain JSON Schema dictionaries because reconstructing an equivalent Pydantic model is not lossless. You can also pass an already configured Instructor client with `InstructorEngine(client)`.

## Explicit behavior composition

Retries, fallback, and confidence policy are wrappers around the Flow's one engine rather than hidden global hooks:

```python
from rlmflow import (
    ChatStructuredEngine,
    ConfidenceGate,
    FallbackEngine,
    Flow,
    JevEngine,
    RetryingEngine,
)

flow = Flow(
    model,
    structured_engine=FallbackEngine(
        ConfidenceGate(
            RetryingEngine(JevEngine(), attempts=2),
            minimum=0.8,
        ),
        ChatStructuredEngine(),
    ),
)
```

Unbound chat engines inside a composition are bound to the Flow automatically. Nothing retries or falls back unless the host passes such a composition.
