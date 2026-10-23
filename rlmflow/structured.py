"""Structured output: the JSON-value codec and the engines that produce values."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import jsonschema
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

Schema = type[BaseModel] | TypeAdapter[Any] | Mapping[str, Any] | str
ContractSource = Literal["submit", "query"]
_JSON_VALUE = TypeAdapter(JsonValue)


@runtime_checkable
class OutputCodec(Protocol):
    """A replaceable JSON-value contract implementation."""

    def normalize(self, schema: Schema) -> dict[str, Any]: ...

    def render_contract(
        self,
        schema: Schema,
        *,
        source: ContractSource,
    ) -> str: ...

    def parse(self, content: str, schema: Schema) -> Any: ...

    def validate(self, value: object, schema: Schema) -> Any: ...


class StructuredOutputError(ValueError):
    def __init__(self, *, content: str, schema: Schema, cause: Exception) -> None:
        self.content = content
        self.schema = schema
        self.cause = cause
        super().__init__(_format_error_message(content, schema, cause))


class JsonSchemaCodec:
    """Default Pydantic/JSON-Schema codec.

    Runtime contracts are normalized to JSON Schema dictionaries before they
    enter agent configuration. Direct compatibility calls may still pass a
    Pydantic model or ``TypeAdapter`` and receive its typed result.
    """

    def normalize(self, schema: Schema) -> dict[str, Any]:
        if isinstance(schema, Mapping | str):
            normalized = dict(_load_json_schema(schema))
        else:
            normalized = _json_schema_for_pydantic(schema)
        jsonschema.validators.validator_for(normalized).check_schema(normalized)
        return normalized

    def render_contract(
        self,
        schema: Schema,
        *,
        source: ContractSource,
    ) -> str:
        schema_hint = json.dumps(self.normalize(schema), indent=2)
        if source == "query":
            return f"Return only JSON matching this schema:\n{schema_hint}"
        return (
            "This run requires structured output. When complete, call "
            "`submit(value)` with a JSON-compatible Python value matching this "
            "JSON Schema exactly — the value itself, not a status string:\n\n"
            f"```json\n{schema_hint}\n```"
        )

    def parse(self, content: str, schema: Schema) -> Any:
        """Parse JSON text and validate it against ``schema``."""
        if isinstance(schema, Mapping | str):
            try:
                value = json.loads(content, parse_constant=_reject_json_constant)
                jsonschema.validate(instance=value, schema=_load_json_schema(schema))
                return value
            except (json.JSONDecodeError, jsonschema.ValidationError, ValueError) as exc:
                raise StructuredOutputError(
                    content=content,
                    schema=schema,
                    cause=exc,
                ) from exc
        try:
            return _adapter_for(schema).validate_json(content)
        except ValidationError as exc:
            raise StructuredOutputError(content=content, schema=schema, cause=exc) from exc

    def validate(self, value: object, schema: Schema) -> Any:
        """Validate a Python JSON value or a pre-encoded JSON string."""
        if not isinstance(value, str):
            try:
                content = json.dumps(as_finish_value(value), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise StructuredOutputError(
                    content=repr(value),
                    schema=schema,
                    cause=exc,
                ) from exc
            return self.parse(content, schema)
        try:
            return self.parse(value, schema)
        except StructuredOutputError:
            # A bare enum string such as "continue" is a value, not a JSON document.
            return self.parse(json.dumps(value), schema)


DEFAULT_OUTPUT_CODEC = JsonSchemaCodec()


def json_schema_for(schema: Schema) -> dict[str, Any]:
    return DEFAULT_OUTPUT_CODEC.normalize(schema)


def as_finish_value(answer: object) -> JsonValue:
    """Return a strict JSON value or reject the completion."""
    try:
        value = _JSON_VALUE.validate_python(answer)
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, ValidationError) as exc:
        raise TypeError(
            "submit(answer) requires a JSON value: strings, finite numbers, "
            "booleans, None, lists, or dictionaries with string keys"
        ) from exc
    return value


def _adapter_for(schema: type[BaseModel] | TypeAdapter[Any]) -> TypeAdapter[Any]:
    if isinstance(schema, TypeAdapter):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return TypeAdapter(schema)
    raise TypeError(
        "schema must be a Pydantic model class, TypeAdapter, "
        "JSON schema dict, or JSON schema string"
    )


def _json_schema_for_pydantic(
    schema: type[BaseModel] | TypeAdapter[Any],
) -> dict[str, Any]:
    return _adapter_for(schema).json_schema()


def _load_json_schema(schema: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(schema, str):
        loaded = json.loads(schema, parse_constant=_reject_json_constant)
        if not isinstance(loaded, Mapping):
            raise TypeError("JSON schema string must decode to an object")
        return loaded
    return schema


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON")


def _format_error_message(content: str, schema: Schema, cause: Exception) -> str:
    schema_text = json.dumps(json_schema_for(schema), indent=2)
    return (
        "Structured output is invalid.\n"
        "Hint: call submit(value) with a JSON-compatible Python value that matches "
        "the expected schema. Do not pass prose, Markdown fences, or a JSON "
        "string containing JSON.\n\n"
        "Expected JSON Schema:\n"
        f"{schema_text}\n\n"
        "Received JSON:\n"
        f"{content}\n\n"
        "Validation error:\n"
        f"{type(cause).__name__}: {cause}"
    )


class UnsupportedStructuredOutputError(ValueError):
    """The selected engine cannot produce the requested contract."""

    def __init__(self, engine: str, reasons: Sequence[str]) -> None:
        self.engine = engine
        self.reasons = tuple(reasons)
        detail = "\n".join(f"- {reason}" for reason in self.reasons)
        super().__init__(f"structured engine {engine!r} does not support this schema:\n{detail}")


class LowConfidenceError(ValueError):
    """A structured result did not satisfy an explicit confidence policy."""


@dataclass(frozen=True)
class StructuredRequest:
    state: object
    schema: dict[str, Any]
    source_schema: Schema | None = None
    instruction: str | None = None
    model: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredResult:
    value: JsonValue
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class StructuredEngine(Protocol):
    """Produce a candidate JSON value for a canonical JSON Schema."""

    def supports(self, request: StructuredRequest) -> Sequence[str]: ...

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult: ...


class ChatStructuredEngine:
    """Prompt a normal chat model, then parse its textual JSON response."""

    def __init__(self, flow: Any = None) -> None:
        self.flow = flow

    def bind(self, flow: Any) -> ChatStructuredEngine:
        self.flow = flow
        return self

    def supports(self, request: StructuredRequest) -> Sequence[str]:
        return ()

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        if self.flow is None:
            raise RuntimeError("ChatStructuredEngine must be bound to a Flow")
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(as_finish_value(request.state), ensure_ascii=False)
        )
        parts = []
        if request.instruction:
            parts.append(request.instruction.strip())
        parts.extend((state, codec.render_contract(request.schema, source="query")))
        reply, usage = await self.flow.call_chat(
            [{"role": "user", "content": "\n\n".join(parts)}],
            request.model or "default",
            **dict(request.options),
        )
        value = codec.parse(reply, request.schema)
        return StructuredResult(
            value=as_finish_value(value),
            metadata={
                "engine": "chat",
                "model": request.model or "default",
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                },
            },
        )


class RetryingEngine:
    """Explicit bounded retries around another structured engine."""

    def __init__(
        self,
        engine: StructuredEngine,
        *,
        attempts: int = 2,
        delay: float = 0.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.engine = engine
        self.attempts = attempts
        self.delay = delay
        self.retry_on = retry_on

    def bind(self, flow: Any) -> RetryingEngine:
        _bind_engine(self.engine, flow)
        return self

    def supports(self, request: StructuredRequest) -> Sequence[str]:
        return self.engine.supports(request)

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        for attempt in range(1, self.attempts + 1):
            try:
                return await self.engine.generate(request, codec)
            except self.retry_on as exc:
                retryable = getattr(exc, "retryable", None)
                if retryable is False:
                    raise
                if attempt == self.attempts:
                    raise
                if self.delay:
                    await asyncio.sleep(self.delay)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await _aclose_engine(self.engine)


class FallbackEngine:
    """Try explicitly ordered engines when generation fails."""

    def __init__(self, *engines: StructuredEngine) -> None:
        if not engines:
            raise ValueError("FallbackEngine requires at least one engine")
        self.engines = engines

    def bind(self, flow: Any) -> FallbackEngine:
        for engine in self.engines:
            _bind_engine(engine, flow)
        return self

    def supports(self, request: StructuredRequest) -> Sequence[str]:
        if any(not engine.supports(request) for engine in self.engines):
            return ()
        return tuple(
            f"{type(engine).__name__}: {reason}"
            for engine in self.engines
            for reason in engine.supports(request)
        )

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        errors: list[BaseException] = []
        for engine in self.engines:
            reasons = engine.supports(request)
            if reasons:
                errors.append(UnsupportedStructuredOutputError(type(engine).__name__, reasons))
                continue
            try:
                return await engine.generate(request, codec)
            except Exception as exc:  # noqa: BLE001 - fallback policy is explicit
                errors.append(exc)
        detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        raise RuntimeError(f"all structured engines failed: {detail}") from errors[-1]

    async def aclose(self) -> None:
        closed: set[int] = set()
        for engine in self.engines:
            if id(engine) not in closed:
                closed.add(id(engine))
                await _aclose_engine(engine)


class ConfidenceGate:
    """Reject results whose reported field confidence falls below a threshold."""

    def __init__(self, engine: StructuredEngine, *, minimum: float) -> None:
        if not 0 <= minimum <= 1:
            raise ValueError("minimum confidence must be between 0 and 1")
        self.engine = engine
        self.minimum = minimum

    def bind(self, flow: Any) -> ConfidenceGate:
        _bind_engine(self.engine, flow)
        return self

    def supports(self, request: StructuredRequest) -> Sequence[str]:
        return self.engine.supports(request)

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        result = await self.engine.generate(request, codec)
        confidence = result.metadata.get("confidence")
        probabilities = result.metadata.get("probabilities")
        values = []
        if isinstance(confidence, dict):
            values.extend(
                float(value) for value in confidence.values() if isinstance(value, int | float)
            )
        if isinstance(probabilities, dict):
            for field, distribution in probabilities.items():
                if isinstance(confidence, dict) and isinstance(confidence.get(field), int | float):
                    continue
                if isinstance(distribution, dict):
                    choices = [
                        float(value)
                        for value in distribution.values()
                        if isinstance(value, int | float)
                    ]
                    if choices:
                        values.append(max(choices))
        if not values:
            raise LowConfidenceError("structured engine did not report confidence")
        below = min(values)
        if below < self.minimum:
            raise LowConfidenceError(
                f"structured result confidence {below:.3f} is below {self.minimum:.3f}"
            )
        return result

    async def aclose(self) -> None:
        await _aclose_engine(self.engine)


async def _aclose_engine(engine: StructuredEngine) -> None:
    close = getattr(engine, "aclose", None)
    if close is not None:
        await close()


def _bind_engine(engine: StructuredEngine, flow: Any) -> None:
    bind = getattr(engine, "bind", None)
    if bind is not None:
        bind(flow)


__all__ = [
    "ChatStructuredEngine",
    "ConfidenceGate",
    "ContractSource",
    "DEFAULT_OUTPUT_CODEC",
    "FallbackEngine",
    "JsonSchemaCodec",
    "LowConfidenceError",
    "OutputCodec",
    "RetryingEngine",
    "Schema",
    "StructuredEngine",
    "StructuredOutputError",
    "StructuredRequest",
    "StructuredResult",
    "UnsupportedStructuredOutputError",
    "as_finish_value",
    "json_schema_for",
]
