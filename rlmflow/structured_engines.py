"""Pluggable producers for state-oriented structured queries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import JsonValue

from rlmflow.structured import OutputCodec, Schema, as_finish_value


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
    "FallbackEngine",
    "LowConfidenceError",
    "RetryingEngine",
    "StructuredEngine",
    "StructuredRequest",
    "StructuredResult",
    "UnsupportedStructuredOutputError",
]
