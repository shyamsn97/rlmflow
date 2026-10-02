"""Optional Instructor-backed structured engine."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from pydantic import BaseModel

from rlmflow.structured import OutputCodec, as_finish_value
from rlmflow.structured_engines import StructuredRequest, StructuredResult


class InstructorEngine:
    """Use an Instructor client for Pydantic-defined structured extraction."""

    def __init__(
        self,
        client: Any = None,
        *,
        provider: str | None = None,
        max_retries: int = 0,
    ) -> None:
        if client is None and provider is None:
            raise ValueError("InstructorEngine requires client= or provider=")
        self.client = client
        self.provider = provider
        self.max_retries = max_retries
        self._owned_client: Any = None

    def _response_model(self, request: StructuredRequest) -> type[BaseModel] | None:
        schema = request.source_schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema
        return None

    def supports(self, request: StructuredRequest) -> tuple[str, ...]:
        if self._response_model(request) is None:
            return (
                "InstructorEngine requires output_schema to be a Pydantic "
                "BaseModel class; canonical JSON Schema cannot be losslessly "
                "converted back into a response model",
            )
        return ()

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            try:
                import instructor
            except ImportError as exc:
                raise ImportError(
                    "InstructorEngine requires the optional dependency; "
                    "install rlmflow[instructor]"
                ) from exc
            self._owned_client = instructor.from_provider(
                self.provider,
                async_client=True,
            )
        return self._owned_client

    @staticmethod
    def _create(client: Any) -> Any:
        create = getattr(client, "create", None)
        if create is not None:
            return create
        return client.chat.completions.create

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        response_model = self._response_model(request)
        if response_model is None:
            raise ValueError(self.supports(request)[0])
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(as_finish_value(request.state), ensure_ascii=False)
        )
        content = (
            f"{request.instruction.strip()}\n\n{state}"
            if request.instruction
            else state
        )
        create = self._create(self._client())
        kwargs = {
            "response_model": response_model,
            "messages": [{"role": "user", "content": content}],
            "max_retries": self.max_retries,
            **dict(request.options),
        }
        if request.model is not None:
            kwargs["model"] = request.model
        if inspect.iscoroutinefunction(create):
            result = await create(**kwargs)
        else:
            result = await asyncio.to_thread(create, **kwargs)
        value = result.model_dump(mode="json")
        value = codec.validate(value, request.schema)
        raw = getattr(result, "_raw_response", None)
        usage = getattr(raw, "usage", None)
        metadata = {
            "engine": "instructor",
            "model": request.model or self.provider or "instructor-client",
            "usage": {
                "input_tokens": int(
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "input_tokens", 0)
                    or 0
                ),
                "output_tokens": int(
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", 0)
                    or 0
                ),
            },
        }
        return StructuredResult(
            value=as_finish_value(value),
            metadata=as_finish_value(metadata),
        )

    async def aclose(self) -> None:
        if self._owned_client is None:
            return
        close = getattr(self._owned_client, "aclose", None)
        if close is not None:
            await close()
        self._owned_client = None


__all__ = ["InstructorEngine"]
