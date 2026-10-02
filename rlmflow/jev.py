"""Optional TypeSafe Jev structured-decision engine."""

from __future__ import annotations

import json
from typing import Any

from rlmflow.structured import OutputCodec, as_finish_value
from rlmflow.structured_engines import StructuredRequest, StructuredResult


class JevEngine:
    """Produce compatible structured decisions with ``jev_jsonschema``.

    The dependency is imported lazily; install ``rlmflow[jev]`` to use the
    default client. A compatible client may be injected for testing or custom
    transport policy.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = "jev-latest",
    ) -> None:
        self.client = client
        self.model = model
        self._owned_client: Any = None
        self._plans: dict[str, Any] = {}

    def _converter(self) -> Any:
        if self.client is not None and hasattr(self.client, "convert"):
            return self.client
        try:
            from jev_jsonschema import JSONSchema2Jev
        except ImportError as exc:
            raise ImportError(
                "JevEngine requires the optional dependency; install rlmflow[jev]"
            ) from exc
        return JSONSchema2Jev()

    @staticmethod
    def _key(schema: dict[str, Any]) -> str:
        return json.dumps(schema, sort_keys=True, separators=(",", ":"))

    def _plan(self, schema: dict[str, Any]) -> Any:
        key = self._key(schema)
        if key not in self._plans:
            self._plans[key] = self._converter().convert(schema)
        return self._plans[key]

    def supports(
        self,
        request: StructuredRequest | dict[str, Any],
    ) -> tuple[str, ...]:
        schema = request.schema if isinstance(request, StructuredRequest) else request
        try:
            self._plan(schema)
        except ImportError as exc:
            return (str(exc),)
        except Exception as exc:  # noqa: BLE001 - optional clients define their error type
            failures = getattr(exc, "failures", None)
            if failures:
                return tuple(
                    f"{getattr(failure, 'key', '?')}: "
                    f"{getattr(failure, 'reason', failure)}"
                    for failure in failures
                )
            return (str(exc),)
        return ()

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._owned_client is None:
            try:
                from jev_jsonschema import AsyncJevClient
            except ImportError as exc:
                raise ImportError(
                    "JevEngine requires the optional dependency; install rlmflow[jev]"
                ) from exc
            self._owned_client = AsyncJevClient()
        return self._owned_client

    async def generate(
        self,
        request: StructuredRequest,
        codec: OutputCodec,
    ) -> StructuredResult:
        client = self._client()
        result = await client.ask(
            self._plan(request.schema),
            state=request.state,
            model=request.model or self.model,
        )
        value = codec.validate(result.output, request.schema)
        usage = result.usage
        response = getattr(result, "response", None)
        resolved_model = getattr(response, "model", request.model or self.model)
        metadata = {
            "engine": "jev",
            "model": resolved_model,
            "probabilities": result.probabilities,
            "confidence": result.confidence,
            "usage": {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
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


__all__ = ["JevEngine"]
