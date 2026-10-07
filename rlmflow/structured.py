"""Structured-output contracts, codecs, and compatibility helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
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


__all__ = [
    "ContractSource",
    "DEFAULT_OUTPUT_CODEC",
    "JsonSchemaCodec",
    "OutputCodec",
    "Schema",
    "StructuredOutputError",
    "as_finish_value",
    "json_schema_for",
]
