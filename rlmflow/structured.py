"""Minimal structured-output helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel, TypeAdapter, ValidationError

Schema = type[BaseModel] | TypeAdapter[Any] | Mapping[str, Any] | str


def json_schema_for(schema: Schema) -> dict[str, Any]:
    if isinstance(schema, Mapping | str):
        json_schema = dict(_load_json_schema(schema))
    else:
        json_schema = _json_schema_for_pydantic(schema)
    jsonschema.validators.validator_for(json_schema).check_schema(json_schema)
    return json_schema


class StructuredOutputError(ValueError):
    def __init__(self, *, content: str, schema: Schema, cause: Exception) -> None:
        self.content = content
        self.schema = schema
        self.cause = cause
        super().__init__(_format_error_message(content, schema, cause))


def system_prompt_hint(schema: Schema) -> str:
    return json.dumps(json_schema_for(schema), indent=2)


def parse_structured_output(content: str, schema: Schema) -> Any:
    """Validate JSON content against a Pydantic or JSON schema."""
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


def parse_structured_answer(answer: object, schema: Schema) -> Any:
    """Validate an answer given as a Python value or as pre-encoded JSON text."""
    if not isinstance(answer, str):
        try:
            content = json.dumps(answer, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(content=repr(answer), schema=schema, cause=exc) from exc
        return parse_structured_output(content, schema)
    try:
        return parse_structured_output(answer, schema)
    except StructuredOutputError:
        # A bare string such as NO_MATCH is a value, not a JSON document.
        return parse_structured_output(json.dumps(answer), schema)


class StructuredOutputParser:
    """Compatibility shim for the original public parser object."""

    system_prompt_hint = staticmethod(system_prompt_hint)
    __call__ = staticmethod(parse_structured_output)


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
        "Hint: call finish(value) with a JSON-compatible Python value that matches "
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
    "Schema",
    "StructuredOutputError",
    "StructuredOutputParser",
    "json_schema_for",
    "parse_structured_answer",
    "parse_structured_output",
    "system_prompt_hint",
]
