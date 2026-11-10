"""JSON values and the one schema check: ``output_schema`` on ``submit`` and ``llm_query``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import jsonschema
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

_JSON_VALUE = TypeAdapter(JsonValue)


def json_schema_for(schema: Any) -> dict[str, Any]:
    """A JSON Schema dict from a dict, a JSON string, a Pydantic model or a ``TypeAdapter``."""
    if isinstance(schema, str):
        schema = json.loads(schema)
    if isinstance(schema, Mapping):
        normalized = json.loads(json.dumps(dict(schema)))
    elif isinstance(schema, TypeAdapter):
        normalized = schema.json_schema()
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        normalized = schema.model_json_schema()
    else:
        raise TypeError(
            "schema must be a JSON Schema dict or string, a Pydantic model, or a TypeAdapter"
        )
    jsonschema.validators.validator_for(normalized).check_schema(normalized)
    return normalized


def as_json_value(value: object) -> JsonValue:
    """Return a strict JSON value or raise ``TypeError``."""
    try:
        checked = _JSON_VALUE.validate_python(value)
        json.dumps(checked, allow_nan=False)
    except (TypeError, ValueError, ValidationError) as exc:
        raise TypeError(
            "expected a JSON value: strings, finite numbers, booleans, None, lists, "
            "or dictionaries with string keys"
        ) from exc
    return checked


class SchemaError(ValueError):
    """A value that doesn't match an agent's ``output_schema``."""


def check(value: object, schema: dict[str, Any]) -> Any:
    """Validate a JSON value against ``schema``; a JSON string is decoded first."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if decoded is not value and _valid(decoded, schema):
            return decoded
    value = as_json_value(value)
    try:
        jsonschema.validate(instance=value, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaError(
            "The value doesn't match the required output schema.\n"
            f"Expected JSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Received:\n{json.dumps(value)[:2000]}\n\n"
            f"Validation error: {exc.message}"
        ) from exc
    return value


def parse(text: str, schema: dict[str, Any]) -> Any:
    """Parse a model's JSON reply (fences allowed) and validate it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"reply is not valid JSON: {exc}") from exc
    return check(value, schema)


def contract(schema: dict[str, Any], *, source: str = "submit") -> str:
    """The prompt text that tells a model the required output shape."""
    hint = json.dumps(schema, indent=2)
    if source == "query":
        return f"Return only JSON matching this schema:\n{hint}"
    return (
        "This run requires structured output. When complete, call `submit(value)` with a "
        "JSON-compatible Python value matching this JSON Schema exactly — the value itself, "
        f"not a status string:\n\n```json\n{hint}\n```"
    )


def _valid(value: Any, schema: dict[str, Any]) -> bool:
    try:
        jsonschema.validate(instance=value, schema=schema)
    except jsonschema.ValidationError:
        return False
    return True


__all__ = ["SchemaError", "as_json_value", "check", "contract", "json_schema_for", "parse"]
