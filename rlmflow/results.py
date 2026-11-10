"""How a finished child's result is shown to its parent: a variable name and a message."""

from __future__ import annotations

import json
import keyword
import re
import textwrap
from collections.abc import Iterable, Mapping
from typing import Any

#: Longest result text one agent contributes to a result node.
RESULT_MAX = 20_000


def json_safe(value: Any) -> Any:
    """``value`` if it survives JSON, else its string."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return str(value)


def result_name(agent_name: str, taken: Iterable[str] = (), reserved: Iterable[str] = ()) -> str:
    """The variable a result node binds a child's result to: its name as an identifier.

    ``reserved`` names get a leading underscore; ``taken`` ones a numeric suffix.
    """
    name = re.sub(r"\W", "_", agent_name)
    if not name or name[0].isdigit() or keyword.iskeyword(name) or name in set(reserved):
        name = f"_{name}"
    taken = set(taken)
    candidate, index = name, 1
    while candidate in taken:
        candidate, index = f"{name}_{index}", index + 1
    return candidate


def result_marker(path: str) -> str:
    return f"[result {path}]"


def render_results(items: list[tuple[str, str, Any]], limit: int) -> str:
    """``(path, config name, result)`` per finished agent, as one message."""
    parts = ["Agents you started have finished. Each block is that agent's result."]
    for path, config_name, result in items:
        body = _result_body(result)
        if len(body) > limit:
            body = body[: max(limit - 3, 0)] + "..."
        parts.append(f"{result_marker(path)}\nconfig: {config_name}\n{body}")
    return "\n\n".join(parts)


def _result_body(value: Any) -> str:
    """A mapping is its fields; anything else is JSON, labeled ``result``."""
    if isinstance(value, Mapping):
        if not value:
            return "result: {}"
        lines = []
        for key, item in value.items():
            rendered = _jsonish(item)
            lines.append(
                f"{key}:\n{textwrap.indent(rendered, '  ')}"
                if "\n" in rendered
                else f"{key}: {rendered}"
            )
        return "\n".join(lines)
    return f"result: {_jsonish(value)}"


def _jsonish(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    return json.dumps(value, default=str, ensure_ascii=False)


__all__ = ["RESULT_MAX", "json_safe", "render_results", "result_marker", "result_name"]
