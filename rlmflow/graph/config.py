"""AgentRoot arguments: ``Input``, walking argument trees, and the limits on names and sizes."""

from __future__ import annotations

import keyword
from collections.abc import Callable
from typing import Any

# Return this from a callback to visit a list or dictionary's children.
RECURSE = object()
ArgumentCallback = Callable[[str, Any], Any]


def transform_argument(value: Any, transform: ArgumentCallback, *, path: str) -> Any:
    """Transform an argument tree in pre-order, recursing on ``RECURSE``."""
    return _traverse(value, transform, path=path, rebuild=True)


def visit_argument(value: Any, visitor: ArgumentCallback, *, path: str) -> None:
    """Visit an argument tree in pre-order, recursing on ``RECURSE``."""
    _traverse(value, visitor, path=path, rebuild=False)


def _traverse(value: Any, callback: ArgumentCallback, *, path: str, rebuild: bool) -> Any:
    result = callback(path, value)
    if result is not RECURSE:
        return result
    if isinstance(value, list):
        mapped = [
            _traverse(item, callback, path=f"{path}[{index}]", rebuild=rebuild)
            for index, item in enumerate(value)
        ]
        return mapped if rebuild else value
    if isinstance(value, dict):
        rebuilt: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"argument {path!r} contains a non-string object key")
            rebuilt[key] = _traverse(item, callback, path=f"{path}.{key}", rebuild=rebuild)
        return rebuilt if rebuild else value
    return value


DEFAULT_QUERY = "Complete the requested task."
INPUT_VALUE_KEY = "$rlmflow_input"
INPUT_DESCRIPTION_KEY = "$rlmflow_description"
_INPUT_MARKER_KEY = "$rlmflow_input_marker"
_INPUT_MARKER_VALUE = "v1"
#: What a node's printed output is cut to; official RLM truncates each REPL block at 20K characters.
MAX_OUTPUT = 20_000
#: The longest task an agent may launch another with.
MAX_TASK = 20_000
DEFAULT_MAX_ITERS = 30


class Input(dict[str, Any]):
    """A task argument paired with caller-supplied model context."""

    def __init__(self, value: Any, *, description: str) -> None:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Input description must be a non-empty string")
        super().__init__(
            {
                _INPUT_MARKER_KEY: _INPUT_MARKER_VALUE,
                INPUT_VALUE_KEY: value,
                INPUT_DESCRIPTION_KEY: description.strip(),
            }
        )

    @property
    def value(self) -> Any:
        return self[INPUT_VALUE_KEY]

    @property
    def description(self) -> str:
        return self[INPUT_DESCRIPTION_KEY]


def input_parts(value: object) -> tuple[Any, str] | None:
    """Return an Input marker's value and description."""
    if (
        isinstance(value, dict)
        and set(value) == {_INPUT_MARKER_KEY, INPUT_VALUE_KEY, INPUT_DESCRIPTION_KEY}
        and value.get(_INPUT_MARKER_KEY) == _INPUT_MARKER_VALUE
        and isinstance(value.get(INPUT_DESCRIPTION_KEY), str)
        and value[INPUT_DESCRIPTION_KEY].strip()
    ):
        return value[INPUT_VALUE_KEY], value[INPUT_DESCRIPTION_KEY]
    return None


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid agent name {name!r}")


def validate_argument_name(name: str) -> None:
    """Require invocation arguments that can be bound as Python variables."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"invalid invocation argument name {name!r}")
    if name == "task":
        raise ValueError("'task' is the invocation task and cannot be passed twice")


__all__ = [
    "DEFAULT_MAX_ITERS",
    "DEFAULT_QUERY",
    "MAX_OUTPUT",
    "MAX_TASK",
    "Input",
    "validate_agent_name",
    "validate_argument_name",
]
