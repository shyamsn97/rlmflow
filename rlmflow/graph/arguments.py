"""Traverse nested agent arguments while preserving their paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Return this from a callback to visit a list or dictionary's children.
RECURSE = object()
ArgumentCallback = Callable[[str, Any], Any]


def transform_argument(
    value: Any,
    transform: ArgumentCallback,
    *,
    path: str,
) -> Any:
    """Transform an argument tree in pre-order, recursing on ``RECURSE``."""
    return _traverse(value, transform, path=path, rebuild=True)


def visit_argument(value: Any, visitor: ArgumentCallback, *, path: str) -> None:
    """Visit an argument tree in pre-order, recursing on ``RECURSE``."""
    _traverse(value, visitor, path=path, rebuild=False)


def _traverse(
    value: Any,
    callback: ArgumentCallback,
    *,
    path: str,
    rebuild: bool,
) -> Any:
    """Shared recursive implementation for argument mapping and inspection."""
    result = callback(path, value)
    if result is not RECURSE:
        return result
    if isinstance(value, list):
        if rebuild:
            return [
                _traverse(
                    item,
                    callback,
                    path=f"{path}[{index}]",
                    rebuild=True,
                )
                for index, item in enumerate(value)
            ]
        for index, item in enumerate(value):
            _traverse(item, callback, path=f"{path}[{index}]", rebuild=False)
        return value
    if isinstance(value, dict):
        result: dict[str, Any] | None = {} if rebuild else None
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"argument {path!r} contains a non-string object key")
            mapped_item = _traverse(
                item,
                callback,
                path=f"{path}.{key}",
                rebuild=rebuild,
            )
            if result is not None:
                result[key] = mapped_item
        return result if result is not None else value
    return value
