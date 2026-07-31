"""Minimal tool metadata and prompt formatting."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    proxy: bool = False
    is_async: bool = False


def _default_tool_name(name: str) -> str:
    return name.removeprefix("tool_")


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool. It defaults
    to auto-detection from the function (``async def`` -> ``True``); pass it
    explicitly only for a proxy that returns a coroutine without being
    ``async def``. (``async`` is a keyword, hence ``is_async``.)
    """

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
        )
        return fn

    return decorator


def get_tool_metadata(fn: Any) -> ToolMetadata | None:
    target = getattr(fn, "__func__", fn)
    return getattr(target, "_tool_meta", None)


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = "(...)"
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


def partition_repl_namespace(
    namespace: Mapping[str, Any],
    *,
    hidden_names: frozenset[str] | set[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    visible: dict[str, Any] = {}
    hidden: dict[str, Any] = {}
    for name, value in namespace.items():
        if name.startswith("_") or name == "SHOW_VARS" or not callable(value):
            continue
        if name in hidden_names:
            hidden[name] = value
        else:
            visible[name] = value
    return visible, hidden


__all__ = [
    "ToolMetadata",
    "format_tool_line",
    "get_tool_metadata",
    "partition_repl_namespace",
    "tool",
]
