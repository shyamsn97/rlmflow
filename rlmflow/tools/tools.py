"""Tool metadata and prompt formatting."""

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
    #: ``(flow, node) -> bool``. ``None`` means always visible. Evaluated for
    #: both the prompt and the REPL binding.
    when: Callable[..., bool] | None = None


def _default_tool_name(name: str) -> str:
    return name.removeprefix("tool_")


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
    when: Callable[..., bool] | None = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool. It defaults
    to auto-detection from the function (``async def`` -> ``True``); pass it
    explicitly only for a proxy that returns a coroutine without being
    ``async def``. (``async`` is a keyword, hence ``is_async``.)

    ``when`` is ``(flow, node) -> bool``. When it returns false, the tool is
    omitted from both the prompt and the REPL for that agent.
    """

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
            when=when,
        )
        return fn

    return decorator


def get_tool_metadata(fn: Any) -> ToolMetadata | None:
    target = getattr(fn, "__func__", fn)
    return getattr(target, "_tool_meta", None)


def tool_allowed(fn: Any, flow: Any = None, node: Any = None) -> bool:
    """Whether ``fn``'s ``when`` predicate allows it on this node."""
    meta = get_tool_metadata(fn)
    if meta is None or meta.when is None:
        return True
    return bool(meta.when(flow, node))


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("rlmflow.tools.agents.", "")
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


def format_tool_docs(tools: Mapping[str, Any]) -> str:
    """Render one ``### Tools`` list for the callables an agent has bound."""
    lines = [line for value in tools.values() if (line := format_tool_line(value))]
    return "\n".join(["### Tools", *lines]) if lines else ""


__all__ = [
    "ToolMetadata",
    "format_tool_docs",
    "get_tool_metadata",
    "tool",
    "tool_allowed",
]
