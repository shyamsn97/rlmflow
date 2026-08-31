"""Tool metadata, toolsets, and prompt formatting."""

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
    #: When False the method is a prompt signature only: Flow still injects the
    #: real callable (a reserved closure, a factory) under the same name.
    inject: bool = True


@dataclass(frozen=True)
class ToolsetMetadata:
    title: str
    #: ``"section"`` emits a top-level ``## {title}`` from ``description()``.
    #: ``"tools"`` (default) groups signatures under ``## Tools``.
    placement: str = "tools"


def toolset(title: str, *, placement: str = "tools") -> Callable:
    """Mark ``cls`` as a source of tools, not a tool.

    The REPL namespace is flat — agent code calls ``grep(...)``, not
    ``FileTools.grep(...)``. ``Flow.add_tool`` flattens ``@tool`` methods and
    renders ``description`` / ``example`` into the prompt.
    """

    if placement not in {"section", "tools"}:
        raise ValueError(f"placement must be 'section' or 'tools', not {placement!r}")

    def decorator(cls: type) -> type:
        cls._toolset_meta = ToolsetMetadata(title=title.strip(), placement=placement)
        return cls

    return decorator


@dataclass(frozen=True)
class PromptExample:
    """One worked example a toolset contributes to the ``## Examples`` section."""

    body: str
    priority: int = 50
    title: str = ""


def _default_tool_name(name: str) -> str:
    return name.removeprefix("tool_")


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
    inject: bool = True,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool. It defaults
    to auto-detection from the function (``async def`` -> ``True``); pass it
    explicitly only for a proxy that returns a coroutine without being
    ``async def``. (``async`` is a keyword, hence ``is_async``.)

    ``inject=False`` keeps the method as a prompt signature without binding it
    into the REPL. Reserved builtins use this: ``finish`` is a per-node closure
    that Flow already injects under the same name.
    """

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
            inject=inject,
        )
        return fn

    return decorator


def get_tool_metadata(fn: Any) -> ToolMetadata | None:
    target = getattr(fn, "__func__", fn)
    return getattr(target, "_tool_meta", None)


def get_toolset_metadata(obj: Any) -> ToolsetMetadata | None:
    target = obj if isinstance(obj, type) else type(obj)
    return getattr(target, "_toolset_meta", None)


def is_toolset(obj: Any) -> bool:
    return get_toolset_metadata(obj) is not None


def toolset_members(instance: Any) -> list[tuple[str, Any]]:
    """Bound ``@tool`` methods in class-body order. First version does not walk mixins."""
    members: list[tuple[str, Any]] = []
    for name, value in type(instance).__dict__.items():
        meta = get_tool_metadata(value)
        if meta is None:
            continue
        members.append((meta.name, getattr(instance, name)))
    return members


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError):
        sig = "(...)"
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


def format_tool_docs(
    toolsets: list[Any],
    tools: Mapping[str, Any],
    *,
    flow: Any = None,
    node: Any = None,
) -> str:
    """Render decorated tools once, preserving toolset titles and descriptions."""
    claimed: set[str] = set()
    blocks: list[str] = []
    for instance in toolsets:
        meta = get_toolset_metadata(instance)
        if meta is None:
            continue
        members = toolset_members(instance)
        claimed.update(name for name, _method in members)
        if meta.placement != "tools":
            continue
        description = getattr(instance, "description", None)
        lines = [f"### {meta.title}"]
        if callable(description) and (text := description(flow, node).strip()):
            lines.append(text)
        lines.extend(line for _name, method in members if (line := format_tool_line(method)))
        blocks.append("\n".join(lines))

    standalone = [
        line
        for name, value in sorted(tools.items())
        if name not in claimed and (line := format_tool_line(value))
    ]
    if standalone:
        blocks.append("\n".join(standalone))
    return "\n\n".join(blocks)


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
    "PromptExample",
    "ToolMetadata",
    "ToolsetMetadata",
    "format_tool_docs",
    "format_tool_line",
    "get_tool_metadata",
    "get_toolset_metadata",
    "is_toolset",
    "partition_repl_namespace",
    "tool",
    "toolset",
    "toolset_members",
]
