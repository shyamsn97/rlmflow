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
    #: ``(flow, node) -> bool``. ``None`` means the tool is visible whenever its
    #: toolset includes it. Evaluated for both the prompt and the REPL binding.
    when: Callable[..., bool] | None = None


@dataclass(frozen=True)
class ToolsetMetadata:
    title: str


def toolset(title: str) -> Callable:
    """Mark ``cls`` as a source of tools, not a tool.

    The REPL namespace is flat — agent code calls ``grep(...)``, not
    ``FileTools.grep(...)``. ``Flow.add_tool`` flattens ``@tool`` methods and
    renders their signatures plus optional toolset ``guidance(flow, node)``.
    """

    def decorator(cls: type) -> type:
        cls._toolset_meta = ToolsetMetadata(title=title.strip())
        return cls

    return decorator


def _default_tool_name(name: str) -> str:
    return name.removeprefix("tool_")


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
    inject: bool = True,
    when: Callable[..., bool] | None = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool. It defaults
    to auto-detection from the function (``async def`` -> ``True``); pass it
    explicitly only for a proxy that returns a coroutine without being
    ``async def``. (``async`` is a keyword, hence ``is_async``.)

    ``inject=False`` keeps the method as a prompt signature without binding it
    into the REPL. Reserved builtins use this: ``finish`` is a per-node closure
    that Flow already injects under the same name.

    ``when`` is ``(flow, node) -> bool``. When it returns false, the tool is
    omitted from both the prompt and the REPL for that agent. Toolsets may
    also implement ``visible_tools(flow, node)`` to name the whole set.
    """

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
            inject=inject,
            when=when,
        )
        return fn

    return decorator


def tool_allowed(fn: Any, flow: Any = None, node: Any = None) -> bool:
    """Whether ``fn``'s ``when`` predicate allows it on this node."""
    meta = get_tool_metadata(fn)
    if meta is None or meta.when is None:
        return True
    return bool(meta.when(flow, node))


def visible_tool_names(instance: Any, flow: Any = None, node: Any = None) -> set[str]:
    """Names this toolset exposes on ``node``: ``visible_tools`` intersect ``when``."""
    members = toolset_members(instance)
    visible = getattr(instance, "visible_tools", None)
    if callable(visible):
        allowed = set(visible(flow, node))
    else:
        allowed = {name for name, _method in members}
    return {
        name for name, method in members if name in allowed and tool_allowed(method, flow, node)
    }


def host_tools_for(flow: Any, node: Any) -> dict[str, Any]:
    """``flow.tools`` entries that should be bound for this agent."""
    claimed: set[str] = set()
    visible: set[str] = set()
    for instance in flow.toolsets:
        claimed.update(name for name, _method in toolset_members(instance))
        visible.update(visible_tool_names(instance, flow, node))
    host: dict[str, Any] = {}
    for name, value in flow.tools.items():
        if name in claimed:
            if name in visible:
                host[name] = value
            continue
        if get_tool_metadata(value) is None or tool_allowed(value, flow, node):
            host[name] = value
    return host


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
    sig = sig.replace("rlmflow.tools.agents.", "")
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


def _prompt_member(name: str, method: Any, tools: Mapping[str, Any]) -> bool:
    """Signature-only builtins stay in the prompt; injected tools follow ``flow.tools``."""
    meta = get_tool_metadata(method)
    return meta is None or not meta.inject or name in tools


def format_tool_docs(
    toolsets: list[Any],
    tools: Mapping[str, Any],
    *,
    flow: Any = None,
    node: Any = None,
) -> str:
    """Render active tools once, grouped by their owning toolset."""
    claimed: set[str] = set()
    blocks: list[str] = []
    for instance in toolsets:
        meta = get_toolset_metadata(instance)
        if meta is None:
            continue
        members = toolset_members(instance)
        claimed.update(name for name, _method in members)
        allowed = visible_tool_names(instance, flow, node)
        active = [
            method
            for name, method in members
            if name in allowed and _prompt_member(name, method, tools)
        ]
        if not active:
            continue
        lines = [f"### {meta.title}"]
        lines.extend(line for method in active if (line := format_tool_line(method)))
        guidance = getattr(instance, "guidance", None)
        if callable(guidance) and (text := guidance(flow, node).strip()):
            lines.extend(("", text))
        blocks.append("\n".join(lines))

    standalone = [
        line
        for name, value in sorted(tools.items())
        if name not in claimed
        and tool_allowed(value, flow, node)
        and (line := format_tool_line(value))
    ]
    if standalone:
        blocks.append("\n".join(["### Tools", *standalone]))
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
    "ToolMetadata",
    "ToolsetMetadata",
    "format_tool_docs",
    "format_tool_line",
    "get_tool_metadata",
    "get_toolset_metadata",
    "host_tools_for",
    "is_toolset",
    "partition_repl_namespace",
    "tool",
    "tool_allowed",
    "toolset",
    "toolset_members",
    "visible_tool_names",
]
