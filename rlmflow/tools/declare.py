"""Tool declarations, metadata, prompt documentation, and normalization."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from rlmflow.repl import Repl, current_repl
from rlmflow.runtime.repl import (
    AGENT_SEND_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    ToolMetadata,
    get_tool_metadata,
)

Where = Literal["host", "repl"]


def tool(
    description: str | None = None,
    *,
    name: str | None = None,
    where: Where = "host",
    is_async: bool | None = None,
    when: Callable[..., bool] | None = None,
    cache: bool = False,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``description`` defaults to the first paragraph of ``fn``'s docstring.
    ``where="host"`` (the default) runs the tool in this process, where its
    clients and connections live; agent code calls it over the REPL protocol.
    ``where="repl"`` copies it into the agent's REPL (and sandbox), for pure
    functions that should run next to the agent's files. ``is_async`` records
    whether agent code must ``await`` the tool; it defaults to ``async def``
    detection. ``when`` is ``(repl) -> bool``, given the run's ``Repl``: when
    false, the tool is left out of the prompt and the REPL.

    ``cache=True`` declares a host tool's arguments and result JSON: each call is
    saved on the node that made it, keyed by its arguments, so running the node
    again (after an edit, or to rebuild its REPL) returns the saved result for
    the same call instead of calling the tool twice. Other tools always run.
    """
    if where not in ("host", "repl"):
        raise ValueError(f"tool where must be 'host' or 'repl', not {where!r}")
    if cache and where != "host":
        raise ValueError("only host tools can cache their results")

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        text = description if description is not None else _first_paragraph(fn.__doc__)
        fn._tool_meta = ToolMetadata(
            name=name or fn.__name__.removeprefix("tool_"),
            description=text.strip(),
            proxy=where == "host",
            is_async=resolved,
            when=when,
            cache=cache,
        )
        return fn

    return decorator


def _first_paragraph(doc: str | None) -> str:
    return inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ") if doc else ""


def as_tool(fn: Any) -> Any:
    """``fn`` as a tool: a plain function gets metadata from its docstring and runs on the host."""
    if get_tool_metadata(fn) is not None or not inspect.isroutine(fn):
        return fn
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

    return tool()(wrapper)


_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_type(prop: Mapping[str, Any]) -> str | None:
    kind = prop.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), None)
    return kind if isinstance(kind, str) else None


def tool_from_schema(
    name: str,
    description: str,
    parameters: Mapping[str, Any] | None,
    call: Callable[[dict[str, Any]], Any],
    *,
    is_async: bool | None = None,
    where: Where = "host",
    when: Callable[..., bool] | None = None,
) -> Callable:
    """A tool whose arguments are described by a JSON Schema object."""
    parameters = parameters or {}
    properties: Mapping[str, Any] = parameters.get("properties") or {}
    required = set(parameters.get("required") or [])
    ordered = [
        *(p for p in properties if p in required),
        *(p for p in properties if p not in required),
    ]
    signature = inspect.Signature(
        [
            inspect.Parameter(
                prop,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty
                if prop in required
                else properties[prop].get("default"),
                annotation=_JSON_TYPES.get(
                    _json_type(properties[prop]) or "", inspect.Parameter.empty
                ),
            )
            for prop in ordered
        ]
    )
    notes = [
        f"`{prop}`: {properties[prop]['description']}"
        for prop in ordered
        if properties[prop].get("description")
    ]
    text = description.strip() + (" Arguments: " + "; ".join(notes) + "." if notes else "")

    if inspect.iscoroutinefunction(call) if is_async is None else is_async:

        async def fn(*args: Any, **kwargs: Any) -> Any:
            return await call(dict(signature.bind(*args, **kwargs).arguments))

    else:

        def fn(*args: Any, **kwargs: Any) -> Any:
            return call(dict(signature.bind(*args, **kwargs).arguments))

    fn.__name__ = fn.__qualname__ = name
    fn.__signature__ = signature  # type: ignore[attr-defined]
    return tool(text, name=name, where=where, when=when)(fn)


@runtime_checkable
class Toolkit(Protocol):
    """Tools that need a live resource, such as a server connection."""

    async def open(self) -> Iterable[Any]: ...

    async def close(self) -> None: ...


def tool_allowed(fn: Any, repl: Repl | None = None) -> bool:
    meta = get_tool_metadata(fn)
    return meta is None or meta.when is None or bool(meta.when(repl))


def at_leaf(repl: Repl | None = None) -> bool:
    """True when the run can't launch runs; pass as ``@tool(..., when=at_leaf)``."""
    repl = repl or current_repl()
    return repl is None or not repl.can_launch()


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError, NameError):
        sig = "(...)"
    sig = (
        sig.replace("rlmflow.agent_tools.view.", "")
        .replace("rlmflow.tools.agents.", "")
        .replace("AgentHandle", "Agent")
    )
    prefix = "await " if meta.is_async else ""
    line = f"- `{prefix}{meta.name}{sig}`"
    return f"{line}: {meta.description}" if meta.description else line


def format_tool_docs(tools: Mapping[str, Any], header: str = "### Tools") -> str:
    """One list, under ``header``, of the callables an agent has bound."""
    lines = [line for value in tools.values() if (line := format_tool_line(value))]
    return "\n".join([header, *lines]) if lines else ""


def tool_name(fn: Any) -> str:
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


RESERVED_TOOLS = frozenset(
    {
        "submit",
        "launch_agent",
        "llm_query",
        "Input",
        "NotYourAgent",
        "asyncio",
        "ENV",
        "AGENTS",
        AGENTS_BINDING,
        AGENT_WAIT_TOOL,
        AGENT_SEND_TOOL,
    }
)


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a flat list; one level of nesting is flattened."""
    if tools is None:
        return []
    if callable(tools) or isinstance(tools, Toolkit):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        items: list[Any] = []
        for item in tools:
            if (
                not callable(item)
                and isinstance(item, Iterable)
                and not isinstance(item, (str, bytes))
            ):
                items.extend(item)
            else:
                items.append(item)
        return items
    raise TypeError(f"tools must be a tool or a sequence of tools, not {type(tools).__name__}")
