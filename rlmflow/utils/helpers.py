"""Small, stateless helpers shared across the package."""

from __future__ import annotations

import inspect
from typing import Any

from rlmflow.graph.nodes import AgentStart, LLMUsage, start
from rlmflow.tools import get_tool_metadata
from rlmflow.utils.code import find_code_blocks


def code_block(text: str) -> str:
    """Return the first fenced code block in ``text`` (or ``""`` if none)."""
    blocks = find_code_blocks(text)
    return blocks[0] if blocks else ""


def accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` accepts keyword ``name`` (explicitly or via ``**kwargs``).

    Used to push ``timeout`` only to clients that take it, so lean fake/test
    clients are never handed an unexpected kwarg.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True  # can't introspect (e.g. a C builtin); assume it accepts it
    return any(
        p.kind is p.VAR_KEYWORD
        or (p.name == name and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        for p in params
    )


def tool_name(fn: Any) -> str:
    """Registered name of a tool callable, falling back to ``fn.__name__``."""
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


def node_from_input(node_or_query: AgentStart | str) -> AgentStart:
    """Coerce a query string into a fresh root node; pass nodes through.

    Pure coercion for batch entry points (``parallel_run``/``parallel_stream``)
    that accept either shape. State resolution (inputs, output_schema, new
    turns) lives in :meth:`Flow.resolve_run`, not here.
    """
    return (
        node_or_query if isinstance(node_or_query, AgentStart) else start(node_or_query)
    )


def usage_from_client(client: Any) -> LLMUsage:
    """Read token usage off an LLM client's ``last_usage`` (0 if unreported)."""
    usage = getattr(client, "last_usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def truncate_output(output: str, limit: int) -> str:
    """Cap an exec observation at ``limit`` chars (0 disables); note the omission."""
    if limit and len(output) > limit:
        omitted = len(output) - limit
        return output[:limit] + (
            f"\n...<truncated {omitted} chars; keep full data in variables>"
        )
    return output


def sampling_kwargs(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Collect the set (non-``None``) LLM sampling knobs into a kwargs dict."""
    pairs = (
        ("temperature", temperature),
        ("top_p", top_p),
        ("max_tokens", max_tokens),
        ("stop", stop),
    )
    return {key: value for key, value in pairs if value is not None}


__all__ = [
    "accepts_kwarg",
    "code_block",
    "node_from_input",
    "sampling_kwargs",
    "tool_name",
    "truncate_output",
    "usage_from_client",
]
