"""Small, stateless helpers shared across the minimal package."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any

from rlmflow.graph import Graph, LLMUsage, Node
from rlmflow.tools import get_tool_metadata
from rlmflow.utils.code import find_code_blocks

ReplKey = tuple[str, str]


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


async def call_sync_or_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function that may be sync or async and return its result.

    Coroutine functions are awaited directly. Plain functions run in a worker
    thread so they never block the event loop; if such a function still returns
    an awaitable (e.g. an async callable object), that is awaited too.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
    return await result if inspect.isawaitable(result) else result


def common_prefix_len(a: list[Node], b: list[Node]) -> int:
    """Length of the shared leading run of nodes, compared by ``node.id``."""
    n = 0
    for x, y in zip(a, b):
        if x.id != y.id:
            break
        n += 1
    return n


def tool_name(fn: Any) -> str:
    """Registered name of a tool callable, falling back to ``fn.__name__``."""
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


def repl_key(graph: Graph) -> ReplKey:
    """Identity of the REPL that backs an agent: ``(graph_id, agent_id)``."""
    return (graph.graph_id, graph.agent_id)


def graph_from_input(graph_or_query: Graph | str) -> Graph:
    """Coerce a query string into a fresh root ``Graph``; pass graphs through.

    Pure coercion for batch entry points (``parallel_run``/``parallel_stream``)
    that accept either shape. State resolution (inputs, output_schema, new
    turns) lives in :meth:`Flow.resolve_run`, not here.
    """
    if isinstance(graph_or_query, Graph):
        return graph_or_query
    return Graph(query=graph_or_query)


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


def iter_budget(depth: int, max_iters: int, child_max_iters: int | None) -> int:
    """Iteration cap for one agent; children may use a tighter bound."""
    if depth > 0 and child_max_iters is not None:
        return child_max_iters
    return max_iters


def budget_exceeded(root: Graph, max_budget: int | None) -> bool:
    """True once the run's total token usage reaches ``max_budget``."""
    return max_budget is not None and root.total_tokens() >= max_budget


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


def llm_output_metadata(model: str, usage: LLMUsage) -> dict[str, Any]:
    """Metadata block recorded on an ``LLMOutput`` node."""
    return {
        "model": model,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }


__all__ = [
    "ReplKey",
    "accepts_kwarg",
    "budget_exceeded",
    "call_sync_or_async",
    "code_block",
    "common_prefix_len",
    "graph_from_input",
    "iter_budget",
    "llm_output_metadata",
    "repl_key",
    "sampling_kwargs",
    "tool_name",
    "truncate_output",
    "usage_from_client",
]
