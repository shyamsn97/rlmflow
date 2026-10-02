"""Small, stateless helpers shared across the package."""

from __future__ import annotations

from typing import Any

from rlmflow.tools import get_tool_metadata
from rlmflow.utils.code import find_code_blocks, find_repl_blocks


def code_block(text: str) -> str:
    """Return the first fenced code block in ``text`` (or ``""`` if none)."""
    blocks = find_code_blocks(text)
    return blocks[0] if blocks else ""


def repl_code_block(text: str) -> str:
    """Return the sole non-empty ``repl`` action block, or ``""`` if invalid."""
    blocks = find_repl_blocks(text)
    if len(blocks) != 1 or not blocks[0]:
        return ""
    return blocks[0]


def tool_name(fn: Any) -> str:
    """Registered name of a tool callable, falling back to ``fn.__name__``."""
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


def truncate_output(output: str, limit: int) -> str:
    """Keep the start and end of a long observation (0 disables truncation)."""
    if limit and len(output) > limit:
        omitted = len(output) - limit
        head = limit // 2
        tail = limit - head
        return (
            output[:head]
            + f"\n...<truncated {omitted} chars; keep full data in variables>...\n"
            + output[-tail:]
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
    "code_block",
    "repl_code_block",
    "sampling_kwargs",
    "tool_name",
    "truncate_output",
]
