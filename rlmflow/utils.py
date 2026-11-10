"""Small helpers shared across rlmflow: code fences, output truncation, sampling, and host-value encoding."""

from __future__ import annotations

import base64
import re
from typing import Any

import cloudpickle

# Generic readers accept ```repl, ```python, and bare ``` fences.
_OPEN = re.compile(r"```(?:repl|python)?[ \t]*\n")
_REPL_OPEN = re.compile(r"```repl[ \t]*\n")
_REPL_FENCE = re.compile(r"```repl(?![\w-])")
_CLOSE = re.compile(r"\n?```[ \t]*(?:\n|$)")


def _find_blocks(
    text: str,
    opening_pattern: re.Pattern[str],
    *,
    greedy_close: bool = True,
) -> list[str]:
    """Extract blocks opened by ``opening_pattern`` from a model reply.

    Generic readers may choose the closing fence greedily so markdown fences
    inside Python strings do not prematurely truncate the block. Strict REPL
    parsing uses the first close so multiple action blocks remain detectable.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        opening = opening_pattern.search(text, pos)
        if not opening:
            break
        start = opening.end()
        closing = None
        for match in _CLOSE.finditer(text, start):
            closing = match
            if not greedy_close:
                break
        if closing is None:
            break
        blocks.append(text[start : closing.start()].strip())
        pos = closing.end()
    return blocks


def find_code_blocks(text: str) -> list[str]:
    """Extract generic fenced Python/code blocks from text."""
    return _find_blocks(text, _OPEN)


def find_repl_blocks(text: str) -> list[str]:
    """Extract only explicit ``repl`` blocks used as executable agent actions."""
    return _find_blocks(text, _REPL_OPEN, greedy_close=False)


def has_repl_fence(text: str) -> bool:
    """Whether ``text`` opens a ``repl`` fence at all, even an unclosed or empty one."""
    return _REPL_FENCE.search(text) is not None


def encode_host_value(value: Any) -> str:
    """Copy one host-controlled value into a worker payload."""
    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")


def decode_host_value(blob: str) -> Any:
    """Decode a host-created payload inside an isolated worker."""
    return cloudpickle.loads(base64.b64decode(blob))


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
    "decode_host_value",
    "encode_host_value",
    "find_code_blocks",
    "find_repl_blocks",
    "has_repl_fence",
    "repl_code_block",
    "sampling_kwargs",
    "truncate_output",
]
