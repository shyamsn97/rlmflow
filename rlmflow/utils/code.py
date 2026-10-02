"""Minimal helpers for reading agent-emitted code blocks."""

from __future__ import annotations

import re

# Generic readers accept ```repl, ```python, and bare ``` fences.
_OPEN = re.compile(r"```(?:repl|python)?[ \t]*\n")
_REPL_OPEN = re.compile(r"```repl[ \t]*\n")
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


__all__ = ["find_code_blocks", "find_repl_blocks"]
