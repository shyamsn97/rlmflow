"""Minimal helpers for reading agent-emitted code blocks."""

from __future__ import annotations

import re

# Accept ```repl, ```python, and bare ``` fences when reading model replies.
_OPEN = re.compile(r"```(?:repl|python)?[ \t]*\n")
_CLOSE = re.compile(r"\n?```[ \t]*(?:\n|$)")


def find_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from a model reply.

    The closing fence is chosen greedily within each block so markdown fences
    inside Python strings do not prematurely truncate the action.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        opening = _OPEN.search(text, pos)
        if not opening:
            break
        start = opening.end()
        last = None
        for match in _CLOSE.finditer(text, start):
            last = match
        if last is None:
            break
        blocks.append(text[start : last.start()].strip())
        pos = last.end()
    return blocks


__all__ = ["find_code_blocks"]
