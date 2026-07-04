"""Static checks on agent-emitted code blocks.

The REPL supports normal top-level Python ``await``. The checks below only keep
private graph transport calls out of agent code and make the public graph
launcher's missing ``await`` recoverable before execution.
"""

from __future__ import annotations

import ast
import re

# Opener for *reading* code blocks: ```repl / ```python / bare ```.
# (Broader than the repl-only opener used for editing, below.)
_OPEN = re.compile(r"```(?:repl|python)?[ \t]*\n")
_CLOSE = re.compile(r"\n?```[ \t]*(?:\n|$)")
# Opener for *editing* code blocks — repl fences only, so injection/replay
# edits never rewrite an incidental ```python/```bash block.
_REPL_OPEN = re.compile(r"```repl[ \t]*\n")


def find_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks (```repl / ```python / bare ```).

    Greedy per block: the *last* closing fence after an opener wins, so markdown
    fences embedded inside a Python string (e.g. ``\"\"\"```bash ... ```\"\"\"``)
    don't prematurely close the block.
    """
    blocks: list[str] = []
    pos = 0
    while True:
        opening = _OPEN.search(text, pos)
        if not opening:
            break
        start = opening.end()
        last = None
        for m in _CLOSE.finditer(text, start):
            last = m
        if last is None:
            break
        blocks.append(text[start : last.start()].strip())
        pos = last.end()
    return blocks


def replace_code_block(text: str, new_code: str) -> str:
    """Keep text up to the first ```repl block, replacing its body with ``new_code``.

    Repl-specific on purpose: trajectory edits (inject/replace) target the
    agent's ``repl`` action block, not incidental fences.
    """
    opening = _REPL_OPEN.search(text)
    if not opening:
        return text
    start = opening.end()
    last = None
    for m in _CLOSE.finditer(text, start):
        last = m
    if last is None:
        return text
    return text[: opening.start()] + f"```repl\n{new_code}\n```"


# Public graph-aware await surface. The old delegate/wait primitives are rejected
# explicitly so stale agent code gets a clear recovery message.
_PUBLIC_AWAITABLE_CALLS = {"launch_subagents"}
_INTERNAL_CONTROL_CALLS = {"flow_wait", "flow_delegate"}


def check_wait_syntax(code: str) -> str | None:
    """Return an error string for unsupported ``await`` usage, else ``None``.

    A genuine ``SyntaxError`` returns ``None`` here — that's reported through
    the normal execution path with the interpreter's own message.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    checker = _WaitSyntaxChecker()
    checker.visit(tree)
    return "ERROR: " + "; ".join(checker.errors) if checker.errors else None


def _is_awaitable_call(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _PUBLIC_AWAITABLE_CALLS
    )


def _call_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


class _WaitSyntaxChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.await_depth = 0
        self.errors: list[str] = []

    def _add(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", None)
        prefix = f"Line {line}: " if line is not None else ""
        self.errors.append(prefix + message)

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802
        name = _call_name(node.value)
        if name in _INTERNAL_CONTROL_CALLS:
            self._add(node, f"`{name}(...)` is internal; use `launch_subagents(...)`")
        self.await_depth += 1
        self.generic_visit(node)
        self.await_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if _is_awaitable_call(node) and self.await_depth == 0:
            name = node.func.id  # type: ignore[union-attr]
            self._add(node, f"`{name}(...)` must be awaited: `await {name}(...)`")
        name = _call_name(node)
        if name in _INTERNAL_CONTROL_CALLS:
            self._add(node, f"`{name}(...)` is internal; use `launch_subagents(...)`")
        self.generic_visit(node)


__all__ = ["check_wait_syntax", "find_code_blocks", "replace_code_block"]
