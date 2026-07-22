"""Built-in file tools for the minimal REPL."""

from __future__ import annotations

import re
from pathlib import Path

from rlmflow.tools.tools import tool


@tool("Read a file and return its contents.")
def read_file(path: str) -> str:
    return Path(path).read_text()


@tool("Write content to a file, creating directories if needed.")
def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


@tool("Append content to a file.")
def append_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {path}"


@tool(
    "Find-and-replace edits. Each edit is (old, new). Each `old` must match "
    "exactly once (add surrounding lines to disambiguate) unless replace_all=True. "
    "Edits apply atomically: if any anchor is missing or ambiguous, nothing is written."
)
def edit_file(path: str, *edits: tuple[str, str], replace_all: bool = False) -> str:
    p = Path(path)
    text = p.read_text()
    for i, (old, new) in enumerate(edits):
        n = text.count(old)
        if n == 0:
            raise ValueError(f"edit {i}: anchor not found in {path}: {old!r}")
        if n > 1 and not replace_all:
            raise ValueError(
                f"edit {i}: anchor appears {n}x in {path}; add surrounding lines "
                f"to make it unique, or pass replace_all=True: {old!r}"
            )
        text = text.replace(old, new)
    p.write_text(text)
    return f"Applied {len(edits)} edits to {path}"


def _display_path(path: Path, *, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


@tool(
    "List files and directories, returning paths usable by other file tools. "
    "For relative inputs, returns workspace-relative paths."
)
def ls(path: str = ".") -> list[str]:
    absolute = Path(path).is_absolute()
    p = Path(path).resolve()
    if p.is_file():
        return [_display_path(p, absolute=absolute)]
    return sorted(_display_path(entry, absolute=absolute) for entry in p.iterdir())


@tool("Read lines start:end (0-indexed, exclusive) from a file.")
def read_lines(path: str, start: int, end: int) -> str:
    return "\n".join(Path(path).read_text().splitlines()[start:end])


@tool("Count the number of lines in a file.")
def line_count(path: str) -> int:
    return len(Path(path).read_text().splitlines())


@tool("Search for lines matching a regex pattern.")
def grep(pattern: str, path: str = ".", *, max_results: int = 50) -> str:
    p = Path(path)
    regex = re.compile(pattern)
    matches: list[str] = []
    files = [p] if p.is_file() else sorted(p.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{f}:{i}: {line}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)
        except (UnicodeDecodeError, PermissionError):
            continue
    return "\n".join(matches)


FILE_TOOLS = [
    read_file,
    write_file,
    append_file,
    edit_file,
    ls,
    read_lines,
    line_count,
    grep,
]


__all__ = [
    "FILE_TOOLS",
    "append_file",
    "edit_file",
    "grep",
    "line_count",
    "ls",
    "read_file",
    "read_lines",
    "write_file",
]
