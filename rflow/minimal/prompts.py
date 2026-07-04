"""Minimal prompt builder and default prompt."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from typing import Any

from rflow.minimal.tools import format_tool_line, partition_repl_namespace

SectionBody = str | Callable[[Any, Any], str]
PROMPT_DOCUMENTED_TOOL_NAMES = frozenset({"done", "launch_subagents"})


class Section:
    def __init__(
        self,
        name: str,
        body: SectionBody = "",
        *,
        title: str | None = None,
        level: int = 2,
    ) -> None:
        self.name = name
        self.body = body
        self.title = title
        self.level = level

    def render(
        self,
        flow: Any = None,
        graph: Any = None,
        body_override: str | None = None,
    ) -> str:
        body = body_override if body_override is not None else self.body
        text = body(flow, graph) if callable(body) else body
        text = textwrap.dedent(text).strip()
        if not text:
            return ""
        if self.title:
            return f"{'#' * max(self.level, 1)} {self.title}\n\n{text}"
        return text


class PromptBuilder:
    def __init__(self) -> None:
        self._sections: list[Section] = []

    def _copy(self) -> "PromptBuilder":
        out = PromptBuilder()
        out._sections = list(self._sections)
        return out

    def section(
        self,
        name: str,
        body: SectionBody = "",
        *,
        title: str | None = None,
        level: int = 2,
        before: str | None = None,
        after: str | None = None,
    ) -> "PromptBuilder":
        out = self._copy()
        new = Section(name, body, title=title, level=level)
        for index, section in enumerate(out._sections):
            if section.name == name:
                out._sections[index] = new
                return out
        if before:
            for index, section in enumerate(out._sections):
                if section.name == before:
                    out._sections.insert(index, new)
                    return out
        if after:
            for index, section in enumerate(out._sections):
                if section.name == after:
                    out._sections.insert(index + 1, new)
                    return out
        out._sections.append(new)
        return out

    def update(self, name: str, body: SectionBody) -> "PromptBuilder":
        out = self._copy()
        for index, section in enumerate(out._sections):
            if section.name == name:
                out._sections[index] = Section(
                    name, body, title=section.title, level=section.level
                )
                return out
        raise KeyError(f"no section named {name!r}; use .section() to add it")

    def remove(self, name: str) -> "PromptBuilder":
        out = self._copy()
        out._sections = [section for section in out._sections if section.name != name]
        return out

    def build(self, flow: Any = None, graph: Any = None, **overrides: str) -> str:
        parts = []
        for section in self._sections:
            rendered = section.render(flow, graph, overrides.get(section.name))
            if rendered.strip():
                parts.append(rendered)
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
        return text + "\n" if text else ""


ROLE_TEXT = """
You are a Recursive Coding Agent: a language model with a prompt, and very
important inputs stored in a Python REPL related to that prompt.

To use the REPL, write code in ```repl``` blocks; the REPL persists across turns.

Available in the REPL:

1. `INPUTS`: a dict of string inputs.
2. `await launch_subagents(specs) -> list`: launch recursive sub-agents. Each spec
   requires `query` and may set `name`, `inputs`, `model`, and `output_schema`.
3. `print(...)`: print concise observations; bare expressions are discarded.
4. `done(answer)`: submit the final answer. If this agent has an output schema,
   pass a JSON-compatible Python value matching it.
"""

STRATEGY_TEXT = """
Inspect inputs before acting. Delegate independent subtasks when useful, then
integrate and verify before calling `done(...)`.
"""

FORMAT_TEXT = """
Execute Python in fenced `repl` blocks. Use exactly one block per assistant
message.
"""

FINAL_TEXT = """
When the task is complete, call `done(answer)` inside a ```repl``` block.
"""

STRUCTURED_OUTPUT_TEXT = """
This run requires structured output. When complete, call `done(value)` with a
JSON-compatible Python value matching this JSON Schema exactly:

```json
{schema_hint}
```
"""


def tools_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None:
        return ""
    visible, _hidden = partition_repl_namespace(
        flow.tool_namespace_for_prompt(graph),
        hidden_names=PROMPT_DOCUMENTED_TOOL_NAMES,
    )
    lines = [
        line for name in sorted(visible) if (line := format_tool_line(visible[name]))
    ]
    clients = getattr(flow, "_llm_clients", {})
    if len(clients) > 1:
        if lines:
            lines.append("")
        lines.append("Available models:")
        lines += [f"- `{key}`" for key in sorted(clients)]
    if not lines:
        return ""
    return "\n".join(
        [
            "Tool functions are already in the REPL namespace; call them directly.",
            "",
            *lines,
        ]
    )


def structured_output_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None or graph.output_schema is None:
        return ""
    hint = flow.output_parser.system_prompt_hint(graph.output_schema)
    return STRUCTURED_OUTPUT_TEXT.replace("{schema_hint}", hint).strip()


def status_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None:
        return ""
    max_depth = getattr(flow, "max_depth", 0)
    if max_depth == 0:
        return "Baseline mode: no sub-agents available."
    note = f"You are at recursion depth **{graph.depth}** of max **{max_depth}**."
    if graph.depth >= max_depth:
        note += " You cannot spawn sub-agents."
    return note


DEFAULT_BUILDER = (
    PromptBuilder()
    .section("role", ROLE_TEXT)
    .section("strategy", STRATEGY_TEXT)
    .section("format", FORMAT_TEXT)
    .section("final", FINAL_TEXT)
    .section("structured-output", structured_output_section, title="Structured Output")
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)

SYSTEM_PROMPT = DEFAULT_BUILDER.build()


__all__ = [
    "DEFAULT_BUILDER",
    "PromptBuilder",
    "SYSTEM_PROMPT",
    "Section",
    "SectionBody",
]
