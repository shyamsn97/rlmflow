"""Minimal prompt builder and default prompt."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from typing import Any

from rflow.prompts.messages import build_inputs_manifest
from rflow.tools import format_tool_line, partition_repl_namespace

SectionBody = str | Callable[[Any, Any], str]
PROMPT_DOCUMENTED_TOOL_NAMES = frozenset({"done", "launch_subagents"})

#: Ceiling for the statically rendered ``SYSTEM_PROMPT`` (no flow/graph bound).
#: A regression guard so the default prompt stays lean; see the test suite.
MAX_STATIC_PROMPT_CHARS = 10_000


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

    def _copy(self) -> PromptBuilder:
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
    ) -> PromptBuilder:
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

    def update(self, name: str, body: SectionBody) -> PromptBuilder:
        out = self._copy()
        for index, section in enumerate(out._sections):
            if section.name == name:
                out._sections[index] = Section(
                    name, body, title=section.title, level=section.level
                )
                return out
        raise KeyError(f"no section named {name!r}; use .section() to add it")

    def remove(self, name: str) -> PromptBuilder:
        out = self._copy()
        out._sections = [section for section in out._sections if section.name != name]
        return out

    @property
    def names(self) -> list[str]:
        """Section names in render order (for inspection and custom builders)."""
        return [section.name for section in self._sections]

    def build(self, flow: Any = None, graph: Any = None, **overrides: str) -> str:
        parts = []
        for section in self._sections:
            rendered = section.render(flow, graph, overrides.get(section.name))
            if rendered.strip():
                parts.append(rendered)
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
        return text + "\n" if text else ""


ROLE_TEXT = """
You are a Recursive Coding Agent: a language model with a user query and important
inputs stored in a Python REPL. You are queried turn-by-turn until you have an
answer. To use the REPL, write code in ```repl``` blocks; it persists across turns.

Available in the REPL:

1. `INPUTS`: a dict of string inputs (may be empty). Your task arrives as the user
   message, not in `INPUTS`. Keys are caller-defined — inspect `list(INPUTS)`
   rather than assuming names; parse JSON with `json.loads(INPUTS["key"])`. Keys
   never shadow REPL variables or tools.
2. `await launch_subagents(specs) -> list`: recursive sub-agent calls. Each spec
   needs a short `query` and may set `inputs` (str -> str), `name`, `model`, and
   `output_schema`. Keep `query` a one/two-sentence instruction and put large
   payloads in `inputs`. Returns a list even for one child.
3. `print(...)`: only stdout is shown back between turns; a bare final expression
   is discarded. Never dump large `INPUTS` values — REPL output is truncated.
4. `done(answer)`: submit the final answer (a schema-matching value if this agent
   has an output schema, else a string). Only call it once verified.
"""

STRATEGY_TEXT = """
When `INPUTS` are present their keys and sizes are already listed for you, so read
what you need straight from the values (`INPUTS["key"]`). It is usually worth a
quick look before acting on large or unfamiliar inputs; inspect long inputs in
targeted windows rather than dumping them, since REPL output is truncated.

Act as an orchestrator, not a solver: delegate independent branches with
`await launch_subagents([...])`, and keep the root for preparing inputs,
integrating and verifying results, and the final `done(...)`. Your context window
is small — push heavy reading/summarizing/verifying into subcalls. Verify a
candidate answer before calling `done(...)`.
"""

FORMAT_TEXT = """
Execute Python in fenced `repl` blocks. Use exactly one block per assistant
message, with the opening and closing triple backticks.
"""

EXAMPLES_TEXT = """
**Observe inputs before acting** (first block when `INPUTS` is non-empty):

```repl
print("input keys:", list(INPUTS))
for key, value in INPUTS.items():
    print(key, "chars=", len(value), "lines=", len(value.splitlines()))
```

**Fan out slices after observation** — keep payloads in child `inputs`, not `query`:

```repl
lines = INPUTS["corpus"].splitlines()
batches = ["\\n".join(lines[i:i + 500]) for i in range(0, len(lines), 500)]
results = await launch_subagents([
    {"name": f"scan-{i}", "query": "Report findings in INPUTS['slice'] or NO_MATCH.", "inputs": {"slice": b}}
    for i, b in enumerate(batches)
])
hits = [r.strip() for r in results if r.strip() and r.strip() != "NO_MATCH"]
done("\\n".join(hits) if hits else "NO_MATCH")
```
"""

FINAL_TEXT = """
When the task is complete and verified, call `done(answer)` inside a ```repl```
block; `answer` must match the query's requested form and ends the run. A failed
check is not a final answer — repair it or delegate a repair, then re-verify.
"""

STRUCTURED_OUTPUT_TEXT = """
This run requires structured output. When complete, call `done(value)` with a
JSON-compatible Python value matching this JSON Schema exactly:

```json
{schema_hint}
```
"""

STRUCTURED_OUTPUT_OPTION_TEXT = """
You may require any subagent to return structured data instead of free text: add
an `output_schema` (a JSON Schema) to its spec. That subagent must then call
`done(value)` with a value matching the schema, and you get the parsed value back
(a `dict`/`list`) rather than a string. This can be very useful when dealing with outputs that require a specific structure.

```repl
results = await launch_subagents([
    {
        "name": "extract",
        "query": "Extract the fields described by the schema from INPUTS['doc'].",
        "inputs": {"doc": INPUTS["doc"]},
        "output_schema": {
            "type": "object",
            "properties": {
                "total": {"type": "number"},
                "currency": {"type": "string"},
            },
            "required": ["total", "currency"],
        },
    },
])
total = results[0]["total"]  # already parsed, not a string
```
"""


FIRST_TURN_TEXT_INPUTS = """
You have not run any code or seen your inputs yet. Make this an inspection turn:
`print(list(INPUTS))` with each value's size, read the windows you need, and wait
for the output before planning, delegating, or calling `done(...)`.
"""

FIRST_TURN_TEXT_BARE = """
You have not run any code yet. Explore and verify with the REPL before calling
`done(...)`; don't answer from assumption on the first turn.
"""


def examples_section(flow: Any = None, graph: Any = None) -> str:
    return EXAMPLES_TEXT.strip()


def first_turn_section(flow: Any = None, graph: Any = None) -> str:
    # Bootstrap-only safeguard: rendered while the agent has produced no output
    # yet, then it drops out once the trajectory has an ``llm_output`` node.
    if graph is None or any(node.type == "llm_output" for node in graph.nodes):
        return ""
    text = FIRST_TURN_TEXT_INPUTS if graph.inputs else FIRST_TURN_TEXT_BARE
    return text.strip()


def inputs_section(flow: Any = None, graph: Any = None) -> str:
    if graph is None:
        return ""
    return build_inputs_manifest(dict(graph.inputs))


def tools_section(flow: Any = None, graph: Any = None) -> str:
    if flow is None or graph is None:
        return ""
    visible, _hidden = partition_repl_namespace(
        flow.tool_namespace_for_prompt(graph),
        hidden_names=PROMPT_DOCUMENTED_TOOL_NAMES,
    )
    lines = []
    for name in sorted(visible):
        line = format_tool_line(visible[name])
        if line:
            lines.append(line)
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


def structured_output_option_section(flow: Any = None, graph: Any = None) -> str:
    # Independent of any required `output_schema`: this teaches the *capability*
    # of requesting structured output from subagents. Only useful when this agent
    # can actually spawn them.
    if flow is None or graph is None:
        return ""
    if not getattr(flow, "enable_structured_output", True):
        return ""
    max_depth = getattr(flow, "max_depth", 0)
    if max_depth == 0 or graph.depth >= max_depth:
        return ""
    return STRUCTURED_OUTPUT_OPTION_TEXT.strip()


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
    .section("examples", examples_section, title="Examples")
    .section("final", FINAL_TEXT)
    .section("structured-output", structured_output_section, title="Structured Output")
    .section(
        "structured-output-option",
        structured_output_option_section,
        title="Structured Subagent Output",
    )
    .section("tools", tools_section, title="Tools")
    .section("inputs", inputs_section, title="Inputs")
    .section("status", status_section, title="Status")
    .section("first-turn", first_turn_section, title="First Turn")
)

SYSTEM_PROMPT = DEFAULT_BUILDER.build()


__all__ = [
    "DEFAULT_BUILDER",
    "MAX_STATIC_PROMPT_CHARS",
    "PromptBuilder",
    "SYSTEM_PROMPT",
    "Section",
    "SectionBody",
]
