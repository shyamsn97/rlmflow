"""Minimal prompt builder and default prompt."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlmflow.prompts.messages import (
    PromptBuilder,
    RenderFn,
    build_inputs_manifest,
)
from rlmflow.structured import system_prompt_hint
from rlmflow.tools import format_tool_docs
from rlmflow.tools.builtins import BuiltIns
from rlmflow.tools.tools import PromptExample, get_toolset_metadata

SectionBody = str | Callable[[Any, Any], str]

#: Ceiling for the statically rendered ``SYSTEM_PROMPT`` (no flow/agent bound),
#: enforced by ``test_the_default_prompt_stays_under_its_size_ceiling``.
#:
#: This is an alarm on silent accretion, not a size target. Size is not what makes
#: a prompt good: the accreted 6,972-char version scored 0.708 on the ten-task
#: suite, cutting it to official-rlm's shape at 4,311 scored 0.468 (paired
#: p = 0.013), and official itself scores 0.613 on 2,094. What matters is which
#: paragraphs are load-bearing. The current static prompt is under 5K; this leaves
#: room for API guidance while still catching substantial accidental accretion.
MAX_STATIC_PROMPT_CHARS = 7_500


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
        agent: Any = None,
        body_override: str | None = None,
    ) -> str:
        body = body_override if body_override is not None else self.body
        text = body(flow, agent) if callable(body) else body
        text = textwrap.dedent(text).strip()
        if not text:
            return ""
        if self.title:
            return f"{'#' * max(self.level, 1)} {self.title}\n\n{text}"
        return text


class Sections(list):
    """An ordered, name-addressable list of ``Section``.

    Editing methods mutate in place and return ``self``, so a section set reads
    top-to-bottom without copy-on-write. This is the natural home for the
    add/update/drop helpers: you are editing the list you are about to render.
    """

    def _index(self, name: str) -> int | None:
        for index, section in enumerate(self):
            if section.name == name:
                return index
        return None

    def add(
        self,
        name: str,
        body: SectionBody = "",
        *,
        title: str | None = None,
        level: int = 2,
        before: str | None = None,
        after: str | None = None,
    ) -> Sections:
        """Add a section (or replace the same-named one), inserting ``before`` or
        ``after`` a named section when given, else appending."""
        new = Section(name, body, title=title, level=level)
        existing = self._index(name)
        if existing is not None:
            self[existing] = new
        else:
            before_index = self._index(before) if before is not None else None
            after_index = self._index(after) if after is not None else None
            if before_index is not None:
                self.insert(before_index, new)
            elif after_index is not None:
                self.insert(after_index + 1, new)
            else:
                self.append(new)
        return self

    def update(self, name: str, body: SectionBody) -> Sections:
        """Swap a section's body, keeping its position/title/level."""
        i = self._index(name)
        if i is None:
            raise KeyError(f"no section named {name!r}; use .add() to add it")
        old = self[i]
        self[i] = Section(name, body, title=old.title, level=old.level)
        return self

    def drop(self, name: str) -> Sections:
        """Remove a section by name (no error if absent)."""
        self[:] = [section for section in self if section.name != name]
        return self

    @property
    def names(self) -> list[str]:
        """Section names in render order (for inspection)."""
        return [section.name for section in self]


class SystemPromptBuilder(PromptBuilder):
    """The system-prompt side.

    As a ``PromptBuilder``, calling it with ``(flow, agent)`` returns the system
    message (``[{"role": "system", "content": …}]``). ``render(flow, agent)``
    gives the bare string when that's what's wanted (snapshots, the static
    ``SYSTEM_PROMPT``). ``self.sections`` is a mutable ``Sections`` seeded from
    ``default_sections()`` — tweak it in place for small changes, override
    ``default_sections`` to ship a different baseline, or override
    ``render``/``__call__`` for fully dynamic per-turn assembly.
    """

    def __init__(self) -> None:
        self.sections: Sections = self.default_sections()

    def default_sections(self) -> Sections:
        return (
            Sections()
            .add("role", role_section)
            .add("builtins", toolset_docs_section)
            .add("tools", tools_section, title="Tools")
            .add("models", models_section, title="Models")
            .add("repl-strategy", repl_strategy_section)
            .add("examples", examples_section, title="Examples")
            .add(
                "structured-output",
                structured_output_section,
                title="Structured Output",
            )
            .add(
                "structured-output-option",
                structured_output_option_section,
                title="Structured Subagent Output",
            )
            .add(
                "prompt-profiles",
                prompt_profiles_section,
                title="Prompt Profiles",
            )
            .add("inputs", inputs_section, title="Inputs")
            .add("agents", agents_section, title="Agents")
            .add("status", status_section, title="Status")
        )

    def render(self, flow: Any = None, agent: Any = None) -> str:
        """Render the sections into the bare system-prompt string."""
        return self.build(self.sections, flow, agent)

    def __call__(self, flow: Any = None, agent: Any = None) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.render(flow, agent)}]

    def build(
        self,
        sections: list[Section],
        flow: Any = None,
        agent: Any = None,
        **overrides: str,
    ) -> str:
        """Render a section list into the prompt string."""
        parts = []
        for section in sections:
            rendered = section.render(flow, agent, overrides.get(section.name))
            if rendered.strip():
                parts.append(rendered)
        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
        return text + "\n" if text else ""

    @property
    def names(self) -> list[str]:
        return self.sections.names


#: A system prompt is, at bottom, a function of the flow + agent. A plain string
#: is a constant one and a ``SystemPromptBuilder`` is a callable one, so all three
#: are accepted anywhere a system prompt source is expected.
SystemPromptFn = Callable[[Any, Any], str]
SystemPromptSource = str | SystemPromptFn | SystemPromptBuilder


def as_system_prompt_fn(source: Any) -> SystemPromptFn:
    """Normalize any system-prompt source to a ``(flow, agent) -> str`` function.

    A ``SystemPromptBuilder`` now returns a chat message from ``__call__``, so its
    string is produced via ``render`` here; a bare function is assumed to already
    return the string; a plain ``str`` is a constant.
    """
    if isinstance(source, str):
        return lambda _flow=None, _agent=None: source
    if isinstance(source, SystemPromptBuilder):
        return source.render
    if callable(source):  # a bare (flow, agent) -> str function
        return source
    raise TypeError(f"unsupported system prompt source: {type(source)!r}")


@dataclass
class PromptProfile:
    """A named system prompt and current-frontier renderer for a child agent.

    ``None`` on either side means *inherit the flow's default* for that side, so a
    profile can override just the system prompt. ``description`` is a short summary
    of when to use the profile; it is what gets advertised to the orchestrator so
    it can select one (see ``prompt_profiles_section``).
    """

    system: SystemPromptSource | None = None
    render_fn: RenderFn | None = None
    description: str = ""


ROLE_TEXT = """
You are a Recursive Language Model (RLM): a language model with a prompt, and very
important inputs stored in a Python REPL related to that prompt.
You can iteratively interact with the Python REPL, which has access to LLM calls and
recursive agents as functions. You will be queried turn-by-turn until you have an
answer to the query.
"""

CHILD_CONTEXT_TEXT = """
This task was assigned by a parent agent. Stay within the assigned scope and
return a result the parent can integrate.
"""

REPL_TEXT = """
To use the REPL, write code in ```repl``` blocks; the REPL persists across turns.
Available in the REPL:

- `INPUTS: dict[str, str]`: the important, potentially very long information
  related to the prompt.
- Top-level `await` is available; do not start another event loop.
"""

REPL_OUTPUT_WITH_QUERIES_TEXT = """
REPL outputs over ~20K characters are truncated, so for longer payloads slice
`INPUTS` and pass slices through `llm_query` rather than `print`-ing them whole.
"""

REPL_OUTPUT_LOCAL_TEXT = """
REPL outputs over ~20K characters are truncated, so inspect bounded slices of
`INPUTS` rather than `print`-ing it whole.
"""

REPL_STRATEGY_TEXT = """
The REPL is NOT a Jupyter cell — only `print(...)` output (stdout) is shown back
to you between turns; a bare expression on the last line is silently discarded.
Always wrap inspections in `print(...)`.

As a general strategy, start by probing `INPUTS` to understand it better (e.g.
print a few keys or lines, count records, etc.). Then use the REPL to build up an
answer to the query.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the
output, then continue on the next turn. Do not call `finish(...)` on turn 1 without
first inspecting `INPUTS` or calling a relevant task tool.

When the prompt depends on data exposed by a custom tool, call that tool for the
required data; never substitute remembered or outside values. Verification must
test the candidate against the original source, examples, or constraints rather
than merely rechecking assumptions produced by the same code.

To submit, first compute and bind the candidate without calling `finish(...)`.
Print the candidate and its checks, then end the block. After reading that output
on the next turn, send a new ```repl``` block whose only statement is
`finish(candidate)`. The run terminates immediately when `finish(...)` executes.
"""

LOCAL_EXAMPLE_TEXT = """
Probe the structure before computing anything:

```repl
import json

records = json.loads(INPUTS["task"])["records"]
print({
    "records": len(records),
    "fields": sorted(set().union(*(r.keys() for r in records))),
})
```

State persists, so the next turn computes over every record, checks the result,
and prints the candidate rather than submitting it unseen:

```repl
assert records and all("value" in record for record in records)
values = [float(record["value"]) for record in records]
assert len(values) == len(records)
mean = sum(values) / len(values)
print({"records": len(values), "mean": mean})
```

The mean is on screen now, so the next turn submits a value already read.
`finish(mean)` is the whole block: appending it to the block above would have
submitted a number never seen.

```repl
finish(mean)
```
"""

DELEGATION_EXAMPLE_TEXT = """
Launch children together, pass paths rather than contents, and keep working
while they run:

```repl
handles = [
    await launch_subagent(
        name="totals",
        model="{current_model}",
        goal="Read the CSV at INPUTS['path'] in full. Return one line per category as `category: total`.",
        inputs={"path": INPUTS["transactions_path"]},
    ),
]
print({"launched": [handle.name for handle in handles]})
```

Collect the reports when integration needs them, build the candidate, and print
it before submitting:

```repl
reports = [await handle.wait_for_result() for handle in handles]
assert reports and all(report.strip() for report in reports)
candidate = "\\n".join(reports)
print({"reports": len(reports), "candidate": candidate})
```

Read that output. If it is correct, submit on the next turn with `finish(...)`
alone:

```repl
finish(candidate)
```
"""

STRUCTURED_OUTPUT_TEXT = """
This run requires structured output. When complete, call `finish(value)` with a
JSON-compatible Python value matching this JSON Schema exactly:

```json
{schema_hint}
```
"""

STRUCTURED_OUTPUT_OPTION_TEXT = """
Child results are plain text by default. Pass `output_schema` only when a
validated typed result is needed; use explicit `properties`, `required`, and
`additionalProperties: false`.
"""

AGENTS_TEXT = """
`AGENTS` is a read-only snapshot of this run's recursive agent tree:

- `AGENTS.get()` returns you; `AGENTS.get(id_or_path_or_name)` finds another agent.
- `AGENTS.get_parent()`, `.get_siblings()`, and `.get_children()` return
  `AgentInfo` objects relative to you (or pass an agent selector).
- `AgentInfo.status` is `running`, `waiting`, `idle`, or `completed`.
  `.result()` returns an already-completed result and raises while one is still
  in flight; `await agent.wait_for_result()` waits automatically.
- `AGENTS.print_graph(show_results=True)` prints the whole tree with statuses and
  bounded result previews.

The snapshot is refreshed before each REPL action. It does not wait, message,
cancel, or steer agents, and repeated queries within one action are not fresher.
"""


def as_agent(node: Any) -> Any:
    """The ``AgentStart`` for ``node``. ``parent_agent`` is a stored field, O(1)."""
    if node is None:
        return None
    parent = getattr(node, "parent_agent", None)
    if parent is not None:
        return parent
    return node if getattr(node, "config", None) is not None else None


def bound_toolsets(flow: Any = None) -> list[Any]:
    if flow is not None:
        return list(getattr(flow, "toolsets", ()) or ())
    return [BuiltIns()]


def collect_examples(flow: Any = None, node: Any = None) -> list[PromptExample]:
    examples: list[PromptExample] = []
    for instance in bound_toolsets(flow):
        producer = getattr(instance, "example", None)
        if not callable(producer):
            continue
        produced = producer(flow, node)
        if produced is None:
            continue
        if isinstance(produced, PromptExample):
            examples.append(produced)
        else:
            examples.extend(produced)
    examples.sort(key=lambda item: item.priority)
    return examples


def can_spawn(flow: Any = None, agent: Any = None) -> bool:
    """Whether this prompt's agent can create another recursion level."""
    agent = as_agent(agent)
    if flow is None or agent is None:
        return True
    return agent.config.max_depth > 0 and agent.config.depth < agent.config.max_depth


def role_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    parts = [ROLE_TEXT.strip()]
    if agent is not None and agent.config.depth > 0:
        parts.append(CHILD_CONTEXT_TEXT.strip())
    return "\n\n".join(parts)


def repl_strategy_section(flow: Any = None, agent: Any = None) -> str:
    has_queries = flow is None or getattr(flow, "use_llm_query", False)
    output_text = REPL_OUTPUT_WITH_QUERIES_TEXT if has_queries else REPL_OUTPUT_LOCAL_TEXT
    return f"{output_text.strip()}\n\n{REPL_STRATEGY_TEXT.strip()}"


def toolset_docs_section(flow: Any = None, agent: Any = None) -> str:
    parts = []
    for instance in bound_toolsets(flow):
        meta = get_toolset_metadata(instance)
        if meta is None or meta.placement != "section":
            continue
        producer = getattr(instance, "description", None)
        text = producer(flow, agent).strip() if callable(producer) else ""
        if not text:
            continue
        parts.append(f"## {meta.title}\n\n{text}")
    return "\n\n".join(parts)


def builtin_api_section(flow: Any = None, agent: Any = None) -> str:
    # Kept as an alias: custom prompt builders may still call it by this name.
    return toolset_docs_section(flow, agent)


def examples_section(flow: Any = None, agent: Any = None) -> str:
    parts = []
    for item in collect_examples(flow, agent):
        body = item.body.strip()
        if item.title and item.title not in body:
            parts.append(f"**{item.title}**\n\n{body}")
        else:
            parts.append(body)
    return "\n\n".join(parts)


def inputs_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    if agent is None:
        return ""
    return build_inputs_manifest(dict(agent.config.inputs))


def agents_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    if flow is None or agent is None or not getattr(flow, "use_agent_tree", False):
        return ""
    return AGENTS_TEXT.strip()


def tools_section(flow: Any = None, agent: Any = None) -> str:
    node = agent
    agent = as_agent(node)
    if flow is None or agent is None:
        return ""
    docs = format_tool_docs(
        bound_toolsets(flow),
        flow.tools,
        flow=flow,
        node=node,
    )
    if not docs:
        return ""
    return (
        "Functions listed here are already available in the REPL. Use a tool "
        "when the task depends on data it provides.\n\n"
        f"{docs}"
    )


def models_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    if flow is None or agent is None or not can_spawn(flow, agent):
        return ""
    clients = getattr(flow, "_llm_clients", {})
    if not clients:
        return ""
    lines = ["Registered models (`model=` is required):"]
    for key in sorted(clients):
        current = " — current model" if key == agent.config.model else ""
        lines.append(f"- `{key}`{current}")
    return "\n".join(lines)


def structured_output_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    schema = agent.config.output_schema if agent is not None else None
    if flow is None or schema is None:
        return ""
    hint = system_prompt_hint(schema)
    return STRUCTURED_OUTPUT_TEXT.format(schema_hint=hint).strip()


def structured_output_option_section(flow: Any = None, agent: Any = None) -> str:
    # Independent of any required `output_schema`: this teaches the *capability*
    # of requesting structured output from subagents. Only useful when this agent
    # can actually spawn them.
    agent = as_agent(agent)
    if flow is None or agent is None:
        return ""
    if not getattr(flow, "enable_structured_output", False):
        return ""
    if not can_spawn(flow, agent):
        return ""
    return STRUCTURED_OUTPUT_OPTION_TEXT.strip()


def prompt_profiles_section(flow: Any = None, agent: Any = None) -> str:
    """Advertise selectable prompt profiles so the orchestrator can choose one
    per child (via ``launch_subagent(..., prompt_profile=...)``).

    Shown when the flow has a non-empty profile registry, the agent can still
    spawn children, and no custom ``prompt_router`` is installed. A callable
    router means the host chooses, so listing names would be noise.
    """
    if flow is None or agent is None:
        return ""
    agent = as_agent(agent)
    profiles = getattr(flow, "prompt_profiles", {})
    if not profiles or getattr(flow, "prompt_router", None) is not None:
        return ""
    if not can_spawn(flow, agent):
        return ""
    lines = ["Available prompt profiles (pass `prompt_profile` to `launch_subagent`):"]
    for name in sorted(profiles):
        desc = getattr(profiles[name], "description", "")
        lines.append(f"- `{name}`" + (f" — {desc}" if desc else ""))
    return "\n".join(lines)


def status_section(flow: Any = None, agent: Any = None) -> str:
    agent = as_agent(agent)
    if flow is None or agent is None:
        return ""
    max_depth = agent.config.max_depth
    if max_depth == 0:
        return "Baseline mode: no sub-agents available."
    note = (
        f"You are using model key **`{agent.config.model}`** at recursion depth "
        f"**{agent.config.depth}** of max **{max_depth}**."
    )
    if agent.config.depth >= max_depth:
        note += " You cannot spawn sub-agents."
    return note


#: The default system-prompt builder instance. Shared and mutable — to customize,
#: construct your own ``SystemPromptBuilder()`` and edit its ``.sections`` rather
#: than mutating this one.
DEFAULT_BUILDER = SystemPromptBuilder()

SYSTEM_PROMPT = DEFAULT_BUILDER.render()


__all__ = [
    "DEFAULT_BUILDER",
    "MAX_STATIC_PROMPT_CHARS",
    "SYSTEM_PROMPT",
    "PromptProfile",
    "Section",
    "SectionBody",
    "Sections",
    "SystemPromptBuilder",
    "SystemPromptFn",
    "SystemPromptSource",
    "as_system_prompt_fn",
]
