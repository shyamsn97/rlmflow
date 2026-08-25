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
from rlmflow.tools import format_tool_line, partition_repl_namespace

SectionBody = str | Callable[[Any, Any], str]
PROMPT_DOCUMENTED_TOOL_NAMES = frozenset({"finish", "done", "launch_subagent"})

#: Ceiling for the statically rendered ``SYSTEM_PROMPT`` (no flow/agent bound).
#: A regression guard so the default prompt stays lean, enforced by
#: ``test_the_default_prompt_stays_under_its_size_ceiling``. Raise it only
#: deliberately: sections accrete one paragraph at a time, and the last time this
#: went unenforced the prompt drifted 28% past it unnoticed.
MAX_STATIC_PROMPT_CHARS = 9_000


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
            .add("role", ROLE_TEXT)
            .add("builtins", builtin_api_section, title="Built-in REPL API")
            .add("context", CONTEXT_TEXT, title="Context and Working Memory")
            .add("format", FORMAT_TEXT)
            .add("examples", examples_section, title="Examples")
            .add("final", FINAL_TEXT)
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
            .add("tools", tools_section, title="Tools")
            .add(
                "prompt-profiles",
                prompt_profiles_section,
                title="Prompt Profiles",
            )
            .add("inputs", inputs_section, title="Inputs")
            .add("agents", agents_section, title="Agents")
            .add("status", status_section, title="Status")
            .add("strategy", strategy_section)
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
You are a Recursive Coding Agent: a language model with a user query and
supporting inputs stored in a Python REPL. You are queried turn-by-turn
until you have an answer. To use the REPL, write code in ```repl``` blocks; it
persists across turns.
"""

DELEGATION_API_TEXT = """
The following names are already bound in the REPL; call them directly without
imports.

### `await launch_subagent(...) -> AgentHandle`

```python
await launch_subagent(
    goal: str,
    model: str,
    name: str | None = None,
    inputs: dict[str, str] | None = None,
    output_schema: object | None = None,
    prompt_profile: str | None = None,
    reuse_repl: bool = False,
)
```

- `goal` (required): one or two sentences describing the child task. Put large
  data in `inputs`; an overlong goal is refused.
- `model` (required): registered model name chosen for this workstream. Select it
  explicitly from **Available models** below.
- `name`: stable child name containing only ASCII letters, digits, `_`, or `-`.
  Names must be unique among siblings. An omitted name is generated.
- `inputs`: copied into the child's `INPUTS`. Values must be strings.
- `output_schema`: JSON Schema (or another supported schema object). When set,
  the child must `finish(...)` with a matching value and `wait_for_result()`
  returns it parsed instead of as text.
- `prompt_profile`: registered child prompt profile. `None` inherits the current
  profile.
- `reuse_repl`: place the child in this agent's worker, sharing live Python
  objects, imports, globals, and worker failure while keeping separate
  transcripts, `INPUTS`, and `ENV`. This changes runtime placement only; the
  child still runs itself.

Subagents are full agents with their own REPL and can use the functions documented
under **Tools**. Every launch starts one in the background and returns an
`AgentHandle` with `id`, `name`, `path`, `status`, `result()`, and
`wait_for_result()`. Handles persist across REPL blocks, `status` and `result()`
read the current block's snapshot, and `await handle.wait_for_result()` waits only
while the child is still running, so awaiting handles one at a time does not
serialize them. A background child is not detached: the root may `finish(...)`
first, and the run keeps driving queued descendants into the same graph.
"""

CORE_API_TEXT = """
### `finish(answer: object) -> NoReturn`

Submits this agent's final answer and ends its run. Without an output schema,
`answer` becomes text; with one, pass a matching JSON-compatible value, and a
validation failure comes back as an error to repair. Call it once, after
verification.

### Namespace and observation

- `INPUTS: dict[str, str]` holds caller-provided payloads under caller-defined
  keys and may be empty. The task itself is the user message; any input may carry
  context, instructions, constraints, or data for it.
- `ENV: dict[str, object]` is persistent per-agent metadata/state; framework keys
  are prefixed `RLMFLOW_` (agent id, depth, max depth, parent id, root, replay).
- `AGENTS` appears only when agent-tree inspection is enabled, documented in the
  **Agents** section below.
- `asyncio` is preloaded; use `await asyncio.gather(...)` for independent async
  calls.
- `print(...)` is the observation channel: only stdout returns between turns, bare
  expressions are discarded, and long output is truncated. The REPL persists, so
  bind reusable results to names before printing an observation.
- Only the names documented here and under **Tools** are bound. Do not invent
  plausible-looking ones such as `run_subagent(...)`, `call_tool(...)`, or
  `get_result(...)`: if it is not documented, it does not exist.

Do not catch tool exceptions merely to convert failures into empty, missing, or
negative results. Catch one only when you can recover or report it; otherwise let
it surface so you can diagnose the real failure.
"""

CONTEXT_TEXT = """
The REPL persists across turns. Use it to inspect `INPUTS`, retain useful state in
variables, call tools, and coordinate subagents. Print bounded observations rather
than copying large inputs.
"""

INPUT_STRATEGY_TEXT = """
Before producing deliverables, inspect the complete relevant content of `INPUTS`
and preserve its explicit requirements and constraints rather than replacing them
with inferred defaults.
"""

STRATEGY_TEXT = """
Size up work. Keep small or tightly coupled work local. Several substantial
components with explicit interfaces are independent workstreams even within one
final product. Then act as an orchestrator: delegate coherent workstreams,
pass exact relevant context through `inputs`, launch independent children before
waiting, and choose each child's registered model explicitly.

Keep synthesis with the parent: combine useful results, verify, and finish without
redundant delegation or review loops.
"""

LEAF_STRATEGY_TEXT = """
Stay centered on the assigned goal and return material the parent can directly
use. Do not expand into the entire parent task unless that is necessary to
complete your scope. Verify the result before finishing.
"""

FORMAT_TEXT = """
Every reply must contain exactly one fenced ```repl``` block, including the one
that calls `finish(...)`. The block is the only thing that runs: prose outside it
wastes the turn, so carry out the next step in code instead of announcing it.

Write both the opening and closing triple backticks. Top-level `await` is
supported for async tools; never call `asyncio.run`, `run_until_complete`, or
`get_event_loop()`, since the REPL already runs inside an event loop and nesting
one raises RuntimeError.
"""

INSPECTION_EXAMPLES_TEXT = """
**Inspection example — task text.** This value happens to be a nested requirement
list, so parse every complete unit and its hierarchy, then inventory its binding
names and prohibitions. A preview such as `text[:100]`, keyword hits, or counts
alone would lose the task:

```repl
import re

requirements = []
for raw in INPUTS["context"].splitlines():
    match = re.match(r"^(\\s*)[-*]\\s+(.*)", raw)
    if match:
        requirements.append(
            {"depth": len(match.group(1)), "text": match.group(2).strip()}
        )
    elif raw.strip() and requirements:
        requirements[-1]["text"] += " " + raw.strip()
identifiers = sorted(set(re.findall(r"`([A-Za-z_]\\w*)`", INPUTS["context"])))
prohibitions = [
    item["text"]
    for item in requirements
    if re.search(r"\\b(?:do not|must not|never|no)\\b", item["text"], re.IGNORECASE)
]
observed = {
    "requirements": requirements,
    "identifiers": identifiers,
    "prohibitions": prohibitions,
}
print(observed)
```

**Inspection example — query-directed data.** The query asks which services
produced at least five errors, so compute that result rather than printing the
records:

```repl
import json
from collections import Counter

records = [json.loads(line) for line in INPUTS["events"].splitlines() if line.strip()]
errors = [record for record in records if record.get("status") == "error"]
errors_by_service = Counter(record["service"] for record in errors)
observed = {
    "records": len(records),
    "fields": sorted({key for record in records for key in record}),
    "services_with_5_errors": {
        service: count
        for service, count in errors_by_service.items()
        if count >= 5
    },
}
print(observed)
```
"""

DELEGATION_EXAMPLES_TEXT = """
**Local trajectory — one deterministic result.**

```repl
import re
section = re.search(
    r"(?ms)^\\[gateway\\]\\s*(.*?)(?=^\\[|\\Z)", INPUTS["config"]
).group(1)
answer = int(re.search(r"(?m)^bind_port\\s*=\\s*(\\d+)", section).group(1))
print({"gateway_port": answer})
```

The inspected value yields one bounded known lookup, so finish locally:

```repl
finish(answer)
```

**Fan-out trajectory — several independent substantial outputs.**

```repl
import json
brief = json.loads(INPUTS["brief"])
chapter_groups = {}
for chapter in brief["chapters"]:
    chapter_groups.setdefault(chapter["area"], []).append(chapter)
print({
    "chapters": len(brief["chapters"]),
    "substantial_scopes": sorted(chapter_groups),
    "parent_work": ["title", "table of contents", "assembly"],
})
```

Several requested chapters share one work area, so delegate one coherent scope
per area rather than one child per chapter. Keep the short report glue local:

```repl
planned_children = [
    {
        "name": f"area-{index}",
        "goal": "Write and verify the chapters described in INPUTS['chapters'].",
        "inputs": {"chapters": json.dumps(chapters)},
    }
    for index, chapters in enumerate(chapter_groups.values(), 1)
]
handles = await asyncio.gather(
    *(
        launch_subagent(
            name=child["name"],
            goal=child["goal"],
            model="{current_model}",
            inputs=child["inputs"],
        )
        for child in planned_children
    )
)
results = await asyncio.gather(*(handle.wait_for_result() for handle in handles))
print({"result_chars": [len(str(result)) for result in results]})
```

```repl
toc = "\\n".join(f"- {chapter['title']}" for chapter in brief["chapters"])
finish(f"# {brief['title']}\\n\\n{toc}\\n\\n" + "\\n\\n".join(map(str, results)))
```
"""

FINAL_TEXT = """
When the task is complete and verified, call `finish(answer)` inside a ```repl```
block; `answer` must match the query's requested form and ends the run. A failed
check is not a final answer — repair it or delegate a repair, then re-verify.
"""

STRUCTURED_OUTPUT_TEXT = """
This run requires structured output. When complete, call `finish(value)` with a
JSON-compatible Python value matching this JSON Schema exactly:

```json
{schema_hint}
```
"""

STRUCTURED_OUTPUT_OPTION_TEXT = """
You may require a subagent to return structured data instead of free text: pass
an `output_schema` (a JSON Schema). That subagent must then call
`finish(value)` with a value matching the schema, and waiting on its handle gives
you the parsed value (a `dict`/`list`) rather than a string. Make the schema
specific enough to reject missing, extra, placeholder, or wrong-kind results:
prefer explicit `properties`, `required`, and `additionalProperties: false`.
Schema validity does not change semantics; metadata is not the requested result.
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


def can_spawn(flow: Any = None, agent: Any = None) -> bool:
    """Whether this prompt's agent can create another recursion level."""
    if flow is None or agent is None:
        return True
    return agent.config.max_depth > 0 and agent.config.depth < agent.config.max_depth


def builtin_api_section(flow: Any = None, agent: Any = None) -> str:
    parts = [CORE_API_TEXT.strip()]
    if can_spawn(flow, agent):
        parts.insert(0, DELEGATION_API_TEXT.strip())
    return "\n\n".join(parts)


def strategy_section(flow: Any = None, agent: Any = None) -> str:
    parts = []
    if agent is None or agent.config.inputs:
        parts.append(INPUT_STRATEGY_TEXT.strip())
    parts.append((STRATEGY_TEXT if can_spawn(flow, agent) else LEAF_STRATEGY_TEXT).strip())
    return "\n\n".join(parts)


def examples_section(flow: Any = None, agent: Any = None) -> str:
    """Render only examples whose capabilities are available to this agent."""
    parts = []
    if agent is None or agent.config.inputs:
        parts.append(INSPECTION_EXAMPLES_TEXT.strip())
    if can_spawn(flow, agent):
        current_model = agent.config.model if agent is not None else "default"
        parts.append(DELEGATION_EXAMPLES_TEXT.replace("{current_model}", current_model).strip())
    return "\n\n".join(parts)


def inputs_section(flow: Any = None, agent: Any = None) -> str:
    if agent is None:
        return ""
    return build_inputs_manifest(dict(agent.config.inputs))


def agents_section(flow: Any = None, agent: Any = None) -> str:
    if flow is None or agent is None or not getattr(flow, "use_agent_tree", False):
        return ""
    return AGENTS_TEXT.strip()


def tools_section(flow: Any = None, agent: Any = None) -> str:
    if flow is None or agent is None:
        return ""
    visible, _hidden = partition_repl_namespace(
        flow.tool_namespace_for_prompt(agent),
        hidden_names=PROMPT_DOCUMENTED_TOOL_NAMES,
    )
    lines = []
    for name in sorted(visible):
        line = format_tool_line(visible[name])
        if line:
            lines.append(line)
    clients = getattr(flow, "_llm_clients", {})
    if clients and can_spawn(flow, agent):
        if lines:
            lines.append("")
        lines.append("Available models (`model=` is required for every `launch_subagent` call):")
        for key in sorted(clients):
            current = " — current model" if key == agent.config.model else ""
            lines.append(f"- `{key}`{current}")
    if not lines:
        return ""
    return "\n".join(
        [
            "Tool functions are already in the REPL namespace; call them directly.",
            "",
            *lines,
        ]
    )


def structured_output_section(flow: Any = None, agent: Any = None) -> str:
    schema = agent.config.output_schema if agent is not None else None
    if flow is None or schema is None:
        return ""
    hint = system_prompt_hint(schema)
    return STRUCTURED_OUTPUT_TEXT.replace("{schema_hint}", hint).strip()


def structured_output_option_section(flow: Any = None, agent: Any = None) -> str:
    # Independent of any required `output_schema`: this teaches the *capability*
    # of requesting structured output from subagents. Only useful when this agent
    # can actually spawn them.
    if flow is None or agent is None:
        return ""
    if not getattr(flow, "enable_structured_output", True):
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
