"""Minimal prompt builder and default prompt."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlmflow.prompts.messages import PromptBuilder, UserPromptSource, build_inputs_manifest
from rlmflow.structured import system_prompt_hint
from rlmflow.tools import format_tool_line, partition_repl_namespace

SectionBody = str | Callable[[Any, Any], str]
PROMPT_DOCUMENTED_TOOL_NAMES = frozenset({"finish", "done", "launch_subagent"})

#: Ceiling for the statically rendered ``SYSTEM_PROMPT`` (no flow/agent bound).
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
    """The system-prompt side, symmetric to ``UserPromptBuilder``.

    As a ``PromptBuilder``, calling it with ``(flow, agent)`` returns the system
    message (``[{"role": "system", "content": …}]``) so ``Flow.messages`` can
    concatenate it with the user turns uniformly. ``render(flow, agent)`` gives
    the bare string when that's what's wanted (``build_system_prompt``, snapshots,
    the static ``SYSTEM_PROMPT``). ``self.sections`` is a mutable ``Sections``
    seeded from ``default_sections()`` — tweak it in place for small changes,
    override ``default_sections`` to ship a different baseline, or override
    ``render``/``__call__`` for fully dynamic per-turn assembly.
    """

    def __init__(self) -> None:
        self.sections: Sections = self.default_sections()

    def default_sections(self) -> Sections:
        return (
            Sections()
            .add("role", ROLE_TEXT)
            .add("builtins", builtin_api_section, title="Built-in REPL API")
            .add("strategy", strategy_section)
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
            .add("first-turn", first_turn_section, title="First Turn")
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
    """A named ``(system, user)`` prompt pair for a child agent.

    ``None`` on either side means *inherit the flow's default* for that side, so a
    profile can override just the system prompt. ``description`` is a short summary
    of when to use the profile; it is what gets advertised to the orchestrator so
    it can select one (see ``prompt_profiles_section``).
    """

    system: SystemPromptSource | None = None
    user: UserPromptSource | None = None
    description: str = ""


ROLE_TEXT = """
You are a Recursive Coding Agent: a language model with a user query and important
inputs stored in a Python REPL. You are queried turn-by-turn until you have an
answer. To use the REPL, write code in ```repl``` blocks; it persists across turns.
"""

DELEGATION_API_TEXT = """
The following names are already bound in the REPL; call them directly without
imports.

### `await launch_subagent(...) -> AgentHandle`

```python
await launch_subagent(
    goal: str,
    name: str | None = None,
    inputs: dict[str, str] | None = None,
    model: str | None = None,
    output_schema: object | None = None,
    prompt_profile: str | None = None,
    reuse_repl: bool = False,
)
```

- `goal` (required): one or two sentences describing the child task. Put large
  data in `inputs`; an overlong goal is refused.
- `name`: stable child name containing only ASCII letters, digits, `_`, or `-`.
  Names must be unique among siblings. An omitted name is generated.
- `inputs`: copied into the child's `INPUTS`. Values must be strings.
- `model`: registered model name. `None` inherits the current agent's model.
- `output_schema`: JSON Schema (or another supported schema object). When set,
  the child must call `finish(value)` with a matching value, and
  `handle.wait_for_result()` returns the parsed Python value instead of text.
- `prompt_profile`: registered child prompt profile. `None` inherits the current
  profile.
- `reuse_repl`: when true, place the child in this agent's worker. The agents
  keep separate transcripts, `INPUTS`, and `ENV`, but share live Python objects,
  imports, globals, and worker failure.

Every launch starts the child in the background and immediately returns an
`AgentHandle` with `id`, `name`, `path`, `status`, `result()`, and
`wait_for_result()`. Launch all independent children first, then continue useful
parent work. Call `await handle.wait_for_result()` only when the result is needed;
it returns immediately if the child has already completed. Once handles have
been launched, awaiting them one-by-one does not serialize their execution.

A background child is not detached from the run. The root may call
`finish(...)` first, but the stream continues driving queued descendants and
recording their work in the same graph. Handles persist across REPL blocks;
`handle.status` and `handle.result()` read the fresh snapshot for the current
block. `await handle.wait_for_result()` waits automatically when needed.
"""

CORE_API_TEXT = """
### `finish(answer: object) -> NoReturn`

Submits this agent's final answer and immediately ends its run. Without an output
schema, `answer` is converted to text. With an output schema, pass a
JSON-compatible value matching it; validation failure is shown as an error to
repair. Call `finish` only once, after verification.

### Namespace and observation

- `INPUTS: dict[str, str]` contains caller-provided payloads and may be empty.
  The task itself is the user message, not an `INPUTS` entry. Inspect keys before
  assuming them and parse encoded JSON with `json.loads`.
- `ENV: dict[str, object]` is persistent per-agent metadata/state. Framework keys
  include `RLMFLOW_AGENT_ID`, `RLMFLOW_DEPTH`, `RLMFLOW_PARENT_AGENT_ID`,
  `RLMFLOW_MAX_DEPTH`, `RLMFLOW_IS_ROOT`, and `RLMFLOW_REPLAY`.
- `AGENTS` is present only when agent-tree inspection is enabled; its API is
  documented in the conditional **Agents** section below.
- `asyncio` is preloaded. Use `await asyncio.gather(...)` for independent async
  calls; do not call `asyncio.run(...)`.
- `print(...)` is the observation channel. Only stdout is returned between turns;
  bare expressions are discarded and long output is truncated.

Do not catch tool exceptions merely to convert failures into empty, missing, or
negative results. Catch an exception only when you can recover or report it;
otherwise let it surface so you can diagnose the real failure.
"""

STRATEGY_TEXT = """
When `INPUTS` are present their keys and sizes are already listed for you, so read
what you need straight from the values (`INPUTS["key"]`). It is usually worth a
quick look before acting on large or unfamiliar inputs; inspect long inputs in
targeted windows rather than dumping them, since REPL output is truncated.

Act as an orchestrator, not a solver: launch handles for independent branches
before collecting any result, and keep the root for preparing inputs, integrating
and verifying results, and the final `finish(...)`. Your context window is small
— push heavy reading/summarizing/verifying into subcalls. Verify a candidate
answer before calling `finish(...)`.

Lean toward decomposition whenever the task splits into separable parts — several
files or modules, independent sub-questions, or per-chunk scans. Give each part
its own subagent instead of solving them one-by-one in the root: independent
branches run concurrently and each child gets a fresh context budget, so the work
is usually both faster and higher quality. Keep steps that depend on each other in
the root (or chain them), and delegate one focused child per independent unit.
Background children overlap with later parent turns; call
`await handle.wait_for_result()` when a result is needed.
"""

LEAF_STRATEGY_TEXT = """
Work directly on the assigned task. Read the `INPUTS` you need, use the available
tools, and verify the result before calling `finish(...)`. Inspect long inputs in
targeted windows rather than dumping them, since REPL output is truncated.
"""

FORMAT_TEXT = """
Every reply you send must contain exactly one fenced ```repl``` block, including
the reply that calls `finish(...)`. The block is the only thing that runs: prose
outside it executes nothing and wastes the turn, so carry out the next step in
code rather than announcing what you are about to do.

Write the block with the opening and closing triple backticks. Top-level `await`
is supported for async tools. Never call `asyncio.run`, `run_until_complete`, or
`get_event_loop()`: the REPL already runs inside an event loop, and nesting one
raises RuntimeError.
"""

EXAMPLES_TEXT = """
**Observe inputs before acting** (first block when `INPUTS` is non-empty):

```repl
print("input keys:", list(INPUTS))
for key, value in INPUTS.items():
    print(key, "chars=", len(value), "lines=", len(value.splitlines()))
```

**Fan out slices after observation** — keep payloads in child `inputs`, not `goal`:

```repl
lines = INPUTS["corpus"].splitlines()
batches = ["\\n".join(lines[i:i + 500]) for i in range(0, len(lines), 500)]
handles = [
    await launch_subagent(
        goal="Report findings in INPUTS['slice'] or NO_MATCH.",
        name=f"scan-{i}",
        inputs={"slice": b},
    )
    for i, b in enumerate(batches)
]
results = [await handle.wait_for_result() for handle in handles]
hits = [r.strip() for r in results if r.strip() and r.strip() != "NO_MATCH"]
finish("\\n".join(hits) if hits else "NO_MATCH")
```

**Start work now and collect it in a later block**:

```repl
audit = await launch_subagent("Run the slow audit.", name="audit")
diagnostics = await launch_subagent(
    "Generate optional diagnostics.", name="diagnostics"
)
print("queued:", audit.name, diagnostics.name)
```

On a later turn, wait for those same handles without relaunching anything:

```repl
audit_result = await audit.wait_for_result()
diagnostics_result = await diagnostics.wait_for_result()
finish({"audit": audit_result, "diagnostics": diagnostics_result})
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
you the parsed value (a `dict`/`list`) rather than a string. This can be very
useful when dealing with outputs that require a specific structure.

```repl
extract = await launch_subagent(
    name="extract",
    goal="Extract the fields described by the schema from INPUTS['doc'].",
    inputs={"doc": INPUTS["doc"]},
    output_schema={
        "type": "object",
        "properties": {
            "total": {"type": "number"},
            "currency": {"type": "string"},
        },
        "required": ["total", "currency"],
    },
)
result = await extract.wait_for_result()
total = result["total"]  # already parsed, not a string
```
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


FIRST_TURN_TEXT_INPUTS = """
You have not run any code or seen your inputs yet. Start with an inspection turn
in this reply's block: `print(list(INPUTS))` with each value's size, read the
windows you need, and wait for the output before planning, delegating, or
answering. The manifest above names and sizes the inputs but does not read them,
so it is not the inspection turn. Call `finish(...)` only once you have the real,
verified final answer — never to end an exploration turn, and never with a
placeholder or status value.
"""

FIRST_TURN_TEXT_BARE = """
You have not run any code yet. If reaching a good answer needs exploration,
computation, or verification, do that first in a ```repl``` block and read the
output rather than answering from assumption; if it is pure reasoning, call
`finish(...)` in your first block. Either way the answer travels through
`finish(...)`, and only once it is real and verified — never to end an
exploration turn, and never with a placeholder or status value.
"""


def can_spawn(flow: Any = None, agent: Any = None) -> bool:
    """Whether this prompt's agent can create another recursion level."""
    if flow is None or agent is None:
        return True
    return flow.max_depth > 0 and agent.config.depth < flow.max_depth


def builtin_api_section(flow: Any = None, agent: Any = None) -> str:
    parts = [CORE_API_TEXT.strip()]
    if can_spawn(flow, agent):
        parts.insert(0, DELEGATION_API_TEXT.strip())
    return "\n\n".join(parts)


def strategy_section(flow: Any = None, agent: Any = None) -> str:
    return (STRATEGY_TEXT if can_spawn(flow, agent) else LEAF_STRATEGY_TEXT).strip()


def examples_section(flow: Any = None, agent: Any = None) -> str:
    return EXAMPLES_TEXT.strip() if can_spawn(flow, agent) else ""


def first_turn_section(flow: Any = None, agent: Any = None) -> str:
    # Bootstrap-only safeguard: rendered while the agent has produced no output
    # yet, then it drops out once the trajectory has an ``llm_output`` node.
    if agent is None or agent.llm_turns():
        return ""
    text = FIRST_TURN_TEXT_INPUTS if agent.config.inputs else FIRST_TURN_TEXT_BARE
    return text.strip()


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
    max_depth = getattr(flow, "max_depth", 0)
    if max_depth == 0:
        return "Baseline mode: no sub-agents available."
    note = f"You are at recursion depth **{agent.config.depth}** of max **{max_depth}**."
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
