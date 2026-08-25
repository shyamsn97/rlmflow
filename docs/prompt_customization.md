# Prompt Customization

`Flow` builds a system prompt from named sections. Most customization should
derive from the default builder instead of replacing the whole prompt, because
the default sections carry the REPL protocol, delegation API, and runtime
manifests while leaving planning free-form.

Use full replacement only when you want to own that entire protocol yourself.

## Inspect The Prompt

Before changing the prompt, render the one your agent already sees. Nothing has
to run first — a root from `flow.start(...)` is enough:

```python
import rlmflow

flow = rlmflow.Flow(llm)
root = flow.start("Summarize this document.", inputs={"document": document})
print(flow.system_prompt.render(flow, root))
```

`render(flow, agent)` renders against that agent's config — its query, inputs,
model, and output schema — and the flow's current prompt and tool configuration.
To see the whole conversation instead, `flow.build_messages(root.frontier)` returns the
message list the model receives, whose first entry is this same text.

## Default Builder Shape

The default builder has these sections, in order:

| Section | Purpose |
| --- | --- |
| `role` | Opening recursive-agent contract. |
| `builtins` | Runtime-aware `finish(...)` and delegation API. |
| `context` | Persistent REPL working memory and bounded observations. |
| `format` | One `repl` block per turn; use `print(...)` for short observations only. |
| `examples` | Input inspection, bounded local work, and coherent fan-out trajectories, rendered only when the agent can use them. |
| `final` | `finish(...)` contract and repair discipline. |
| `structured-output` | Per-agent `finish(value)` schema when the agent has an `output_schema`. |
| `structured-output-option` | How to request structured output from subagents (only when `enable_structured_output=True` and the agent can spawn children). |
| `tools` | Runtime-generated tool list (custom tools registered with the runtime, plus extra model aliases). |
| `inputs` | Runtime-generated metadata-only manifest of the agent's `INPUTS` (keys and sizes, never values). |
| `status` | Runtime-generated agent depth / spawn-budget status. |
| `strategy` | Final stable system instruction: inspect inputs, keep small work local, orchestrate substantial independent work, and retain parent synthesis. |

The static text sections render back-to-back so the prompt reads as one
continuous narrative; the split exists so each piece is independently swappable
via `prompt.sections.update(name, ...)`. `builtins`, `strategy`, `examples`,
`tools`, `inputs`, `status`, and `structured-output*` are callable sections
filled from the current Flow and Node at build time.

## Recommended: Edit `SystemPromptBuilder().sections`

The system prompt is a `SystemPromptBuilder`. Its `.sections` is a mutable,
name-addressable list (`Sections`); edit it in place and hand the builder to the
flow. Construct a fresh `SystemPromptBuilder()` rather than mutating the shared
`DEFAULT_BUILDER`.

The `.sections` editing methods (`add`, `update`, `drop`) mutate in place and
return the list, so edits read top-to-bottom.

### Add Project Rules

`add` inserts a section `before`/`after` a named one (or appends):

```python
import rlmflow
from rlmflow import SystemPromptBuilder

project_rules = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
- When changing public behavior, update docs in the same pass.
"""

prompt = SystemPromptBuilder()
prompt.sections.add("project_rules", project_rules, title="Project Rules", after="final")

flow = rlmflow.Flow(llm, system_prompt=prompt)
# or set it after construction:
flow.system_prompt = prompt
```

### Swap A Single Section

`update` swaps a section's body, keeping its position:

```python
domain_strategy = """
**When to delegate:** choose coherent work packages large enough to repay a full
agent's coordination cost. Group related small outputs and keep cross-package
integration in the root. Verify children mechanically before `finish()`.
"""

prompt = SystemPromptBuilder()
prompt.sections.update("strategy", domain_strategy)
```

### Prepend A Persona

Slip a small role section before `role` rather than overwriting the protocol:

```python
prompt = SystemPromptBuilder()
prompt.sections.add(
    "persona",
    "You are a recursive security auditor. Reproduce concrete risks and "
    "propose minimal fixes.",
    title="Persona",
    before="role",
)
```

### Remove A Section

`drop` removes a section by name — but removing core sections like `role`,
`strategy`, `format`, or `final` strips the delegation/REPL protocol. Prefer
dropping only sections you added.

```python
prompt = SystemPromptBuilder()
prompt.sections.drop("project_rules")
```

### Reusable Customization: Subclass

For a customization you want every time, subclass and override
`default_sections` (the static baseline).

```python
from rlmflow import SystemPromptBuilder

class AuditPrompt(SystemPromptBuilder):
    def default_sections(self):
        return super().default_sections().add(
            "rules", RULES, title="Rules", after="final"
        )

flow = rlmflow.Flow(llm, system_prompt=AuditPrompt())
```

### Build A Prompt From Scratch

Override `default_sections` to return your own `Sections`. Include the built-in
callable sections if you want the standard runtime-generated tools/status blocks.

```python
from rlmflow import SystemPromptBuilder
from rlmflow.prompts import Sections, status_section, tools_section

class MinimalPrompt(SystemPromptBuilder):
    def default_sections(self):
        return (
            Sections()
            .add("role", "You are a minimal REPL agent.", title="Role")
            .add(
                "protocol",
                """
- Use exactly one ```repl``` block per assistant message.
- Call `finish(answer)` exactly once when finished.
- Use tools to inspect or modify files.
""",
                title="Protocol",
            )
            .add("tools", tools_section, title="Tools")
            .add("status", status_section, title="Status")
        )

flow = rlmflow.Flow(llm, system_prompt=MinimalPrompt())
```

## The `system_prompt` Source

`system_prompt` (constructor arg or settable attribute) accepts any of three
things — a `SystemPromptBuilder`, a plain string, or a `(flow, agent) -> str`
function — and is resolved fresh on every turn:

- **`SystemPromptBuilder`** (the default, `DEFAULT_BUILDER`) — the section
  machinery described above.
- **string** — a constant prompt, bypassing the builder entirely.
- **function** — a dynamic prompt without subclassing anything.

```python
import rlmflow

# constant string (most fragile — you own the whole protocol)
flow = Flow(
    llm,
    system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `finish(answer)` exactly once when finished.
""",
)

# or a function of (flow, agent)
def prompt_for(flow, agent):
    depth = agent.config.depth
    tail = "Return an executive summary." if depth == 0 else "Return findings only."
    return f"You are an auditor. {tail}"

flow = rlmflow.Flow(llm, system_prompt=prompt_for)
```

A string that omits `launch_subagent`, `INPUTS`, `HISTORY`, or the `finish(...)`
rule means the model will not reliably use those features — prefer a
`SystemPromptBuilder` unless you intend to own the entire protocol.

## Dynamic Prompts

When the prompt should depend on the current agent, depth, query, available
tools, or project state, override `render` on a `SystemPromptBuilder` subclass.
It receives `(flow, agent)` — the `AgentStart` whose prompt is being built —
assembles `Sections`, and renders them with `self.build(...)`. The agent's query
is `agent.content`, and its model, inputs, depth, and output schema are on
`agent.config`.

```python
import rlmflow
from rlmflow import SystemPromptBuilder


class AuditPrompt(SystemPromptBuilder):
    def render(self, flow=None, agent=None) -> str:
        sections = self.default_sections()
        extra = (
            "At root depth, produce an executive summary after verification."
            if agent is None or agent.config.depth == 0
            else "As a child call, return only structured findings."
        )
        sections.add("audit_depth_rules", extra, title="Depth Rules", after="strategy")
        return self.build(sections, flow, agent)


flow = rlmflow.Flow(llm, system_prompt=AuditPrompt())
```

You can also replace narrower callable sections directly:

```python
from rlmflow import SystemPromptBuilder
from rlmflow.prompts import tools_section


def careful_tools(flow, agent):
    return tools_section(flow, agent) + "\n- Prefer read-only tools before write tools."


prompt = SystemPromptBuilder()
prompt.sections.update("tools", careful_tools)
```

(Assigning `flow.system_prompt` — a builder, or any `(flow, agent) -> str`
callable — still works if you'd rather own the whole prompt at the flow level,
and `prompt_profiles` with a `prompt_router` picks one per agent.)

## Callable Sections

The dynamic prompt hook above works, but it is heavier than it needs to be for
small additions like project rules or runtime notes. A prompt section can be
either static text or a function:

```python
def section(flow: rlmflow.Flow, agent: rlmflow.AgentStart) -> str:
    ...
```

The signature is intentionally just `flow, agent`. There is no context dict
and no separate prompt context object. If a section needs runtime tools, model
registrations, config, or the current agent id, those are already reachable
from `flow` and `agent`.

## Child-Specific Prompts

The easiest way to steer a child is the goal you pass to `launch_subagent`.
Use the global prompt for stable behavior and child goals for local contracts.

```python
api = await launch_subagent(
        "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        model="default",
        name="api",
        inputs={"spec": api_spec},
)
tests = await launch_subagent(
        "Implement tests for src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        model="default",
        name="tests",
        inputs={"spec": test_spec},
)
results = [await api.wait_for_result(), await tests.wait_for_result()]
```

## Per-child prompts

Sometimes a child agent should run under a *different* prompt than its parent — an
orchestrator RLM spawning coding agents, say. Register named **prompt profiles**
on the flow, parallel to `llm_clients`/`model`:

```python
import rlmflow
from rlmflow import PromptProfile

flow = rlmflow.Flow(
    llm,
    prompt_profiles={
        "coder": PromptProfile(
            system="You are a coding agent. ...",
            description="implement code changes",   # shown to the orchestrator
        ),
        "reviewer": PromptProfile(system="You are a terse reviewer."),
    },
)
```

A `PromptProfile` bundles a `system` source and a current-node `render_fn`;
`None` on either side
inherits the flow default. When omitted from `prompt_profiles`, `"default"`
means the flow's own `system_prompt`/`render_fn`; callers may also define it
explicitly.

By default, Flow reads the profile name from the agent's config. With a
non-empty registry, profile names and descriptions are
advertised in the orchestrator's system prompt, so it can name one per child:

```python
impl = await launch_subagent(
    "...", model="default", name="impl", prompt_profile="coder"
)
review = await launch_subagent(
    "...", model="default", name="review", prompt_profile="reviewer"
)
```

Pass a callable `prompt_router` only when host policy should choose the profile
dynamically. A custom router also suppresses profile advertising:

```python
flow = rlmflow.Flow(
    llm,
    prompt_profiles={"coder": CODER},
    prompt_router=lambda flow, agent: "coder" if agent.config.depth > 0 else "default",
)
```

The launch call's `prompt_profile` is stored on `UserQuery.prompt_profile` and
serialized. When omitted, a cold child inherits its immediate parent agent's
profile; a prepared branch keeps its own profile. Without a router, that stamp
is authoritative. With a router, the callable's result is authoritative.
Unknown names raise `ValueError`.

## Customizing The User Turns

Everything above shapes the *system* message. The rest of the conversation — the
user query, the assistant's replies, and the REPL/observation turns fed back in
— has two explicit layers:

- **canonical history** — each `Node.render()` returns a list of messages, and
  `node.project()` flattens those lists while walking history;
- **current frontier** — `Flow(render_fn=...)` or
  `PromptProfile(render_fn=...)` may render the node currently being sent
  differently without rewriting historical projection.

The default current renderer calls `node.render()` and adds live
background-agent status. Inspect, plan, final, continue, and truncation
instructions are typed nodes, not injected strings.

`Flow.build_messages` reserves the current renderer's messages inside
`keep_n_messages`, projects the remaining capacity from `node.prev`, prepends
the system message, and preserves every rendered message in order. Adjacent
messages with the same role remain separate.

Override a node's canonical `render()` when the representation must persist in
future history:

```python
import rlmflow
from rlmflow import ExecOutput


class LabeledOutput(ExecOutput):
    def render(self):
        return [
            {
                "role": "user",
                "content": "REPL OUTPUT:\n" + self.content,
            }
        ]
```

Use a `render_fn` for live material which applies only to the current frontier.
`Flow` passes its runtime explicitly, so the renderer can inspect the current
agent's REPL without closing over the flow. This is the renderer used by the
Shepherd example:

```python
def render_worker(runtime: Runtime, node: Node) -> list[dict[str, str]]:
    messages = default_render(runtime, node)
    content = board_prompt(runtime, node.parent_agent, simple=simple_moves)
    if content:
        messages.append({"role": "user", "content": content})
    return messages


flow = rlmflow.Flow(
    llm,
    prompt_profiles={
        "worker": PromptProfile(render_fn=render_worker),
    },
)
```

`RenderFn` is `(Runtime, Node) -> list[dict[str, str]]`. Keep `Node.render()`
runtime-independent so saved graphs retain a canonical projection; use the
current renderer for transient state such as REPL `ENV`.

A `UserQuery` subclass inherits the user turn with no builder edit. Node types
that are tree bookkeeping rather than turns (`ExecAction`, `DoneOutput`)
render as `[]`. To customize an engine instruction, subclass its typed node or
replace the relevant `StepFunction`.
