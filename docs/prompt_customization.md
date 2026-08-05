# Prompt Customization

`Flow` builds a system prompt from named sections. Most customization should
derive from the default builder instead of replacing the whole prompt, because
the default sections carry the REPL protocol, the
`launch_subagents` delegation rules,
`INPUTS`, `HISTORY`, and the worked examples that keep recursive execution
well-formed.

Use full replacement only when you want to own that entire protocol yourself.

## Inspect The Prompt

Before changing the prompt, render the one your agent already sees:

```python
import rlmflow

flow = rlmflow.Flow(llm)
root = rlmflow.start("Summarize this document.", inputs={"document": document})
print(flow.build_system_prompt(root))
```

You can render without starting a run by constructing the Node shape you
want to inspect:

```python
import rlmflow

flow = rlmflow.Flow(llm)
root = rlmflow.start("Summarize this document.")
print(flow.build_system_prompt(root))
```

`build_system_prompt(node)` renders against the Node's active query settings and
the Flow's current prompt/tool configuration.

## Default Builder Shape

The default builder has these sections, in order:

| Section | Purpose |
| --- | --- |
| `role` | Opening contract + REPL namespace. |
| `strategy` | Orchestrator principles: probe inputs, decompose/fanout, truncation, fix failures before `done()`. |
| `format` | One `repl` block per turn; use `print(...)` for inspection. |
| `examples` | Core recipes (observe inputs, fan out slices, delegate). |
| `final` | `done(...)` contract and repair discipline. |
| `structured-output` | Per-agent `done(value)` schema when the agent has an `output_schema`. |
| `structured-output-option` | How to request structured output from subagents (only when `enable_structured_output=True` and the agent can spawn children). |
| `tools` | Runtime-generated tool list (custom tools registered with the runtime, plus extra model aliases). |
| `inputs` | Runtime-generated manifest of the agent's `INPUTS`. |
| `status` | Runtime-generated agent depth / spawn-budget status. |
| `first-turn` | Bootstrap-only safeguard; drops out once the agent has produced any `llm_output`. |

The static text sections render back-to-back so the prompt reads as one
continuous narrative; the split exists so each piece is independently swappable
via `prompt.sections.update(name, ...)`. `tools`, `inputs`, `status`,
`structured-output*`, `examples`, and `first-turn` are callable sections filled
from the current Flow and Node at build time.

## Recommended: Edit `SystemPromptBuilder().sections`

The system prompt is a `SystemPromptBuilder` — symmetric to the
`UserPromptBuilder` on the user side. Its `.sections` is a mutable,
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

agent = rlmflow.Flow(llm, max_depth=2, system_prompt=prompt)
# or set it after construction:
agent.system_prompt = prompt
```

### Swap A Single Section

`update` swaps a section's body, keeping its position:

```python
domain_strategy = """
**When to delegate:** spawn one child per independent file/module. Keep the root
agent's job to planning, dispatch, and integration. Verify children mechanically
before `done()`.
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
`default_sections` (the static baseline). This is the system-side analogue of
overriding a `render_*` method on `UserPromptBuilder`.

```python
from rlmflow import SystemPromptBuilder

class AuditPrompt(SystemPromptBuilder):
    def default_sections(self):
        return super().default_sections().add(
            "rules", RULES, title="Rules", after="final"
        )

agent = rlmflow.Flow(llm, system_prompt=AuditPrompt())
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
- Call `done(answer)` exactly once when finished.
- Use tools to inspect or modify files.
""",
                title="Protocol",
            )
            .add("tools", tools_section, title="Tools")
            .add("status", status_section, title="Status")
        )

agent = rlmflow.Flow(llm, system_prompt=MinimalPrompt())
```

## The `system_prompt` Source

`system_prompt` (constructor arg or settable attribute) accepts any of three
things — a `SystemPromptBuilder`, a plain string, or a `(flow, node) -> str`
function — and is resolved fresh on every turn:

- **`SystemPromptBuilder`** (the default, `DEFAULT_BUILDER`) — the section
  machinery described above.
- **string** — a constant prompt, bypassing the builder entirely.
- **function** — a dynamic prompt without subclassing anything.

```python
import rlmflow

# constant string (most fragile — you own the whole protocol)
agent = rlmflow.Flow(
    llm,
    system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `done(answer)` exactly once when finished.
""",
)

# or a function of (flow, agent)
def prompt_for(flow, agent):
    depth = agent.config.depth
    tail = "Return an executive summary." if depth == 0 else "Return findings only."
    return f"You are an auditor. {tail}"

agent = rlmflow.Flow(llm, system_prompt=prompt_for)
```

A string that omits `launch_subagents`, `INPUTS`, `HISTORY`, or the `done(...)`
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


agent = rlmflow.Flow(llm, system_prompt=AuditPrompt())
```

You can also replace narrower callable sections directly:

```python
from rlmflow import SystemPromptBuilder
from rlmflow.prompts import tools_section


def careful_tools(flow, node):
    return tools_section(flow, node) + "\n- Prefer read-only tools before write tools."


prompt = SystemPromptBuilder()
prompt.sections.update("tools", careful_tools)
```

(Overriding `Flow.build_system_prompt` still works if you'd rather own prompt
selection at the flow level.)

## Callable Sections

The dynamic prompt hook above works, but it is heavier than it needs to be for
small additions like project rules or runtime notes. A prompt section can be
either static text or a function:

```python
def section(flow: rlmflow.Flow, node: rlmflow.Node) -> str:
    ...
```

The signature is intentionally just `flow, node`. There is no context dict
and no separate prompt context object. If a section needs runtime tools, model
registrations, config, or the current agent id, those are already reachable
from `flow` and `node`.

## Child-Specific Prompts

The easiest way to steer a child is the query you pass in a
`launch_subagents([...])` spec. Use the global prompt for stable behavior and
use child queries for local contracts.

```python
results = await launch_subagents([
    {
        "name": "api",
        "query": "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "inputs": {"spec": api_spec},
    },
    {
        "name": "tests",
        "query": "Implement tests for src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        "inputs": {"spec": test_spec},
    },
])
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

A `PromptProfile` bundles a `system` and a `user` source; `None` on either side
inherits the flow default. When omitted from `prompt_profiles`, `"default"`
means the flow's own `system_prompt`/`user_prompt`; callers may also define it
explicitly.

By default, Flow reads the profile name stamped on the Node's latest
`UserQuery`. With a non-empty registry, profile names and descriptions are
advertised in the orchestrator's system prompt, so it can name one per child:

```python
await launch_subagents([
    {"name": "impl",   "query": "...", "prompt_profile": "coder"},
    {"name": "review", "query": "...", "prompt_profile": "reviewer"},
])
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

The spawn spec's `prompt_profile` is stored on `UserQuery.prompt_profile` and
serialized. When omitted, a cold child inherits its immediate parent agent's
profile; a prepared branch keeps its own profile. Without a router, that stamp
is authoritative. With a router, the callable's result is authoritative.
Unknown names raise `ValueError`.

## Customizing The User Turns

Everything above shapes the *system* message. The rest of the conversation — the
user query, the assistant's replies, and the REPL/observation turns fed back in
— is produced by the **user prompt source**, symmetric to `system_prompt`:

- **`UserPromptBuilder`** (the default) — prepares the Node trajectory for this turn
  (optional per-turn content + continue/forced-final nudge as real `UserQuery`
  nodes), then projects the trajectory into chat turns (`render_*` per node type).
- **function** — a `(flow, node) -> str | None` build hook. Wrapped as
  `UserPromptBuilder(build_fn=…)` so you can pass it as `user=` /
  `user_prompt=` without subclassing. Return a string to commit as a `UserQuery`
  this turn (e.g. a live observation), or `None` for nothing.

`Flow.messages` takes the node the prompt is being built for, projects backwards
from it, keeps the last `keep_n_messages` turns (replacing what it dropped with a
truncation notice), prepends the system message, and coalesces adjacent same-role
turns into one (so an injected instruction landing
right after a REPL output becomes a single user message — required by chat APIs
that reject two same-role turns in a row).

Override one method to re-shape how a single node type becomes a message (return
`None` to drop it):

```python
import rlmflow
from rlmflow import UserPromptBuilder


class LabeledUser(UserPromptBuilder):
    def render_exec_output(self, node):
        return {"role": "user", "content": "REPL OUTPUT:\n" + (node.content or node.output)}


agent = rlmflow.Flow(llm, user_prompt=LabeledUser())
```

The overridable renderers are `render_user_query`, `render_llm_output`,
`render_exec_output`, `render_error_output`, and `render_supervising_output`,
dispatched by `render_node`. Node types that are tree bookkeeping rather than
turns (`ExecAction`, `DoneOutput`) are dropped.

To change the nudge or truncation wording, subclass `Flow` — they are class
attributes:

```python
class TerseFlow(rlmflow.Flow):
    continue_nudge = "Continue."
    final_action = "Give your final answer now via done(...)."
```
