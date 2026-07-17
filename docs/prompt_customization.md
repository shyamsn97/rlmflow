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
graph = rflow.Graph(query="Summarize this document.", inputs={"document": document})
print(agent.build_system_prompt(graph))
```

You can also render without starting a run by constructing the graph shape you
want to inspect:

```python
import rflow

graph = rflow.Graph(query="Summarize this document.", agent_id="root", depth=0)
print(agent.build_system_prompt(graph))
```

`build_system_prompt(graph)` is deterministic for a given `graph`, so the prompt
an agent saw is always reproducible from its saved `Graph`.

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
via `DEFAULT_BUILDER.update(name, ...)`. `tools`, `inputs`, `status`,
`structured-output*`, `examples`, and `first-turn` are callable sections filled
from the current flow and graph at build time.

## Recommended: Derive From `DEFAULT_BUILDER`

The default prompt is a `PromptBuilder`: an ordered list of named sections.
`.section(...)` returns a new builder, so the module-level default is never
mutated.

### Add Project Rules

Add a new section anywhere relative to the existing ones:

```python
import rflow
from rflow.prompts import DEFAULT_BUILDER

project_rules = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
- When changing public behavior, update docs in the same pass.
"""

prompt = DEFAULT_BUILDER.section(
    "project_rules",
    project_rules,
    title="Project Rules",
    after="final",
)

agent = rflow.Flow(llm, max_depth=2)
agent.prompt_builder = prompt
```

### Swap A Single Section

Replace just the piece you want to customize. The rest of the prompt is unchanged:

```python
from rflow.prompts import DEFAULT_BUILDER

domain_strategy = """
**When to delegate:** spawn one child per independent file/module. Keep the root
agent's job to planning, dispatch, and integration. Verify children mechanically
before `done()`.
"""

prompt = DEFAULT_BUILDER.update("strategy", domain_strategy)
```

### Prepend A Persona

Slip a small role section before `role` rather than overwriting the protocol:

```python
prompt = DEFAULT_BUILDER.section(
    "persona",
    "You are a recursive security auditor. Reproduce concrete risks and "
    "propose minimal fixes.",
    title="Persona",
    before="role",
)
```

### Remove A Section

You can remove sections, but be careful — removing core sections like `role`,
`strategy`, `format`, or `final` strips the delegation/REPL protocol. Prefer
removing only sections you added.

```python
prompt = DEFAULT_BUILDER.remove("project_rules")
```

### Build A Prompt From Scratch

Use this when you want complete control while still using the section renderer.
If you want the standard runtime-generated tools and status blocks, include the
built-in callable sections.

```python
from rflow.prompts import PromptBuilder, status_section, tools_section

prompt = (
    PromptBuilder()
    .section("role", "You are a minimal REPL agent.", title="Role")
    .section(
        "protocol",
        """
- Use exactly one ```repl``` block per assistant message.
- Call `done(answer)` exactly once when finished.
- Use tools to inspect or modify files.
""",
        title="Protocol",
    )
    .section("tools", tools_section, title="Tools")
    .section("status", status_section, title="Status")
)
```

## Full System Prompt Replacement

`Flow(system_prompt=...)` bypasses the builder entirely:

```python
import rflow

agent = rflow.Flow(
    llm,
    system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `done(answer)` exactly once when finished.
""",
)
```

This is the most fragile option. If the prompt omits `launch_subagents`, `INPUTS`, `HISTORY`, or the
`done(...)` rule, the model will not reliably use those features.

## Dynamic Prompts

Subclass `Flow` when the prompt should depend on the current agent,
depth, query, available tools, or project state. The hook receives the
agent's `Graph` — all run-invariants are flat fields on it
(`agent_id`, `depth`, `query`, `inputs`, `model`, …).

```python
import rflow
from rflow.prompts import DEFAULT_BUILDER


class AuditFlow(rflow.Flow):
    def build_system_prompt(self, graph: rflow.Graph) -> str:
        extra = (
            "At root depth, produce an executive summary after verification."
            if graph.depth == 0
            else "As a child call, return only structured findings."
        )

        builder = DEFAULT_BUILDER.section(
            "audit_depth_rules",
            extra,
            title="Depth Rules",
            after="strategy",
        )
        return builder.build(self, graph)
```

You can also replace narrower callable sections directly:

```python
from rflow.prompts import tools_section


def careful_tools(engine, graph):
    return tools_section(engine, graph) + "\n- Prefer read-only tools before write tools."


prompt = DEFAULT_BUILDER.section("tools", careful_tools, title="Tools")
```

## Callable Sections

The dynamic prompt hook above works, but it is heavier than it needs to be for
small additions like project rules or runtime notes. A prompt section can be
either static text or a function:

```python
def section(flow: rflow.Flow, graph: rflow.Graph) -> str:
    ...
```

The signature is intentionally just `flow, graph`. There is no context dict
and no separate prompt context object. If a section needs runtime tools, model
registrations, config, or the current agent id, those are already reachable
from `flow` and `graph`.

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

If every child of a flow needs a different system prompt, use a subclass and
branch on `graph.depth`, `graph.agent_id`, or `graph.query`.
