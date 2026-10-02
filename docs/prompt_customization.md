# Prompt Customization

rlmflow uses one mutable `Prompt` class for both system and user prompts. A `Prompt` owns ordered, keyed sections and renders to one string. Message roles are assigned only when the prompts are converted into chat messages.

```python
system = flow.build_system_prompt(node)
user = flow.default_user_prompt(node)

print(system)
print(user)
```

The model still receives ordinary messages:

```python
[
    {"role": "system", "content": str(system)},
    *history,
    *node.message_dicts(),
    *live_request_overlays,
]
```

`Node.build_messages(user, flow)` creates `node.messages` once. Later requests, save/load, and forks reuse that immutable snapshot instead of rerunning prompt or runtime-dependent code.

## Default Shape

The default system `Prompt` contains independently variable concerns:

- `protocol.core`: the pinned REPL-only action and completion contract;
- `policy.evidence`: evidence acquisition, validation, and failure transparency;
- `policy.execution`: spawn-capable or assigned-work policy;
- `capabilities.tools`: active REPL tools, when present;
- `context.arguments`: direct argument names and sizes, when present;
- `contract.output_schema`: the pinned structured-output contract, when required;
- `capabilities.depth`: recursion information, when recursion is enabled.

The current user `Prompt` normally contains `turn.instruction`: the query, continuation, or final-turn instruction. When the agent has children, a `Background subagents:` status list is appended to the request only; it is never stored in the node's message snapshot.

The exactly-one-REPL-block rule appears once in `protocol.core`. Ordinary turns do not repeat it.

## The `Prompt` API

Callers normally add text without constructing `PromptSection` themselves:

```python
prompt = rlmflow.Prompt()
prompt.add_section(
    "project.rules",
    "Preserve API compatibility and run focused tests.",
    order=800,
    source="project",
)

prompt.update_section(
    "project.rules",
    "Preserve compatibility, run tests, and verify generated artifacts.",
)
prompt.remove_section("project.rules")
```

Inspection is keyed:

```python
section = prompt.get("protocol.core")
for section in prompt.sections():
    print(section.key, section.order, section.source, section.pinned)
```

Replacing or removing a pinned section raises `PinnedPromptSectionError`.

## Reusable System Prompt Modifiers

Use a modifier to add reusable policy to Flow's fresh default system `Prompt`:

```python
from pathlib import Path

def add_project_prompt(prompt, flow=None, node=None):
    prompt.add_section(
        "project.rules",
        Path("AGENTS.md").read_text(),
        order=800,
        source="add_project_prompt",
    )
```

```python
flow = rlmflow.Flow(llm, system_prompt_modifiers=add_project_prompt)
```

`Flow.default_system_prompt(node)` creates a fresh default and applies configured and profile modifiers. `Flow.build_system_prompt(node)` then passes that prompt through the node's system hook.

## Flow-Level Modifiers

A modifier edits a `Prompt` in place:

```python
def add_project_rules(prompt, flow, node):
    prompt.add_section(
        "project.rules",
        "Prefer small patches with focused validation.",
        order=800,
        source="project",
    )


flow = rlmflow.Flow(
    llm,
    system_prompt_modifiers=add_project_rules,
)
```

Pass one modifier or an ordered iterable:

```python
flow = rlmflow.Flow(
    llm,
    system_prompt_modifiers=[
        add_project_rules,
        add_domain_playbook,
    ],
)
```

User-prompt modifiers use the same `Prompt` API:

```python
def add_review_instruction(prompt, flow, node):
    prompt.add_section(
        "review.requirements",
        "Check correctness and API compatibility.",
        order=1050,
        source="review",
    )


flow = rlmflow.Flow(
    llm,
    user_prompt_modifiers=add_review_instruction,
)
```

## Node-Owned Current-Turn Messages

Flow prepares the user `Prompt`; query nodes may edit it and decide how many current-turn messages to emit:

```python
from dataclasses import dataclass

from rlmflow import UserQuery


@dataclass
class ReviewQuery(UserQuery):
    def build_messages(self, user_prompt, flow):
        user_prompt.add_section(
            "review.policy",
            "Review carefully and report only actionable findings.",
            order=1010,
            source=type(self).__name__,
        )
        return super().build_messages(user_prompt, flow)
```

Nodes return only current-turn messages. Flow remains responsible for the system message, projected history, and truncation.

## Profiles

Profiles may replace the system source and independently modify system and user prompts:

```python
from rlmflow import PromptProfile


def add_research_policy(prompt, flow, node):
    prompt.add_section(
        "profile.research",
        "Prefer primary sources and record publication dates.",
        order=800,
        source="research",
    )


flow = rlmflow.Flow(
    llm,
    prompt_profiles={
        "research": PromptProfile(
            system_modifiers=add_research_policy,
        )
    },
)
```

Global modifiers apply first; profile modifiers may update the same non-pinned key.

## Live Harness State

Keep mutable state in the harness and project it into each newly built `Prompt`:

```python
memory: dict[str, str] = {}


def apply_memory(prompt, flow, node):
    for key, lesson in memory.items():
        prompt.add_section(
            f"memory.{key}",
            lesson,
            order=850,
            source="live-memory",
        )


flow = rlmflow.Flow(
    llm,
    system_prompt_modifiers=apply_memory,
)
```

ACE-style CRUD is ordinary host mutation:

```python
memory["inspection"] = "Inspect bounded samples before drawing conclusions."
memory["inspection"] = "Inspect bounded samples and verify writes afterward."
del memory["inspection"]
```

Every model turn builds fresh system and user `Prompt` instances, so the next turn observes current harness state without a special collection abstraction.

## Whole-System Overrides

A string still owns the complete system protocol:

````python
flow = rlmflow.Flow(
    llm,
    system_prompt="""
Use the persistent REPL for every turn.

Include exactly one non-empty fenced ```repl``` block. Call finish(value)
inside that block to complete the run.
""",
)
````

A `(flow, node) -> str` callable is also supported. Flow-level system modifiers may add sections after the override.

Use whole-system replacement only when you intentionally own the runtime contract. For normal customization, modify the default `Prompt`.

## Composition Order

System prompt:

```text
Flow.default_system_prompt(node)
→ global system modifiers
→ profile system modifiers
→ node.build_system_prompt(system_prompt, flow)
```

User prompt:

```text
Flow.default_user_prompt(node)
→ global user modifiers
→ profile user modifiers
→ node.build_messages(user_prompt, flow)
→ durable node message snapshot

Flow.build_messages(node)
→ copy durable current messages
→ append live background-agent state and named transitions
```

Adding the same non-pinned key updates it without duplicating text. Pinned runtime and schema sections reject conflicting updates.

## Multiple Current-Turn Messages

Override `Node.build_messages` when one turn needs more than one message:

```python
class BoardQuery(UserQuery):
    def build_messages(self, user_prompt, flow):
        return [
            {"role": "user", "content": render_board(flow.runtime, self)},
            {"role": "user", "content": str(user_prompt)},
        ]
```

The supplied `user_prompt` already contains configured/profile modifiers. The result is snapshotted, so later history sees the same board rather than recomputing it from current runtime state. Live background and transition sections are added afterward by Flow and are not persisted. A node may use `flow` for runtime state, but must not assemble history or call `flow.default_*_prompt()` itself.

## Arguments and tools

Argument values are not copied into the system prompt. The prompt contains names and sizes; the values are direct variables in the persistent REPL.

Tool documentation is rendered directly from `Flow.build_tools(node)`, the complete map of Flow-owned, application, and dynamically bound environment tools. `Flow.build_namespace(node)` adds non-tool execution bindings afterward. `when=` predicates hide application tools from both the prompt and execution.

## Actions and Completion

The prompt architecture does not override runtime routing:

```text
exactly one non-empty repl → execute block
anything else             → protocol error; execute nothing
finish(value)             → submit the final JSON-compatible value
```

`finish` is available from an agent's first block. The pinned `protocol.core`
asks the model to finish immediately when code can derive the answer from its
inputs and to look first when their size, format, or contents are unknown.
Prompt modifiers can refine the replaceable evidence policy, but cannot remove
the pinned protocol or make an unavailable tool appear in the rendered tool
list.

System and user prompts use the same `Prompt` class, but their message roles and owners remain distinct:

```text
Flow builds fresh system/user defaults
node hooks customize system and shape current-turn messages
Flow assembles system, history, and current messages
```
