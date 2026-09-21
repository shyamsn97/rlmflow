# Prompt Customization

`Flow.build_system_prompt(node)` renders the inherited protocol prompt. `UserQuery.build_system_prompt(flow)` is what a turn actually sends: the default is that same string, and query nodes wrap it. Actions and outputs have no prompt hook. User turns are `Turn N/M:` plus a first-turn “look at INPUTS first” safeguard.

Most customization should subclass `PromptBuilder` and concatenate after `super().__call__(flow, node)`, because the default template carries the REPL protocol and live API list. Use a full string replacement only when you want to own that entire protocol yourself.

## Inspect The Prompt

Before changing the prompt, render the one your agent already sees. Nothing has to run first — a root from `flow.start(...)` is enough:

```python
import rlmflow

flow = rlmflow.Flow(llm)
root = flow.start("Summarize this document.", inputs={"document": document})
print(flow.build_system_prompt(root))
```

That is the inherited protocol, capability-gated work policy, tools, INPUTS sizes, and depth. `flow.build_messages(node)` returns the full message list; on a `UserQuery` the first entry is its system text, and on any other node it is the inherited prompt.

## Default Shape

The default `PromptBuilder` fills one official-style template:

- You are a Recursive Language Model with a Python REPL that persists across turns.
- `INPUTS` is the long context; inspect it with `print(...)`.
- A spawn-capable agent gets a compact allocation, graph, and integration policy. A leaf gets a compact assigned-work policy.
- Live REPL bindings (`finish`, `SHOW_VARS`, `launch_subagent`, `llm_query*` when enabled, plus host tools) fill `{tools}`.
- Tool guidance contains one local example plus, when spawning is available, one parallel example and one dependency-handle example.
- REPL stdout over ~20K characters is truncated; only printed stdout is shown back.
- Do not `finish(...)` on turn 1 without inspecting `INPUTS`.

`Flow.render_tools(node)` is the dynamic “Available in the REPL” list. `Flow.render_inputs(node)` is INPUTS sizes plus per-agent schema and depth.

`PromptBuilder` selects the graph policy when the current agent can spawn and the assigned-work policy when it cannot. This is capability gating, not a request to delegate. `PlanQuery` adds only a short checklist for the next objective, work ownership, known dependencies, and one executable REPL block. `FinalQuery.build_system_prompt` appends last-turn submit guidance and drops the “don’t finish on turn 1” line.

## Recommended: Subclass `PromptBuilder`

`PromptBuilder` is equivalent to a `(flow, node) -> str` function. Anything that depends on the live flow is a named method on `Flow` (`render_tools`, `render_inputs`), not a private helper.

```python
from rlmflow import PromptBuilder

class SkillsPrompt(PromptBuilder):
    def __init__(self, library):
        self.library = library

    def __call__(self, flow=None, node=None) -> str:
        extra = self.library.render()
        text = super().__call__(flow, node)
        return f"{text}\n\n{extra}" if extra else text

flow = rlmflow.Flow(llm, system_prompt=SkillsPrompt(library))
```

The same pattern covers project rules, a persona, or an adapter prompt: concatenate after `super().__call__`. `PlanQuery` still sees those extras because it starts from `flow.build_system_prompt(self)`.

### Add Project Rules

```python
from rlmflow import PromptBuilder

PROJECT_RULES = """
- Preserve API compatibility unless the task explicitly asks for a breaking change.
- Prefer small patches with focused tests.
"""

class ProjectPrompt(PromptBuilder):
    def __call__(self, flow=None, node=None) -> str:
        return super().__call__(flow, node) + "\n\n" + PROJECT_RULES.strip()

flow = rlmflow.Flow(llm, system_prompt=ProjectPrompt())
```

### Replace The Protocol

A string with `{tools}` is a template filled by `flow.render_tools(node)`. Any other string is a constant replacement.

````python
flow = rlmflow.Flow(
    llm,
    system_prompt="""
You are a Python REPL agent.

- Use exactly one ```repl``` block per assistant message.
- Use available tools to make progress.
- Call `finish(answer)` exactly once when finished.
""",
)
````

A `(flow, node) -> str` function is also accepted:

```python
def prompt_for(flow, node):
    depth = node.parent_agent.config.depth
    tail = "Return an executive summary." if depth == 0 else "Return findings only."
    return f"You are an auditor. {tail}"

flow = rlmflow.Flow(llm, system_prompt=prompt_for)
```

A string that omits `launch_subagent`, `INPUTS`, or the `finish(...)` rule means the model will not reliably use those features — prefer a `PromptBuilder` subclass unless you intend to own the entire protocol.

## The `system_prompt` Source

`system_prompt` (constructor arg or settable attribute) accepts a `PromptBuilder`, a plain string, or a `(flow, node) -> str` function, and is resolved fresh on every turn. `None` uses `PromptBuilder()`.

`Flow.build_system_prompt(node)` picks `profile.system` or `flow.system_prompt` or `PromptBuilder()`. `UserQuery.build_system_prompt(flow)` defaults to that string; query nodes wrap it. `can_spawn(agent)` is `rlmflow.graph.config.can_spawn` (also re-exported from `rlmflow.graph.nodes` and `rlmflow.prompts.prompts`):

```python
class PromptBuilder:
    def __call__(self, flow=None, node=None) -> str:
        agent = as_agent(node)
        policy = EXECUTION_POLICY if can_spawn(agent) else WORKER_ADDENDUM
        return RLM_SYSTEM_PROMPT.format(
            execution_policy=policy,
            tools=flow.render_tools(node),
        )
```

A custom node can still replace or wrap its system prompt:

```python
class ReviewQuery(UserQuery):
    def build_system_prompt(self, flow) -> str:
        return flow.build_system_prompt(self) + "\n\nReview rules."
```

Methods live on the class, so they survive load. There is no `prompt_extra`, no `Flow(orchestrator=)`, and no `Node.system_prompt` ClassVar.

## Child-Specific Prompts

The easiest way to steer a child is the goal you pass to `launch_subagent`. Use the global prompt for stable behavior and child goals for local contracts.

```python
api = await launch_subagent(
        "Implement src/api.py. Return ONLY JSON {\"files\": [str], \"checks\": [str]}.",
        model="default",
        name="api",
        inputs={"spec": api_spec},
)
```

## Per-child prompts

Sometimes a child agent should run under a *different* prompt than its parent — an orchestrator RLM spawning coding agents, say. Register named **prompt profiles** on the flow, parallel to `llm_clients`/`model`:

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

A `PromptProfile` bundles a `system` source and a current-node `render_fn`; `None` on either side inherits the flow default. When omitted from `prompt_profiles`, `"default"` means the flow's own `system_prompt`/`render_fn`; callers may also define it explicitly.

By default, Flow reads the profile name from the agent's config. With a non-empty registry, profile names and descriptions are advertised in the orchestrator's system prompt, so it can name one per child:

```python
impl = await launch_subagent(
    "...", model="default", name="impl", prompt_profile="coder"
)
```

Pass a callable `prompt_router` only when host policy should choose the profile dynamically. A custom router also suppresses profile advertising:

```python
flow = rlmflow.Flow(
    llm,
    prompt_profiles={"coder": CODER},
    prompt_router=lambda flow, agent: "coder" if agent.config.depth > 0 else "default",
)
```

The launch call's `prompt_profile` is stored on the child config and serialized. When omitted, a cold child inherits its immediate parent agent's profile. Without a router, that stamp is authoritative. With a router, the callable's result is authoritative. Unknown names raise `ValueError`.

## Customizing The User Turns

Everything above shapes the *system* message. The rest of the conversation — the user query, the assistant's replies, and the REPL/observation turns fed back in — has two explicit layers:

- **canonical history** — each `Node.render()` returns a list of messages, and `node.project()` flattens those lists while walking history;
- **current frontier** — `Flow(render_fn=...)` or `PromptProfile(render_fn=...)` may render the node currently being sent differently without rewriting historical projection.

The default current renderer calls `node.render()` and adds live background-agent status. Recurring plan-and-act, final-answer, and truncation instructions are typed nodes, not injected strings.

`PlanQuery.instruction()` is `Turn {n}/{max}:` plus a short checklist: next objective, local or owned work, known dependencies, then one REPL block. Turn 1 also includes the official first-turn safeguard (“you have not interacted with the REPL… look at INPUTS first”). Last-budget-turn user content stays on `FinalQuery`; the submit instruction lives on `FinalQuery.build_system_prompt`. `format_transition_footer` is engine bookkeeping, not prompt policy.

`Flow.build_messages` reserves the current renderer's messages inside `keep_n_messages`, projects the remaining capacity from `node.prev`, prepends `UserQuery.build_system_prompt(self)` when the frontier is a query (otherwise the inherited protocol), and preserves every rendered message in order. Adjacent messages with the same role remain separate.

Override a node's canonical `render()` when the representation must persist in future history:

```python
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

Use a `render_fn` for live material which applies only to the current frontier. `Flow` passes its runtime explicitly, so the renderer can inspect the current agent's REPL without closing over the flow:

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

`RenderFn` is `(Runtime, Node) -> list[dict[str, str]]`. Keep `Node.render()` runtime-independent so saved graphs retain a canonical projection; use the current renderer for transient state such as REPL `ENV`.

A `UserQuery` subclass inherits the user turn with no builder edit. Node types that are tree bookkeeping rather than turns (`ExecAction`, `DoneOutput`) render as `[]`. To customize an engine instruction, subclass its typed node or register `@Flow.transitions.on(...)`.
