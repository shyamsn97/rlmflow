# Historical review: a behavior lives in one file

Status: historical design record. The requested-transition path described below was removed after this review exposed that its generic `ExecOutput` row preempted the guarded landing row. In 0.7.0 the successor `.choices(...)` / `transition("name")` feature was removed as well; every edge is now an automatic transition function. `PlanQuery`, `to_plan`, `child_returned`, and the synthetic turn message were subsequently removed; ordinary frontier nodes now call the model directly. `tools/builtins.py` now contains the Flow-owned `submit` and `llm_query` tools; `graph.config.can_spawn` is the one spawn predicate. `graph/prompts.py` was merged with the protocol text into `prompts/text.py`, a leaf module both `graph` and `prompts` import. The remaining names and diagrams on this page are intentionally historical.

The current design is smaller:

- default automatic policy sits in `steps/defaults.py`;
- the reusable `Transitions` table type sits in `steps/transitions.py`;
- `.choices(CurrentQuery, *targets)` declares model-selectable routes;
- the always-bound `transition("name")` tool is documented only when choices exist;
- `run_repl` resolves a valid choice directly from `ExecAction` to the selected `UserQuery`;
- there is no intermediate `ExecOutput`, `requested_transition` field, `has_request` guard, or `apply_requested` producer.

See [`step_table.md`](step_table.md) for the current API and transition map. The rest of this page preserves the original hop audit that motivated the simplification; names and snippets below are intentionally pre-refactor.

## The original hop

```57:70:rlmflow/steps/defaults.py
@Flow.transitions.on(ExecOutput, when=has_request)
def apply_requested(flow: Flow, node: Node) -> Node:
    selected = requested_transition(node)
    if selected is None:
        raise TransitionProtocolError("transition(...) is only valid during a working turn")
    option = flow.transitions.resolve_offer(node, selected)
    ...
    return option.target()
```

`has_request` is not a local predicate. It is a pointer into `transitions.py`, which is a pointer into `graph/utils.py`, which peeks at `ExecAction.requested_transition`, which was stamped in `node_for_run` after `transition_tool` raised `TransitionSignal` in the REPL. The decorator that looks like the whole story is the last hop of eight.

```text
tools/namespace.py      transition() raises TransitionSignal
runtime/repl.py         ReplRun(status=TRANSITION, transition=name)
steps/defaults.py         node_for_run stamps ExecAction.requested_transition
graph/utils.py          requested_transition(node) peeks at prev
steps/transitions.py   has_request / chose / resolve_offer / available
steps/defaults.py         apply_requested lands option.target()
prompts/messages.py     transition_footer advertises the menu
graph/nodes.py          UserQuery.name / transition_description / TurnMode
```

Eight files for `transition("verify")`. The REPL, the node payload, and the prompt phrasing are real layers. The mess is the three engine hops in the middle: guard, accessor, producer.

## What went wrong

Extract-by-kind:

| Kind            | Landed in                                                    |
| --------------- | ------------------------------------------------------------ |
| table type      | `steps/transitions.py` (`Transitions`, `Row`, `Offer`)       |
| default rows    | `steps/defaults.py`                                          |
| default guards  | `steps/transitions.py` (bottom of the same file as the type) |
| tiny accessors  | `graph/utils.py`                                             |
| turn strings    | `graph/prompts.py`                                           |
| REPL callables  | `tools/namespace.py`                                         |
| prompt phrasing | `prompts/messages.py`                                        |

A kind-of-helper split is cheap to write and expensive to read. `has_request` exists so `steps.py` does not import `requested_transition` — except `steps.py` imports it anyway, so the guard is a name, not a boundary.

`chose(name)` is the same family and has **no callers**. The generic dispatcher (`has_request` → `resolve_offer`) won; the per-name guard was left behind.

## Full hop audit

Every default row, and every other story that currently takes more than two files.

### Default table (what `Flow.step` actually runs)

Registration order is the policy. First matching row wins. Own rows, then `_base`.

```text
SOURCES + out_of_room        → to_final          → FinalQuery
SOURCES + needs_truncation   → to_summary       → TruncationSummary
PLAN_SOURCES                 → to_plan          → PlanQuery
UserQuery + child_returned    → to_plan          → PlanQuery
ExecOutput + has_request     → apply_requested  → offered UserQuery
UserQuery                    → complete        → LLMOutput | DoneOutput
LLMOutput + repl             → to_action       → ExecAction
LLMOutput + invalid protocol → to_missing_action → ExecAction
ExecAction                   → run_repl        → node_for_run(...)
```

`step_table.md` prints the first four producers and skips `apply_requested`. The named-transition land is the leftover call form, so it is easy to miss even in the doc that claims to be the whole table.

### Last turn / budget — 6 files

```text
steps/transitions.py   at_final, budget_nearly_spent, out_of_room
flow.py                 budget_exceeded (hard stop; different predicate)
steps/defaults.py         to_final
graph/prompts.py        FINAL_ANSWER_ACTION
graph/nodes.py          FinalQuery.modify_prompt
```

Two budget policies. `out_of_room` is the soft nudge: one more average turn would pass `max_budget`, or this is the last `max_iters` turn, so inject `FinalQuery`. `Flow.budget_exceeded` is the hard stop: tokens already spent, skip the model, land `[budget exceeded]`. They share a name-shaped concern and live in different files, so a reader of `to_final` never sees the hard gate.

### Plan turn — 5 files

```text
steps/transitions.py  PLAN_SOURCES, child_returned
steps/defaults.py        to_plan
graph/prompts.py       USER_PROMPT
graph/nodes.py         PlanQuery.modify_prompt
```

`to_plan` is three lines. `PlanQuery` now owns its compact user `Prompt` instruction directly. `child_returned` is `type(node) is UserQuery and isinstance(node.prev, DoneOutput)` — a one-liner that exists only so the decorator does not say that.

### Truncation — 5 files

```text
steps/transitions.py  needs_truncation
steps/defaults.py        to_summary
graph/prompts.py        TRUNCATION_SUMMARY
graph/nodes.py          TruncationSummary
prompts/messages.py     build_messages re-finds the summary to prepend
```

The row creates the node. `build_messages` walks backwards for the same node to splice its render into the kept window. Two sites, one story.

### `complete` — 3 files, acceptable

```text
steps/defaults.py        complete (max_iters hard stop, model call)
flow.py                 llm_for_step, build_messages wrapper
prompts/messages.py     build_messages
```

The producer is the story. Prompt assembly is a layer. This is the shape to keep: **policy in the row, rendering in prompts**.

### `run_repl` — 4 files, mostly a real layer

```text
steps/defaults.py         run_repl, node_for_run, _turn_mode
runtime/repl.py         ReplRun / ReplStatus / TransitionSignal
tools/namespace.py      finish() / transition() raise the signals
graph/nodes.py          TurnMode on the query that preceded the action
```

`node_for_run` is the policy (which status becomes which node). The REPL is a layer. `_turn_mode` walks `ExecAction → LLMOutput → query` because `TurnMode` lives on the query, not the action. That walk is engine policy sitting in `steps.py` already — good. Do not extract it.

### Delegation — 6 files, plus a drifted helper

```text
tools/namespace.py     build_namespace wires launch_subagent
tools/launch.py   launch_tool, resolve_child, new_child, submit_child
flow.py                 one-line wrappers
graph/nodes.py          AgentStart / AppendChild
graph/config.py         AgentConfig.child
graph/utils.py          _can_spawn
tools/builtins.py       a second _can_spawn
prompts/prompts.py      can_spawn
```

Three spawn predicates. They disagree on `None`:

| Function                    | `flow is None or agent is None` |
| --------------------------- | ------------------------------- |
| `graph.utils._can_spawn`    | `False`                         |
| `tools.builtins._can_spawn` | `True`                          |
| `prompts.prompts.can_spawn` | `True`                          |

`PlanQuery.build_system_prompt` uses the graph copy (no orchestrator addendum when unbound). Prompt docs use the prompt copy (show launch tools when unbound). That is not a layer. That is a copy that drifted.

`flow.py` still has `resolve_child` / `new_child` / `submit_child` wrappers. `flow_extract.md` already said those bodies belong in `delegation.py`. The wrappers are the facade. The three `can_spawn`s are the leftover mess.

### `graph/utils.py` is not a module

It is the drawer that caught whatever `nodes.py` did not want to keep:

| Helper                                            | Actual owner                                                  |
| ------------------------------------------------- | ------------------------------------------------------------- |
| `new_agent_id`, `new_node_id`, `system_prompt_id` | identity, used at construction                                |
| `active_step`, `running_step`, `AgentBusyError`   | append / in-flight (used by `execution.py`, `repl_client.py`) |
| `agent_payload`                                   | `to_record` on `AgentStart` / `UserQuery`                     |
| `working_instruction`                             | `PlanQuery.instruction`                                       |
| `requested_transition`                            | `apply_requested` / `has_request`                             |
| `_can_spawn`                                      | spawn policy (see above)                                      |
| `_isoformat`                                      | persistence                                                   |
| `_rebuild_index`                                  | `view/timeline.py`                                            |

A file named `utils` that both the engine and the graph import is how `has_request` grew a third hop. Delete the drawer by parking each helper with its caller.

`nodes.py` already re-exports most of these names. Callers can keep `from rlmflow.graph.nodes import requested_transition` until the method move below. They should never import `graph.utils`.

### Docs hop too

| Doc               | What it covers                 | What it omits                             |
| ----------------- | ------------------------------ | ----------------------------------------- |
| `step_table.md`   | decorator API, first four rows | `apply_requested`, guards, `node_for_run` |
| `internals.md`    | high-level graph of rows       | where a guard lives, named offers         |
| `node_model.md`   | node types, sample transcripts | the table that produces them              |
| `streaming.md`    | queue + one ASCII table        | same omission as step_table               |
| `flow_extract.md` | how Flow got thin              | that extract-by-kind split the rows       |

A reader of `apply_requested` is sent to `step_table.md`, which does not show that function.

## The rule

**Mechanism can live in its own file. Policy for a default row lives next to the producer.**

- `Transitions` / `Row` / `Offer` / errors stay in `steps/transitions.py`. That is a type. Tests construct bare `Transitions()` tables. Subclasses `derive()`. Keep it.
- Default guards, `SOURCES`, `PLAN_SOURCES`, and `requested_transition` move next to the `@Flow.transitions.on` functions that use them.
- Node payload fields stay on the node (`ExecAction.requested_transition`, `UserQuery.name`). That is data, not control.
- REPL tools stay in `tools/namespace.py`. That is a layer: agent code raises, the engine interprets.
- Prompt phrasing stays in `prompts/` and `graph/prompts.py`. A string constant next to the node type that renders it is fine. A predicate that chooses the next node is not a string.

Read-path budget for a default row: **the producer file, plus at most one layer**. Allowed second files: `graph/nodes.py` (the type you return), `runtime/repl.py` (the status you interpret), `prompts/messages.py` (how you phrase it). Not allowed: a sibling engine file whose only job is a one-line `when=`.

## Target

```text
rlmflow/steps/transitions.py    Transitions, Row, Offer, errors
                                (no default policy, no SOURCES, no has_request)

rlmflow/steps/defaults.py         every default @on row
                                the guards those rows use
                                requested_transition, node_for_run
                                SOURCES / PLAN_SOURCES

rlmflow/graph/nodes.py          types + payload
                                working_instruction next to PlanQuery
                                one can_spawn, used by PlanQuery and prompts

rlmflow/graph/prompts.py        strings the node types render
                                (keep; this split is a layer)

rlmflow/graph/config.py         AgentConfig
                                (keep; this split is a type)

rlmflow/tools/namespace.py      finish / transition / wait / observe factories

rlmflow/tools/launch.py    launch_tool + resolve / new / submit
```

Delete `graph/utils.py`. Identity, in-flight, and `agent_payload` go into `nodes.py` (they are already re-exported from there). `_isoformat` stays with persistence. `_rebuild_index` stays with the replay that calls it, or becomes a method on `AgentStart`.

`chose(name)` either moves with the guards and gets a host-app example, or it is deleted. An unused factory that looks like the way named transitions work is part of the mess.

## What a row looks like after

```python
def has_request(node: Node) -> bool:
    return requested_transition(node) is not None


@Flow.transitions.on(ExecOutput, when=has_request)
def apply_requested(flow: Flow, node: Node) -> Node:
    selected = requested_transition(node)
    ...
```

Same functions. Same names. They sit in `steps.py`. Jump-to-definition on `has_request` lands in the file you are already in. `requested_transition` is either a function in that file or a method on `Node`:

```python
# on Node, next to prev / next
def requested_transition(self) -> str | None:
    previous = self.prev
    return previous.requested_transition if isinstance(previous, ExecAction) else None
```

A method is the better end state: the peek is graph navigation, and `graph/utils.py` was the wrong home only because it was a drawer. Do not keep both a function and a method.

`apply_requested` still calls `flow.transitions.resolve_offer`. That is the table type doing table work. That hop stays.

## Order

1. **Move default policy into `steps.py`.** `SOURCES`, `PLAN_SOURCES`, `at_final`, `budget_nearly_spent`, `out_of_room`, `child_returned`, `needs_truncation`, `has_request`, `chose`, `requested_transition`. Re-export from `rlmflow.engine.steps` for one release if anything outside `steps.py` imported them (`benchmarks/eval/delegation/conditions.py` imports `PLAN_SOURCES` and `child_returned` today). Then point those callers at `engine.steps`.
1. **Park `working_instruction` on `PlanQuery`.** It is the body of `PlanQuery.instruction`. Inline it, or keep the function in `nodes.py` under the class.
1. **One `can_spawn`.** Keep `prompts.prompts.can_spawn` (the public one). Delete the two private copies. Decide the `None` case once: unbound prompt rendering may show launch tools (`True`); a live `PlanQuery` with no flow cannot spawn (`False`). Those are two callers, not two functions — pass the predicate the caller means, or make `can_spawn` require an agent and let prompt code skip the section when `flow is None`.
1. **Delete `graph/utils.py`.** Each helper next to its caller, as in the table above.
1. **Put the full default table in `step_table.md`.** Every row, including `apply_requested` and `run_repl`. One ASCII map at the top of `steps.py` that names the layer hops that remain (REPL signal, offer table, prompt footer).
1. Stop. Do not merge `transitions.py` into `steps.py`. Do not merge `delegation.py` into `namespace.py`. Do not move `run_streaming`.

## What we do not do

- **Do not dump the table type into `steps.py`.** `Transitions` is reused by blank graphs, subclasses, and tests. The bug is default policy living beside it, not the type existing.
- **Do not put guards on `Flow`.** `@Flow.transitions.on(..., when=Flow.has_request)` is a second object with extra steps.
- **Do not add `TransitionService` / `StepPolicy` / mixins.** `flow_extract.md` already forbade this. A behavior is a function.
- **Do not merge REPL tools into the engine.** `transition()` raising `TransitionSignal` is the runtime contract. `node_for_run` interpreting that status is the engine contract. Two files is the right number for that layer pair.
- **Do not merge `graph/prompts.py` into `rlmflow.prompts`.** (Later reversed: all prompt prose now lives in `prompts/text.py`.) Node-owned strings (`FINAL_ANSWER_ACTION`, `TRUNCATION_SUMMARY`) travel with the node type. The inherited protocol lives in `rlmflow.prompts`. That split is already the rule in `graph/prompts.py`'s module docstring.
- **Do not keep `graph/utils.py` as a facade over the new homes.** If the file has no remaining bodies, delete it.
- **Do not "document the hops" as the fix.** A map in `internals.md` is a courtesy after the move, not a substitute for colocating the guards.

## Acceptance

- Opening `steps/defaults.py` shows every default row, the `when=` next to it, and the predicate body.
- `steps/transitions.py` is the empty table type plus `offer` / `resolve` / `available`. No `has_request`, no `out_of_room`, no `SOURCES`.
- `graph/utils.py` does not exist. `requested_transition` is a `Node` method or a function in `steps.py`.
- There is one `can_spawn`. The two private copies are gone.
- `docs/step_table.md` lists `apply_requested` and `run_repl`.
- Jump-to-definition from `@Flow.transitions.on(ExecOutput, when=has_request)` does not leave `steps.py`.
- Public names (`Flow.step`, `@Flow.transitions.on`, `.offer`, `transition("...")` in the REPL) do not change.
