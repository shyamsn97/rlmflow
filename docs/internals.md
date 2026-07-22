# rlmflow Internals

A reference for the engine's mechanics: the `Flow`/`Graph`/`Run` split, the two
phases of a run, the per-agent scheduler, the exec turn, delegation, REPL
lifecycle, prompts, tools, and the overridable seams on `Flow`. If you want to
subclass the engine, debug a weird run, or build something on top of rlmflow,
this is the doc.

User-facing topic guides cover the *what*: [`control.md`](control.md),
[`streaming.md`](streaming.md), [`node_model.md`](node_model.md),
[`observability.md`](observability.md), [`runtimes.md`](runtimes.md),
[`prompt_customization.md`](prompt_customization.md), and
[`security.md`](security.md). This doc covers the *how*.

---

## Architecture at a glance

Three objects carry the whole engine:

```text
Flow      policy + environment: config, LLM clients, pool, prompt builder,
          tools, live REPLs, and the run registry. Graph-agnostic and shared.
Graph     the durable state: one recursive node per agent. Caller-owned. The
          run *is* the graph — Flow only mutates it and emits events.
Run       per-graph run state: the Graph, its TaskQueue scheduler, and its
          `until` boundary. One active run per `graph_id`.
```

Supporting pieces:

- **`TaskQueue`** (`rlmflow/tasks.py`) — a small, graph-free async scheduler. One
  in-flight task per agent id, a follow-up slot, and an event stream.
- **`Runtime` / `ReplBackend`** (`rlmflow/runtime/`) — the code sandbox. `Flow`
  opens one REPL per agent and drives `start(code)` / `resume(value)` on it. See
  [`runtimes.md`](runtimes.md).
- **`SystemPromptBuilder`** (`rlmflow/prompts/`) — named sections that render the
  system prompt from the current `flow` and `graph`; `UserPromptBuilder` projects
  the trajectory into user/assistant turns. See
  [`prompt_customization.md`](prompt_customization.md).

The principle: **`Graph` holds state; `Flow` holds policy.** Anything that is a
pure function of a graph is a graph method or a free helper in `rlmflow/utils`;
anything that needs engine config, clients, or live REPLs is a method on `Flow`.

---

## The data model

A `Graph` is a recursive structure: it *is* one agent, and `graph[other_aid]`
returns the `Graph` rooted at any descendant. Per-agent invariants are flat
fields; sub-agents live in `graph.children`; the trajectory lives in
`graph.nodes`.

```python
graph.agent_id          # str  — "root", "root.search", ...
graph.graph_id          # str  — trajectory id (a fork mints a new one)
graph.depth             # int
graph.query             # str
graph.inputs            # dict[str, str] — this agent's INPUTS
graph.model             # str  — model label for this agent
graph.prompt_profile    # str  — PromptProfile selector (not prompt text)
graph.output_schema     # required done() schema, if any
graph.parent_agent_id   # str | None

graph.nodes             # list[Node] — this agent's trajectory (seq order)
graph.children          # dict[str, Graph] — direct sub-agents
graph.agents            # dict[agent_id, Graph] — every agent in the subtree
graph.walk()            # iterator over this agent, then all descendants
graph.finished          # root node terminal and all descendants finished
```

Trajectories strictly alternate **observations** and **actions**; every action
is followed by exactly one observation. There is no `LLMAction` node — the model
turn is the `LLMOutput` observation itself, and `ExecAction` records the code the
engine then ran. See [`node_model.md`](node_model.md) for the full taxonomy and
[`observability.md`](observability.md) for the query surface.

---

## A run in two phases

`run`, `arun`, and `run_streaming` are keyword-only and share one shape. Each
call is resolved (phase 1) and then driven (phase 2).

```python
async def run_streaming(self, *, graph=None, query=None, inputs=None,
                        output_schema=None, merge_inputs=True,
                        until="done", n=None, close_repls=False):
    run = self.resolve_run(graph=graph, query=query, inputs=inputs,
                           output_schema=output_schema, merge_inputs=merge_inputs)
    async for event in self._drive(run, until=until, n=n, close_repls=close_repls):
        yield event
```

### Phase 1 — `resolve_run`

Turns the caller's request into a live, **registered** `Run`, emitting the
structural events observers need. Exactly one of two shapes:

- **`query` alone** builds a fresh `Graph`, registers it, and announces it with a
  `GraphCreated` event.
- **`graph`** is resumed. `inputs`/`output_schema` are applied to the graph *in
  place* first (so they take effect even on a finished graph); `inputs` merges
  into the existing dict unless `merge_inputs=False` replaces it, and any warm
  REPL's `INPUTS` is re-synced. If `query` is also given, it is appended as a
  fresh `UserQuery` turn (an `AppendNode` event), which flips `finished` back to
  false — this is the multi-turn / long-running path.

Registration happens *before* emission because `apply_action` only streams
events for graphs that already have a run.

### `run_for` — the run registry

`run_for(graph)` is a pure registry keyed by `graph.graph_id`: resume the
existing (live or paused) run, or register a fresh one. It never mutates the
graph or emits — all resolution lives in `resolve_run` ahead of it, so a new turn
on a finished graph is resolved correctly instead of being skipped by an early
registry hit.

### Phase 2 — `_drive`

Assumes a resolved, registered run. It schedules ready agents and yields events
until the `until` boundary:

```python
while True:
    if not run.tasks.has_events():
        for agent in graph.walk():
            self.schedule_agent(run, agent)          # one task per ready agent
    async for item in run.tasks.items():
        yield item.item                              # stream the event
        agent = graph.agent_for(item.agent_id)
        if self.should_continue(run, agent, item.item):
            self.schedule_agent(run, agent)          # enqueue next unit
        else:
            run.tasks.stop(agent.agent_id)           # "stop" = no follow-up
    if graph.finished or until in {"done", "finished"}:
        break
    ...                                              # non-full boundary: pause
```

Stopping is just the *absence* of a next task for that agent — there is no pause
primitive. `should_continue` encodes the `until` policy (`next`, `idle`, `error`,
`supervising`, a callable, or full-run). `SupervisingOutput` and `ResumeAction`
events never continue their own agent. A finished (or full-run) drive tears down
the run via `finish_run`; a paused bounded run keeps its queue so a later call
resumes from the graph. See [`streaming.md`](streaming.md) for the boundary
semantics and parallelism model.

---

## `run_agent_task` — one graph-producing unit

Each scheduled task advances one agent by exactly one node-producing step:

```python
async def run_agent_task(self, graph):
    if graph.finished: return
    if graph.agent_id in self._terminate_requested:
        self.append_node(graph, DoneOutput(result=TERMINATED)); return
    if budget_exceeded(root, self.max_budget):
        self.append_node(graph, DoneOutput(result=BUDGET_EXCEEDED)); return

    current = graph.current()
    if isinstance(current, LLMOutput):        # model spoke -> run its code
        self.append_node(graph, ExecAction(code=current.code)); return
    if isinstance(current, ExecAction):        # code queued -> execute it
        await self.exec_turn(graph, current.code); return

    if llm_turns >= max_iters:                 # cap reached
        self.append_node(graph, DoneOutput(result=MAX_ITERS_EXCEEDED)); return
    self._ensure_nudge(graph, force_final=...)         # nudge as a real node
    messages = self.messages(graph)                    # then call the model
    reply, usage = await self._call_chat(messages, graph.model)
    self.append_node(graph, LLMOutput(content=reply, code=code_block(reply), ...))
```

`max_iters` counts `LLMOutput` nodes; children fall back to `child_max_iters` via
`iter_budget`. Budget/terminate are checked at the top of every unit.
`run_agent(graph)` is the self-driving loop (`while not finished: run_agent_task`)
used by delegated children.

## `exec_turn` — the observation half

`exec_turn(graph, code)` runs code against the agent's REPL and appends exactly
one observation node:

- `repl.done_result` set → `DoneOutput` (terminal; never truncated).
- `repl.errored` or a dead session → `ErrorOutput`.
- otherwise → `ExecOutput` (stdout, capped at `max_output_length`).

`done(...)` is a REPL tool that stores `repl.done_result` (validating against
`output_schema` if present) and raises `DoneSignal`. `replay=True` returns the
node without appending — used to rebuild a live REPL from a graph that already
holds the result nodes.

---

## Delegation

Delegation is a single async REPL tool, `launch_subagents`, built per agent by
`make_launch_subagents` (`rlmflow/tools/builtins.py`). When agent code runs
`results = await launch_subagents([...])` inside an exec turn, the tool:

1. for each **cold** spec, creates a child `Graph` from `query` (sharing the
   parent's `graph_id`, depth+1); for each **warm** spec that already carries a
   prepared `graph`, calls `flow.adopt(parent, child, name=...)` to reparent it
   — each child is announced with an `AddChild` event. Specs past `max_depth` or
   over `max_query_chars` become placeholder result strings;
2. appends a `SupervisingOutput(waiting_on=child_ids)` to the parent;
3. submits each child to the **run's** `TaskQueue` (children self-drive with
   `run_agent`), then awaits `tasks.changed()` until all children are finished;
4. appends a `ResumeAction(resumed_from=child_ids)`, collects each child's
   `result()` (validated against its `output_schema` if any), and returns the
   list into the suspended `await`.

So the whole delegate → wait → resume cycle happens *within one parent exec turn*.
Submitting children to the same queue means the driving stream observes their
events, and the queue's one-task-per-agent rule prevents double-driving. If no
run exists (a bare `run_agent` with no stream), a throwaway `TaskQueue` runs the
children. See the delegation/resume flow in [`node_model.md`](node_model.md).

Payloads go in each spec's `inputs` (the child's `INPUTS`); `query` stays a short
instruction. `prompt_profile` (legacy key `prompt` still accepted) stamps the
child's `Graph.prompt_profile`. `output_schema` per spec makes the child's
`done(value)` validated.

**Host warm path.** `flow.adopt(parent, child, *, name)` is the structural
primitive (remap ids, move the REPL, emit `AddChild` — does not run). 
`flow.launch_subgraphs(parent, children, *, queries=None, names=None)` is the
graph-first wrapper that builds warm specs and calls `launch_subagents`, so
host/example code never has to write the internal `graph` key. Use it for
fork/rewind → fan-out → compare patterns (see `examples/shepherd/`).

---

## Graph mutation and events

Every state change goes through one seam so observers and checkpointers see it:

```python
def append_node(self, graph, node):          # node or str
    return self.apply_action(graph, graph.inject_action(node))

def apply_action(self, graph, action):        # any GraphAction
    apply_graph_action(base, action)          # mutate the graph value
    run = self.runs.get(graph.graph_id)
    if run is not None:                        # emit only for registered runs
        run.tasks.emit(agent_id, replace(action, graph_id=graph.graph_id))
    return action
```

`GraphAction` (in `rlmflow/graph/events.py`) is the closed set of transitions:
`GraphCreated`, `AppendNode`, `InsertNode`, `ReplaceNode`, `RemoveNode`,
`AddChild`, `RemoveChild`. Each subclasses `Event` and exposes a uniform
`node` / `node_id` / `node_type` view plus a stamped `graph_id`, so a merged
stream (`parallel_stream`) is self-describing. Graph-only edits (`graph.inject`,
`append`, `replace`, `rewind`, …) mutate the value directly; routing them through
`Flow.append_node` / `Flow.apply_action` also emits them into a live run. See
[`injections.md`](injections.md).

---

## REPL lifecycle

`Flow` keeps live REPLs in `self.repls`, keyed by `repl_key(graph)` =
`(graph_id, agent_id)`, so each agent's namespace, tools, and variables are
isolated and survive across turns.

- `repl_for(graph)` — return the agent's REPL, lazily opening one via
  `runtime.open(graph)` and seeding it with `build_tools(...)`, the agent's
  `INPUTS`, and process env (`AGENT_ID`, `DEPTH`, `MAX_DEPTH`, …).
- `close_repl(graph)` / `close_repls(graph_id=None)` — tear down one agent's or a
  whole trajectory's REPLs. `finish_run(close_repls=True)` closes on completion;
  the default keeps them warm for pause/resume, fork, and inspection.
- `rebuild_repl(graph)` — replay the agent's `ExecAction` code blocks (no LLM
  calls, no appended nodes) to reconstruct REPL state. The correctness floor for
  forks and reverts.
- `fork` / `merge` / `discard` — branch a graph (deep copy, fresh `graph_id`) with
  its REPL rebuilt; fold a child branch's delta (`ExecAction`s + one summary node)
  and its variables back into a parent REPL; or eagerly free rejected branches.
- `adopt(parent, child, *, name)` — reparent a prepared single-agent graph under
  `parent` (warm attach; see Delegation above).
- `get_var(graph, name)` / `get_env_var(graph, name)` — read the REPL Python
  namespace or the `ENV` metadata channel back onto the host.
- `_sync_repl_inputs(graph)` — re-inject `INPUTS` into a warm REPL after
  `resolve_run` updates a resumed graph's inputs.

---

## Prompts, LLM calls, and tools

**Prompt.** `messages(graph)` assembles the chat list: `build_system_prompt(graph)`
resolves `self.system_prompt` (a `SystemPromptBuilder`, string, or
`(flow, graph) -> str` function via `as_system_prompt_fn`), and the user side
comes from `self.user_prompt` (a `UserPromptBuilder` or `(flow, graph) -> turns`
function, both already callable) which projects the trajectory into
`user`/`assistant` turns. `messages` then prepends the system message, truncates
to `max_messages`, and coalesces adjacent same-role turns into one. The trailing
nudge ("Continue." / "Give your final answer.") is not synthesized here — it is
materialized as a real `UserQuery` node by `_ensure_nudge` before the call, so it
persists in the trajectory. Customize with `SystemPromptBuilder` sections, a
`UserPromptBuilder` subclass, or a `Flow` subclass — see
[`prompt_customization.md`](prompt_customization.md).

**LLM calls.** `_call_chat(messages, model)` is the single leaf call. Blocking
clients run on `self.pool` (a `ThreadPool(workers=N)` that caps concurrent
blocking calls); async clients stay on the loop and bypass the pool.
`llm_request_timeout` bounds both. `llm_client(model)` resolves a named client
from `llm_clients` (default is `llm`). `llm_query_batched` runs independent
one-shot prompts concurrently through the same path (opt in with
`use_llm_query=True`).

**Tools.** `build_tools(graph, repl)` seeds each REPL namespace with the framework
tools `done`, `launch_subagents`, and `INPUTS`, plus (optionally)
`llm_query_batched`, then the runtime's registered tools and the flow's tools.
`add_tool` / `remove_tool` mutate every live REPL so tools can appear mid-run
(the prompt's tool section reflects them next turn). `done`, `launch_subagents`,
and `INPUTS` are reserved and cannot be overridden.

---

## Persistence

`graph.save(path)` writes a self-contained run directory; `Graph.load(path)`
rehydrates the same recursive shape. The manifest is `graph.json`; per-agent logs
live under `agents/<agent-id>/` (`agent.json`, `session.jsonl`, `latest.json`).
Cross-agent edges derive from the recursive structure and `SupervisingOutput`
wait sets — there is no separate edge object to maintain. Save inside the stream
loop for live checkpointing. See [`observability.md`](observability.md).

---

## The overridable surface

Every method on `Flow` is an extension seam; the engine dispatches through
`self`, so subclass overrides take effect. Common ones:

| Stage | Methods |
|---|---|
| Drive lifecycle | `run`, `arun`, `run_streaming`, `resolve_run`, `run_for`, `_drive`, `finish_run`, `terminate` |
| Scheduling | `schedule_agent`, `should_continue`, `run_agent_task`, `run_agent` |
| Exec / REPL | `exec_turn`, `repl_for`, `close_repl(s)`, `rebuild_repl`, `fork`, `merge`, `discard`, `adopt`, `get_var`, `get_env_var` |
| Delegation | `launch_subagents`, `launch_subgraphs` |
| Graph mutation | `append_node`, `apply_action` |
| Prompt / messages | `messages`, `build_system_prompt`, `build_tools`, `prompt_profile`, `prompt_name_for` |
| LLM half | `_call_chat`, `llm_client`, `llm_query_batched` |
| Tools | `add_tool`, `remove_tool` |

```python
class ReviewingFlow(rlmflow.Flow):
    async def exec_turn(self, graph, code, *, replay=False):
        if not replay and not approved(code):
            node = rlmflow.ErrorOutput(content="rejected", output="rejected", error="rejected")
            self.append_node(graph, node)
            return node
        return await super().exec_turn(graph, code, replay=replay)


class RoutedFlow(rlmflow.Flow):
    def llm_client(self, model="default"):
        if model == "default" and self.depth_hint >= 2:
            return super().llm_client("fast")
        return super().llm_client(model)
```

For prompt-only customization (the common case), don't subclass `Flow` — edit a
`SystemPromptBuilder`'s sections. See
[`prompt_customization.md`](prompt_customization.md).

---

## Where to read next

- [`node_model.md`](node_model.md) — node taxonomy, action/observation
  alternation, delegation wait/resume flow.
- [`streaming.md`](streaming.md) — the scheduler, `until` boundaries, `n`, and
  parallelism.
- [`control.md`](control.md) — user-facing loop, save/resume, rewind, fork,
  multi-turn runs, custom tools/prompts.
- [`observability.md`](observability.md) — querying the `Graph`, run layout,
  viewer, exports.
- [`runtimes.md`](runtimes.md) — `Runtime` protocol and shipped backends.
- [`security.md`](security.md) — trust boundary, isolation knobs, approval gates.
