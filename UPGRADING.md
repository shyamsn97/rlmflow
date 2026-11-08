# Upgrading to nodes, edges and a REPL

The core is rebuilt on one record, `Node`, plus edges; an agent is its root
node. See [The graph](docs/node_model.md) for the reference and
[Internals](docs/internals.md) for how a run works. Saved runs in
format 5 are not migrated: loading one raises, so rerun it.

## Agents are root nodes

| Before | Now |
| --- | --- |
| `Agent` record | The root `Node`; its identity is `root.root_data` (`RootData`) |
| `agent.name`, `agent.task`, `agent.config_name` | `root.root_data.name`, `.task`, `.config_name` |
| `agent.nodes` | `root.root_data.history[1:]` (the history starts with the root) |
| `agent.args`, `agent.resolved_args()` | `root.input`, which is `{"task": ..., **args}` |
| `agent.argument_descriptions()` | `root.root_data.arg_descriptions` |
| `agent.bind_config(config)`, `agent.metadata` | `root.root_data.bind_config(config)`, `root.root_data.metadata` |
| `node.agent_id` | `node.owner` (the root's id); `g.owner(node)` for the root |
| `agent.base` | A `fork` edge from the base node to the fork's root; `g.base(fork)` |
| `g.history(agent)` was `[]` for a fresh root | It starts with the root node |
| `g.nodes("code", agent=a)` | `[n for n in g.history(a) if n.runs_code]`; `g.nodes()` includes roots |

`g.add_agent(task, **args)`, `g.agents()`, `g.roots()`, `g.children(a)`,
`g.parent(a)`, `g.status(a)`, `g.result(a)` and `g.path(a)` keep their names and
take or return root nodes.

## Node types become role, code and flags

| Before | Now |
| --- | --- |
| `node.type == "llm"` | An assistant node with `"model" in node.data`; its `text` is the reply |
| `node.type == "code"` | `node.runs_code`; the code is on the same node as the reply that wrote it |
| `node.type == "message"` | `node.role == "user"`; `node.producer` says who sent it |
| `node.type == "done"` | `node.data.get("terminal")`, with `node.result` |
| `node.type == "idle"` | `node.data.get("idle")` |
| `node.data["code" \| "output" \| "result" \| "text" \| "error"]` | `node.code`, `.output`, `.result`, `.text`, `.error` |
| `node.data["calls"]` | `node.calls` |
| `data["pending"]` | `node.state` is `"ready"` or `"running"` |

`rlmflow.view.text.kind(node)` gives a display name (`agent`, `message`, `done`,
`idle`, `model`, `code`, `reply`) for code that printed `node.type`. The node
that ended a turn has `"seen" in node.data`; `g.last_turn(agent)` finds it.

```python
async for step in flow.run_streaming(g):
    print(g.path(step.agent), kind(step.node))  # was step.node.type

node = next(n for n in g.history(root) if n.runs_code)  # was n.type == "code"
node.calls  # was node.data["calls"]
g.replace(node, 0, "cached")
```

## Edges

| Before | Now |
| --- | --- |
| `g.add_edge(kind, a, b, **data)` | `g.link(kind, a, b, **data)` |
| `g.remove_edge(edge)` | `g.unlink(edge)` |
| `g.link_return(done, node)` | `g.link(RETURN, done, node)` |
| `message` edges | `write` edges, from the sending node to the message node |

## Messages are nodes

`g.inbox`, `agent.inbox` and `g.deliver` are gone. `g.send(agent, text, by=None,
sender=None, **inputs)` appends a user node to the agent's history at once;
its inputs become variables in the agent's REPL, and the agent reads it at its
next turn. Agent code passes values the same way:
`AGENTS["reviewer"].send("See these lines.", lines=[40, 41])`. Unawaited
children's results arrive as one user node binding each result as a variable
named after its child (`worker_0` for `worker-0`).

## Model clients and workers

`PooledLLMClient`, `Pool`, `ThreadPool`, `SequentialPool` and `TaskQueue` are
removed, with `rlmflow.engine`. Cap concurrent model calls with
`Flow(workers=n)`, and make one call with
`await rlmflow.llm.complete(client, messages, timeout=...)` (or
`flow.complete(messages, model)`). `LocalWorkspace` and `GitWorktrees` import
from `rlmflow.workspaces` (or `rlmflow`).

## Running code

| Before | Now |
| --- | --- |
| `current_code()` | `current_step().node` |
| `repl.seed(tools, args)` | `repl.seed(tools)`; arguments are the root's `input`, bound once when the root runs |
| Saved calls matched by position per tool | Keyed by tool and arguments; `CallLog(node)` per run, on `step.calls` |
| Every host tool's result saved | Only `@tool(cache=True)` host tools are saved |
| `g.busy`, `g.revision`, `g.subscribe`, `g.edits` | Gone; edit a running graph with `await flow.edit(g, op, node, ...)` |

## Saved runs

`graph.json` is format 6: `nodes` (each root lists its history by id) and
`edges`. Format 5 is not migrated.

# Upgrading to core v2

Core v2 rebuilds the graph around agents and edges and makes it editable. Saved
runs from earlier versions don't load; rerun them.

## The graph

| Before | Now |
| --- | --- |
| `agent_start` node | The `Agent` record: `agent.nodes`, `agent.inbox`, `agent.task`, `agent.args`. `g.agent(id)`, not `g.node(id)`. |
| `llm_output` | `llm` (`rlmflow.LLM`) |
| `waiting` | `idle`, only for session agents waiting for messages |
| `from` / `next` edges | `spawn`, `return`, `message`; an agent's order is its `nodes` list, and a fork is `agent.base` |
| `g.attach(parent, *agents)` | `g.add_agent(task, parent=agent)` |
| `g.depends_on`, `inputs={...}` | Await the producer and pass its result |
| `reports_to` edges | Message the agent, or add your own edge kind with `g.add_edge` |
| `g.as_of(seq)` | `rlmflow.view.timeline.snapshot(g, count)` |
| Append-only history | `g.edit`, `g.delete`, `g.remove`, `g.replace` |
| `g.store`, `STORE_TOOLS` | A host tool over your own dict (see the orchestrate skill's task board) |

```python
g = Graph()
root = g.add_agent("Audit this repository.")
flow.run(g)

g.history(root)  # [llm, code, ..., done], no agent_start
node = next(n for n in g.history(root) if n.type == "code")
node.data["calls"]  # what each tool call in the node's code returned
g.replace(node, 0, "cached")  # the node re-runs with this value
g.edit(g.history(g.children(root)[0])[-1], result="42", dry_run=True)
```

## Agent code

- `await agent` and `await agent.send(...)` work for agents you launched and their
  descendants. For anyone else, `agent.send(...)` and read the reply at your next
  turn; awaiting raises `NotYourAgent`.
- `launch_agent(task, spec=other_agent)` raises `TypeError`; use
  `launch_agent(task, spec=await other_agent)`.
- `send_agent(target, message)` is gone; use `AGENTS[target].send(message)`.
- `AGENTS.siblings` is gone; use `AGENTS.parent.children`.
- `agent.status` is `"done"`, `"idle"` or `"running"`.

## Removed integrations

`FlowLLM`, `messages_to_query`, `TinkerClient`, `DSPyFlow`, the Textual TUI
(`rlmflow tui`, `FlowTUI`) and `rlmflow.rao` are removed. `rlmflow run print
"..."` (or `rlmflow run --query "..."`) streams the agent tree in the terminal.

# Upgrading to 0.7

Version 0.7 is a clean break. It removes the old transition registry, async launch admission, model-selected child policy, and model-facing `structured_query` surface. The old `finish(value)` completion name remains executable only as a hidden compatibility alias for persisted REPL code; new agent code and prompt documentation use `submit(value)`.

## Graph queries and writes

`rlmflow.graph.reads` and `rlmflow.graph.ops` are gone; call the same functions on the graph, without passing it:

```python
g.result(agent)  # was reads.result(g, agent)
g.history(agent)  # was reads.history(g, agent)
g.send(agent, "text")  # was ops.send(g, agent, "text")
g.fork(node, name="b")  # was ops.fork(g, node, name="b")
```

## Child launch

Replace:

```python
child = await launch_subagent(
    "task",
    model="fast",
    name="worker",
    inputs={"context": context},
    output_schema=schema,
)
result = await child.wait_for_result()
```

with:

```python
child = launch_subagent(task="task", context=context, schema=schema)
result = await child
```

Flow selects child policy with `delegation_model`, `delegation_prompt_profile`, and `delegation_reuse_worker`. Arguments are strict JSON values or nested `AgentHandle` references. Colocated children reuse the process but keep isolated namespaces. An optional `name=` gives the child a stable sibling-unique name; otherwise the Flow assigns `childN`. Use `verify=` for a worker-local `value -> str | None` submission verifier.

## Agent completion

Agent code finishes with `submit(answer)`. The old `done(value)` name is not present in the REPL, and `finish(value)` is retained only as the hidden compatibility alias described above.

A model reply with no fenced `repl` block also finishes the agent: its text is checked exactly like `submit(text)`. If you scored runs on the assumption that prose was a protocol error, expect fewer recovery turns and plain-text results. Code that caught `TransitionProtocolError` should check for an `ErrorOutput` containing `MissingSubmitError` instead, and code that imported `MISSING_REPL_NOTE` should match the `MissingReplError:` prefix of the error observation.

Without `output_schema`, `submit(answer)` accepts only strict JSON values: strings, finite numbers, booleans, `None`, lists, and dictionaries with string keys, recursively. Unsupported values such as tuples, sets, bytes, non-finite floats, arbitrary objects, and dictionaries with non-string keys now produce an execution error instead of being stringified. A string remains a string even when its contents happen to be JSON.

With an explicit `output_schema`, a JSON value or JSON-encoded string is parsed and validated, and `Flow.run()`, `agent.result()`, and `wait_for_result()` return the validated typed value.

## Structured output guidance

When a schema is supplied, pass a JSON Schema object with explicit `properties`, `required`, and `additionalProperties`. Without a schema, child and root results retain JSON-compatible Python values as described above.

`AgentConfig.child_output_schema` is the default schema for launched children. The coding CLI (`rlmflow run` / `rlmflow tui`) and `examples/coding/agent.py` set it to `CODING_HANDOFF_SCHEMA`, so those children `submit` `{done, diff_summary, concerns, followups}` while the root stays untyped. Launch `schema=` overrides it.

## Multiple roots

`parallel_run` and `parallel_stream` are removed. Pass every root to the one multi-root driver directly:

```python
roots = [flow.start("query A"), flow.start("query B")]
async for node in flow.run_streaming(*roots):
    ...
results = [root.result() for root in roots]
```

## Node.append

`node.append(child)` mutates the tree and returns `None`. Capture the child before appending:

```python
created = LLMOutput(content=reply, code=code_block(reply))
node.append(created)
return created
```

Do not write `return node.append(...)` or `next = node.append(...)`. `append_child` and `AppendChild.attach` still return the attached subtree.

## Graph traversal and identity

- Use `node.walk()` for iterative parent-before-child traversal.
- Use `node.iter_backwards()` for same-agent history.
- Use `node.to_record()` for one shallow record.
- Use `root.usage` and `root.stats` for indexed whole-run aggregates.
- Use `node.subtree_usage()` when an explicit subtree scan is intended.
- `node.fork()` always creates fresh graph and agent identities.

The removed forms are `walk(reverse=True)`, `to_dict()`, `tokens()`, and `fork(new_ids=False)`.

## Agent limits and truncation

Library runs are unbounded by default: `AgentConfig.max_iters` and `max_budget` are `None`. The CLI still supplies its own finite iteration default. Set limits explicitly for unattended or user-facing runs.

`child_max_iters` is removed. Every child inherits its parent's `max_iters`; use one per-agent turn-limit contract rather than a second descendant-specific default.

`max_budget` is a soft whole-run threshold, not an exact billing ceiling. Once recorded root usage reaches it, no new child may launch and every unfinished agent gets one `FinalQuery`. Descendants finish first so parents can synthesize their results. Calls already in flight and one final turn per unfinished agent may run beyond the configured threshold.

`max_output_length` and `max_query_chars` now default to 20,000 characters. `max_output_length` bounds one REPL observation; `max_query_chars` bounds a delegated child's goal. Set either field on `AgentConfig`, through `flow.start(...)`, or in `Flow(root_config=...)`.

`resume_on_child_done` defaults to `True`. A parent that has already submitted still receives a child-handoff `UserQuery` and keeps driving. Pass `False` on scored evals that must stay terminal once the parent submits. Background children still deliver a `UserQuery` after the launcher's next `ExecOutput` either way. A mapping submission value is listed as fields in that query; a scalar stays labeled JSON.

A `UserQuery` appended after `ExecOutput` calls the model directly. Host injections and child handoffs are already real user messages; the engine does not insert another planning message between them and the model.

## Dependent child inputs

Pass each prerequisite handle under the input name the downstream child should receive:

```python
solution = launch_subagent(
    task="Solve using evidence.",
    evidence=evidence,
)
```

JSON and handle values may be mixed and nested. The downstream child waits for every referenced handle, then receives each validated result at the same path.

Wrap a root or child argument with `Input(value, description=...)` when its caller already knows useful structure. The agent receives `value` directly under the original argument name; the description is added to the argument manifest and persisted with the graph. `Input` may wrap a handle without changing dependency scheduling.

Dependency dataflow no longer changes ownership. All children stay beneath the agent action that launched them, and `max_depth` limits recursive spawning rather than dependency-chain length. `AgentDependency` is no longer exported.

## Persistence

Persistence stores one unversioned, flat, parent-linked graph document, including each node's durable message snapshot and `AgentStart.args`. Handle-valued arguments retain stable agent-reference markers, and every `ExecAction` retains its restorable tool results in `saved_tool_calls`. Use `persistence.to_document()` and `persistence.from_document()`. Loading validates the document's actual fields and graph structure. A custom Node type is a dataclass subclass with its own `type` name: every data field is saved, and loading finds the class once it has been imported. There is nothing to register and no `to_record` to write.

Cold restore reconstructs required worker sessions, not just individually unfinished agents. `launch_subagent` restores its existing child handle, and `llm_query` restores its saved result instead of repeating external work. If an unfinished `reuse_worker=True` child depends on a terminal parent's worker, eager and lazy replay rebuild that terminal owner first. Agent namespaces remain isolated.

Regenerate development runs saved with older graph fields before upgrading.

## Checkpointing and TUI

`GraphCheckpointer` uses `interval_s` and `interval_nodes`; call `flush()` when an immediate checkpoint is required.

Initialize `FlowTUI` with `ui.init(flow)` and then call `ui.run()`. The old `ui.run(drive)` callback entry point is removed.

## Agent and model helpers

- Call `AgentInfo.result()`; `get_result()` is removed.
- Import `client_for` from `rlmflow.llm`; `examples.common.build_client` is removed.

## Step table

`await flow.step(node)` returns the created `Node`. Add application rules to a Flow instance:

```python
@flow.transition(source=ErrorOutput, when=lambda node: node.error == "retry")
def retry(node):
    return UserQuery(content="Retry with a new approach.")
```

`Transitions`, `Flow.transitions`, `transitions=`, `.on`, and subclass registry mutation are removed. Model-selected routes are removed too: `flow.choices(...)`, the REPL `transition("name")` tool, `InvalidTransitionError`, and the prompt's transition footer no longer exist. Express a mode switch as an ordinary `@flow.transition(...)`, using a string `when=` if the switch needs a judgment.

Removed: `StepFunction`, `LLMRequestStep`, `LLMOutputStep`, `ExecActionStep`, `update_step_fn`, `DEFAULT_STEPS`, `DEFAULT_TRANSITIONS`, `Flow.always`, `Flow.when`, `MessageBuilder`, and `FlowMessages`. The queue still uses an internal `Transition`; do not take `.created` off `flow.step`.

## Planning

The legacy turn-counter `PlanQuery`, `to_plan`, `resume_after_child`, `child_returned`, and synthetic `Turn N/M:` message remain removed. `AgentStart` calls the model directly, and every response plans and executes its next step in one REPL block:

```text
AgentStart -> LLMOutput -> ExecAction -> ExecOutput | DoneOutput
```

`submit` is available from the first block. The pinned core protocol tells the model to plan and execute the immediate next action, to look first when the inputs' size, format, or contents are unknown, and to submit only after reading any printed evidence needed to support the answer.

## System prompts

`SystemPromptBuilder`, `PromptBuilder`, `.sections`, `DEFAULT_BUILDER`, `Section`, and `Sections` are removed. Pass a string or `(flow, node) -> str` to `Flow(system_prompt=...)` for whole-system ownership. For additive changes, pass `system_prompt_modifiers=` or profile modifiers that edit the supplied `Prompt`.

Flow creates fresh defaults with `default_system_prompt(node)` and `default_user_prompt(node)`. Nodes customize them through `build_system_prompt(system_prompt, flow)` and `build_messages(user_prompt, flow)`. The latter returns current-turn messages only; `Flow.build_messages(node)` remains the complete request assembler. `render_fn` and `build_user_prompt` are removed.

Node chat `render()` is also removed. `build_messages(user_prompt, flow)` is the only message customization hook; its result is snapshotted on the node and used for all later history projection. Graph persistence stores those snapshots. Background-agent status remains a live request overlay and is not stored in history.

## Tools

`@tool` is the only tool decorator; `@toolset`, `FileTools`, the `BuiltIns` class, and `@tool(inject=...)` are removed. `description` is a string. Replace a toolset class with module-level `@tool` functions grouped in a tuple; `Flow(tools=[...])` flattens one level of nesting.

Host tools' JSON results are saved on the node that called them, so rebuilding a REPL returns them instead of calling the tool again; `@tool` takes no `restore=` or `receipt=`. Results that aren't JSON, or are over 64KB, aren't saved and run again on replay.

`FILE_TOOLS` is now a tuple of functions. Replace `FILE_TOOLS.read_file` with `from rlmflow.tools.files import read_file`.

`Flow.build_tools(node)` is the complete model-visible tool map. Flow-owned built-ins, configured application tools, and subclass-provided environment tools are all ordinary decorated callables in that map. The prompt renders it directly; invocation arguments are bound as direct REPL variables alongside `asyncio` and internal execution bindings. Use `flow.llm_query` from host code.

Tools execute on the host by default, so they can use host-owned state, clients and connections. Use `@tool(..., where="repl")` for a pure function that should be copied into the worker instead. `proxy=True` is gone; delete it.

## REPL preimports

Every runtime now preimports the modules in `rlmflow.runtime.DEFAULT_PREIMPORTS`: `json`, `re`, `os`, `sys`, `pathlib`, `csv`, `math`, `asyncio`, `collections`, `datetime`, `itertools`, `statistics`, `textwrap`, and `time`.

Pass `preimports=` to `LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`, or another runtime to replace that list. `preimports=[]` binds no modules; names may include installed third-party packages.

## Runtime trust boundary

Worker-to-host tool arguments, `ENV`, and `Runtime.get_var()` results must be JSON-compatible data. Arbitrary Python objects may still be copied from the trusted host into a worker, but executable worker-controlled payloads are never deserialized on the host.

Publish the plain fields the host needs through `ENV` instead of retrieving a custom worker object.

## Execution approval

Pass `execution_guard=` to `Flow`. The guard receives each `ExecAction` before runtime execution and returns `None` to allow it or an error message to reject it.

## Tool visibility

`@tool(..., when=at_leaf)` now omits the tool from the REPL as well as the prompt. `FILE_TOOLS` writes bind only on agents that cannot spawn. `launch_subagent` is unbound on those agents too. A `max_depth=1` root can launch; its children can write. Intermediate depths that can still spawn keep reads and lose writes. `rlmflow run` / `tui` now default to `max_depth=1` for that reason.
