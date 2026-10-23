# Upgrading to 0.7

Version 0.7 is a clean break. It removes the old transition registry, async launch admission, model-selected child policy, and model-facing `structured_query` surface. The old `finish(value)` completion name remains executable only as a hidden compatibility alias for persisted REPL code; new agent code and prompt documentation use `submit(value)`.

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

For proxied tools whose external work must not repeat during namespace reconstruction, use `@tool(..., restore="result")`. Successful JSON results are saved on the owning action. Pass a restore callable only when that JSON value must be rebuilt into a runtime object.

`FILE_TOOLS` is now a tuple of functions. Replace `FILE_TOOLS.read_file` with `from rlmflow.tools.files import read_file`.

`Flow.build_tools(node)` is the complete model-visible tool map. Flow-owned built-ins, configured application tools, and subclass-provided environment tools are all ordinary decorated callables in that map. The prompt renders it directly; invocation arguments are bound as direct REPL variables alongside `asyncio` and internal execution bindings. Use `flow.llm_query` from host code.

Normal tools are serialized into local workers. Use `@tool(..., proxy=True)` when a tool must execute against host-owned mutable state or another host-only resource.

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
