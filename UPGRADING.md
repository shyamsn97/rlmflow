# Upgrading to 0.6

Version 0.6 is a deliberate breaking beta release. It does not include compatibility aliases for the pre-0.6 graph, completion, persistence, or TUI APIs. Persistence format v5 is strict; older run directories require an explicit migration or must remain on the runtime that created them.

## Agent completion

Agent code must call `finish(answer)`. The old `done(value)` name is not present in the REPL.

Without `output_schema`, `finish(answer)` accepts only strict JSON values: strings, finite numbers, booleans, `None`, lists, and dictionaries with string keys, recursively. Unsupported values such as tuples, sets, bytes, non-finite floats, arbitrary objects, and dictionaries with non-string keys now produce an execution error instead of being stringified. A string remains a string even when its contents happen to be JSON.

With an explicit `output_schema`, a JSON value or JSON-encoded string is parsed and validated, and `Flow.run()`, `agent.result()`, and `wait_for_result()` return the validated typed value.

## Structured output guidance

`Flow(enable_structured_output=True)` is now the default. This flag controls whether spawn-capable agents see guidance for optional child `output_schema` values; it does not require every child or root to use a schema. Pass `enable_structured_output=False` to hide that guidance.

When a schema is supplied, pass a JSON Schema object with explicit `properties`, `required`, and `additionalProperties`. Without a schema, child and root results retain JSON-compatible Python values as described above.

`AgentConfig.child_output_schema` is the default `output_schema` for launched children. The coding CLI (`rlmflow run` / `rlmflow tui`) and `examples/coding/agent.py` set it to `CODING_HANDOFF_SCHEMA`, so those children `finish` `{done, diff_summary, concerns, followups}` while the root stays untyped. Launch `output_schema=` still overrides. Scored evals and other `Flow`s do not set this field. A custom `--agent` factory that wants untyped children leaves it `None`.

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

`resume_on_child_done` defaults to `True`. A parent that has already `finish()`ed still receives a child-handoff `UserQuery` and keeps driving. Pass `False` on scored evals that must stay terminal once the parent submits. Background children still deliver a `UserQuery` after the launcher's next `ExecOutput` either way. A mapping `finish()` value is listed as fields in that query; a scalar stays labeled JSON.

A `UserQuery` appended after `ExecOutput` now enters `PlanQuery` instead of calling the model immediately. Host injections after idle REPL output get the same extra plan turn as a child-handoff.

## Dependent child inputs

`launch_subagent(after=...)` is removed. Pass each prerequisite handle under the input name the downstream child should receive:

```python
# Before
solution = await launch_subagent(
    "Solve using INPUTS.",
    model="default",
    after={"evidence": evidence},
)

# Now
solution = await launch_subagent(
    "Solve using INPUTS.",
    model="default",
    inputs={"evidence": evidence},
)
```

String and handle values may be mixed in the same `inputs` mapping. Handles are accepted only as top-level values. The downstream child waits for every handle-valued input, then receives each terminal result as a string under that key.

Dependency dataflow no longer changes ownership. All children stay beneath the agent action that launched them, and `max_depth` limits recursive spawning rather than dependency-chain length. `AgentDependency` is no longer exported.

## Persistence

Persistence v5 stores one flat, parent-linked graph document. Handle-valued inputs retain their stable agent-reference markers, and every `ExecAction` retains launch receipts. The separate `dependencies` payload is removed. Use `persistence.to_document()` and `persistence.from_document()`. Loading rejects v4 and older versions rather than migrating them. Register custom Node types with `persistence.register_node_type`.

Cold restore reconstructs complete worker sessions, not just individually unfinished agents. If an unfinished `reuse_repl=True` child depends on a terminal parent's worker, eager and lazy replay rebuild that terminal owner first. Terminal shared tenants in a required session are also replayed because their heap mutations may still be visible to unfinished agents; unrelated terminal sessions remain skipped.

There is no in-process v2 migration path. Export or regenerate old development runs with the version that created them before upgrading.

## Checkpointing and TUI

`GraphCheckpointer` uses `interval_s` and `interval_nodes`; call `flush()` when an immediate checkpoint is required.

Initialize `FlowTUI` with `ui.init(flow)` and then call `ui.run()`. The old `ui.run(drive)` callback entry point is removed.

## Agent and model helpers

- Call `AgentInfo.result()`; `get_result()` is removed.
- Import `client_for` from `rlmflow.llm`; `examples.common.build_client` is removed.

## Step table

`await flow.step(node)` returns the created `Node`. Register automatic producers with `@Flow.transitions.on(...)`. Declare model-selectable routes with `.choices(current, *targets)`; `transition("name")` then lands the selected `UserQuery` directly from the current `ExecAction`. A blank table is `Flow(llm, transitions=Transitions())`. Extra automatic behavior is `@Review.transitions.on(...)`.

Removed: `StepFunction`, `LLMRequestStep`, `LLMOutputStep`, `ExecActionStep`, `update_step_fn`, `DEFAULT_STEPS`, `DEFAULT_TRANSITIONS`, `Flow.always`, `Flow.when`, `MessageBuilder`, and `FlowMessages`. The queue still uses an internal `Transition`; do not take `.created` off `flow.step`.

## Planning control query

`PlanQuery` now precedes every ordinary model action. Its user turn is `Turn N/M:` (plus a first-turn INPUTS safeguard). `InspectQuery`, `INSPECTION_ACTION`, and the automatic `InspectQuery -> PlanQuery` transition are removed. Custom step registrations must use `PlanQuery`; persisted development runs containing `inspect_query` nodes must be regenerated before loading.

## System prompts

`SystemPromptBuilder`, `.sections`, `DEFAULT_BUILDER`, `Section`, and `Sections` are removed. Subclass `PromptBuilder` and concatenate after `super().__call__(flow, node)`, or pass a string / `(flow, node) -> str` to `Flow(system_prompt=...)`.

`Flow.build_system_prompt(node)` is the inherited protocol. `UserQuery.build_system_prompt(flow)` is what a turn sends; `PlanQuery` wraps with the orchestrator addendum when the agent can spawn, and `FinalQuery` wraps with last-turn submit guidance. `Node.build_system_prompt` is gone — actions and outputs have no prompt hook. Inspect a prompt with `flow.build_system_prompt(root)` or `flow.build_messages(node)`, not `flow.system_prompt.render(...)`.

## Tools and toolsets

Use `@tool` for one callable and `@toolset("Title")` for a class that owns several related tools and their prompt documentation. Pass either form as an item in `Flow(tools=[...])`; toolset methods are flattened into the REPL namespace, so agent code calls the method name directly rather than through the class.

Each active toolset renders its own `### Title` subsection. Its `@tool` methods provide the callable signatures and one-line descriptions; an optional `guidance(flow, node)` method adds context-sensitive ordering, shared constraints, or a concise example. Standalone `@tool` callables share a `### Tools` subsection. Empty toolset sections and their guidance are omitted. Reserved builtins use `inject=False` signatures because `Flow` installs their real per-node closures separately.

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

`@tool(..., when=at_leaf)` and toolset `visible_tools(flow, node)` now omit the tool from the REPL as well as the prompt. `FILE_TOOLS` writes bind only on agents that cannot spawn. `launch_subagent` is unbound on those agents too. A `max_depth=1` root can launch; its children can write. Intermediate depths that can still spawn keep reads and lose writes. `rlmflow run` / `tui` now default to `max_depth=1` for that reason.
