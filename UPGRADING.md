# Upgrading to 0.5

Version 0.5 is a deliberate breaking beta release. It does not include compatibility aliases for the pre-0.5 graph, completion, persistence, or TUI APIs.

## Agent completion

Agent code must call `finish(value)`. The old `done(value)` name is not present in the REPL.

Without `output_schema`, `finish(value)` returns `str(value)`. With an explicit
`output_schema`, `Flow.run()`, `agent.result()`, and `wait_for_result()` return
the validated typed value.

## Graph traversal and identity

- Use `node.walk()` for iterative parent-before-child traversal.
- Use `node.iter_backwards()` for same-agent history.
- Use `node.to_record()` for one shallow record.
- Use `root.usage` and `root.stats` for indexed whole-run aggregates.
- Use `node.subtree_usage()` when an explicit subtree scan is intended.
- `node.fork()` always creates fresh graph and agent identities.

The removed forms are `walk(reverse=True)`, `to_dict()`, `tokens()`, and `fork(new_ids=False)`.

## Persistence

Persistence v3 stores one flat, parent-linked graph document. Use `persistence.to_document()` and `persistence.from_document()`. Loading rejects older versions and unknown node types instead of guessing. Register custom Node types with `persistence.register_node_type`.

There is no in-process v2 migration path. Export or regenerate old development runs with the version that created them before upgrading.

## Checkpointing and TUI

`GraphCheckpointer` uses `interval_s` and `interval_nodes`; call `flush()` when an immediate checkpoint is required.

Initialize `FlowTUI` with `ui.init(flow)` and then call `ui.run()`. The old `ui.run(drive)` callback entry point is removed.

## Agent and model helpers

- Call `AgentInfo.result()`; `get_result()` is removed.
- Import `client_for` from `rlmflow.llm`; `examples.common.build_client` is removed.

## Planning control query

`PlanQuery` now combines iterative input investigation and decomposition guidance, then enters the ordinary REPL loop. `InspectQuery`, `INSPECTION_ACTION`, and the automatic `InspectQuery -> PlanQuery` transition are removed. Custom step registrations and persisted development runs must use `PlanQuery`.

## Runtime trust boundary

Worker-to-host tool arguments, `ENV`, and `Runtime.get_var()` results must be JSON-compatible data. Arbitrary Python objects may still be copied from the trusted host into a worker, but executable worker-controlled payloads are never deserialized on the host.

Publish the plain fields the host needs through `ENV` instead of retrieving a custom worker object.

## Execution approval

Pass `execution_guard=` to `Flow`. The guard receives each `ExecAction` before runtime execution and returns `None` to allow it or an error message to reject it.
