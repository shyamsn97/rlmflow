# Transition policy

Default policy is declared beside its producers in `rlmflow.engine.steps` with the internal module-level `@transition(...)` decorator. Each decorated callable is one `TransitionFunction` carrying its source, guard, stable identity, and producer. `rlmflow.engine.steps.resolve(functions, node, judge)` resolves deterministic guards in order and semantic guards as one choice set.

Applications add an earlier, instance-local rule with `@flow.transition`:

```python
flow = Flow(client)


@flow.transition(source=ErrorOutput, when=lambda node: node.error == "retry")
async def retry(node):
    return UserQuery(content="Retry with the recorded error.")
```

Application handlers and callable guards receive only the source node and may be synchronous or asynchronous. `source="all"` matches every node. At the first eligible string `when=`, every source-compatible string guard before the next unconditional fallback becomes one multiple-choice structured request (plus `continue`) through the Flow's structured engine. The model returns only the selected transition function name; the engine maps that compact label to the durable rule ID. One `TransitionDecision` records the complete candidate set and selected rule. A selected semantic rule wins at that point; `continue` resumes ordered deterministic resolution. Because this uses `Flow.structured_query`, Chat, Instructor, Jev, retry, and fallback structured engines all use the same path.

There is no registry: no `Transitions`, `Flow.transitions`, constructor table injection, subclass mutation hook, or model-selected `transition("name")` route. Every edge is an automatic transition function.

`await flow.step(node)` resolves one rule, appends its returned node, and returns that node. Producers return unattached nodes; the driver alone appends them.

The full default table is the `rlmflow.engine.steps` module docstring. After an `LLMOutput`, `to_action` runs a single valid `repl` block, `to_text_reply` treats a non-empty reply with no `repl` fence as the final answer, and `to_missing_action` handles every other reply by writing an `ExecAction` that raises `MissingReplError` with the specific reason. Protocol errors therefore never leave the normal loop: the raising action executes, replays, and is recovered from like any other failing code.
