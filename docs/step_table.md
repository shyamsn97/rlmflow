# `transitions.on` is the whole table

`Flow.transitions` starts empty. Functions register onto it afterward. Default policy — producers and the guards they use — lives in `rlmflow.engine.steps`. The table type is `engine.transitions`.

```python
class Flow:
    transitions = Transitions()


@Flow.transitions.on(AgentStart, UserQuery, ErrorOutput, ExecOutput, when=out_of_room)
async def to_final(flow: Flow, node: Node) -> FinalQuery:
    return FinalQuery()


@Flow.transitions.on(AgentStart, UserQuery, ErrorOutput, ExecOutput, when=needs_truncation)
async def to_summary(flow: Flow, node: Node) -> TruncationSummary:
    return TruncationSummary()


@Flow.transitions.on(AgentStart, ErrorOutput, ExecOutput)
@Flow.transitions.on(UserQuery, when=child_returned)
async def to_plan(flow: Flow, node: Node) -> PlanQuery:
    return PlanQuery()


# child_returned: type(node) is UserQuery and prev is DoneOutput or ExecOutput


@Flow.transitions.on(UserQuery)
async def complete(flow: Flow, node: UserQuery) -> Node:
    agent = node.parent_agent
    if agent.config.max_iters is not None and agent.llm_turns() >= agent.config.max_iters:
        return DoneOutput(result=MAX_ITERS_EXCEEDED)
    messages = flow.build_messages(node)
    reply, usage = await join_chunks(flow.llm_for_step(node).stream(messages))
    return LLMOutput(
        content=reply,
        code=code_block(reply),
        usage=usage,
        prompt_id=agent.record_prompt(messages[0]["content"]),
    )


@Flow.transitions.on(LLMOutput)
async def to_action(flow: Flow, node: LLMOutput) -> ExecAction:
    return ExecAction(code=node.code)


@Flow.transitions.on(ExecAction)
async def run_repl(flow: Flow, node: ExecAction) -> Node:
    run = await flow.wrapped_runtime.execute(node)
    if run.status is ReplStatus.TRANSITION:
        return transition_target(flow, node, run)
    return node_for_run(node, run)
```

A blank graph is `Flow(llm, transitions=Transitions())`. More rows, same decorator:

```python
class Review(Flow):
    pass


@Review.transitions.on(ExecOutput)
async def to_review(flow: Flow, node: Node) -> ReviewQuery:
    return ReviewQuery()
```

`__init_subclass__` `derive()`s so decorating `Review.transitions` does not mutate `Flow.transitions`.

## What `.on` is

A decorator. The function **is** the dest. It returns an unattached node. It does not append.

`when=` is a guard in the same file as the producer. First matching row wins. Own rows, then `_base`.

Named choices are a second, explicit control path:

```python
Flow.transitions.choices(WorkQuery, VerifyQuery, WorkQuery)
```

This means that while `WorkQuery` is the current behavior, agent code may call `transition("verify")` or `transition("work")`. The tool is always bound in the REPL namespace, but the prompt footer only documents it when `available(node)` contains choices.

`transition(...)` ends the REPL action immediately, so stdout from that block is not emitted as an observation. `run_repl` resolves the name with `resolve_choice` and returns the selected `UserQuery` directly:

```text
ExecAction containing transition("verify") -> VerifyQuery
```

There is no intermediate `ExecOutput` and no hidden requested-transition field. An unknown name raises `InvalidTransitionError` listing the choices available from the current behavior.

## The driver

```python
async def step(self, node: Node) -> Node:
    produced = self.transitions.resolve(node)(self, node)
    landed = await produced if inspect.isawaitable(produced) else produced
    node.append(landed)
    return landed
```

`await flow.step(node)` returns the created node. The queue still uses an internal `Transition`.

## Who mutates the tree

[`node_append.md`](node_append.md): `append` returns `None`. The decorated function returns the child. `step` appends.
