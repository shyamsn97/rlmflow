# Node Model

`rlmflow` records every agent run as a typed trajectory. The trajectory is a
strict alternation of **observations** and **actions**:

- **Observations** are inputs the system received or observed: a user query, an
  LLM reply, REPL output, a suspension, an error, or a terminal result.
- **Actions** are work the system did: execute code, or resume a suspended
  runtime.

Every action is followed by exactly one observation. This makes each transition
auditable: the graph says what the engine decided to do and what happened next.

## Hierarchy

```text
Node
├── ObservationNode          (adds `content`)
│   ├── UserQuery
│   ├── LLMOutput
│   ├── ExecOutput
│   ├── SupervisingOutput
│   ├── ErrorOutput
│   └── DoneOutput
└── ActionNode
    ├── ExecAction
    └── ResumeAction
```

There are eight concrete node types under three base classes. The model "turn"
is the `LLMOutput` observation itself — there is no separate `LLMAction` node;
`ExecAction` records the code the engine then ran.

## Node Fields

All nodes share:

- `type`: stable serialized discriminator, such as `"llm_output"`;
- `id`: generated node ID;
- `agent_id`: owning agent ID, such as `"root"` or `"root.search"`;
- `seq`: per-agent sequence number;
- `metadata`: free-form dict (e.g. `metadata["usage"]` holds token deltas).

`ObservationNode` subclasses add a `content` string. The concrete payloads are:

| Class | `type` | Base | Key payload |
| --- | --- | --- | --- |
| `UserQuery` | `user_query` | `ObservationNode` | `content` |
| `LLMOutput` | `llm_output` | `ObservationNode` | `content` (the reply), `code`, `metadata["usage"]` |
| `ExecAction` | `exec_action` | `ActionNode` | `code` |
| `ExecOutput` | `exec_output` | `ObservationNode` | `output`, `content` |
| `SupervisingOutput` | `supervising_output` | `ObservationNode` | `output`, `waiting_on` |
| `ErrorOutput` | `error_output` | `ObservationNode` | `error`, `output`, `content` |
| `DoneOutput` | `done_output` | `ObservationNode` | `result`, `output`, `content` |
| `ResumeAction` | `resume_action` | `ActionNode` | `resumed_from` |

`LLMOutput.code` is the source of truth for executed code; `ExecAction.code` is a
debug/UI echo of what ran. `ResumeAction.resumed_from` lists the children whose
completion unpaused a suspended parent. `DoneOutput` is the only terminal node.

## Normal Flow

A one-turn successful run looks like this:

```text
UserQuery
  -> LLMOutput(code="done('answer')")
  -> ExecAction
  -> DoneOutput(result="answer")
```

A multi-turn run loops through LLM and exec halves:

```text
UserQuery
  -> LLMOutput(code="x = compute()")
  -> ExecAction
  -> ExecOutput(output="...")
  -> LLMOutput(code="done(x)")
  -> ExecAction
  -> DoneOutput(result="...")
```

Errors are observations too. The next LLM turn sees the error message and can
recover:

```text
LLMOutput(code="1 / 0")
  -> ExecAction
  -> ErrorOutput(error="exec_exception", output="ZeroDivisionError: ...")
  -> LLMOutput(code="done(...)")
```

If the LLM reply contains no executable code block, the engine records a normal
exec half with `ErrorOutput(error="no_code_block")`.

## Delegation And Resume Flow

Agents delegate with:

```python
results = await launch_subagents([
    {"name": "search", "query": "Find the evidence", "inputs": {"chunk": chunk}},
])
```

When code awaits `launch_subagents(...)`, the parent runtime suspends and the
engine writes:

```text
ExecAction
  -> SupervisingOutput(waiting_on=["root.search"])
```

The scheduler then runs the child agent. When all children listed in
`waiting_on` are terminal, the parent becomes runnable again:

```text
SupervisingOutput(waiting_on=["root.search"])
  -> ResumeAction(resumed_from=["root.search"])
  -> ExecOutput(output="...")
```

After the resume observation, the parent returns to normal LLM/exec flow.

## Streaming Semantics

`flow.run_streaming(graph=graph, until=..., n=...)` advances the run and yields the
`Event`s emitted; the graph is mutated in place. One logical reasoning turn is
usually multiple node transitions:

1. LLM half: `ObservationNode -> LLMOutput`.
2. Exec half: `LLMOutput -> ExecAction -> <observation>`.

Resume is also an observation-to-observation transition:

```text
SupervisingOutput -> ResumeAction -> <observation>
```

The scheduler decides which agents are runnable:

- finished agents do nothing;
- an agent at `LLMOutput` runs code next;
- an agent at `UserQuery`, `ExecOutput`, or `ErrorOutput` calls the LLM next;
- an agent at `SupervisingOutput` resumes only after all children in
  `waiting_on` are terminal;
- otherwise, the scheduler descends into unfinished children.

## Persistence

Node sequence numbers are assigned when nodes are appended. Callers populate
payload fields; the engine assigns `agent_id`, `seq`, and `id`.

Saved run directories persist the per-agent trajectory under `agents/<agent-id>/`,
while the recursive graph manifest links agents through `Graph.children`.
Cross-agent edges are derived from the recursive structure and `SupervisingOutput`
wait sets; there is no separate edge object to maintain by hand.

Walk the tree and switch on `node.type` (or `isinstance`) when inspecting traces:

```python
for agent in graph.walk():
    for node in agent.nodes:
        if node.type == "supervising_output":
            print(node.agent_id, "waiting on", node.waiting_on)
        elif node.type == "done_output":
            print("result:", node.result)
```
