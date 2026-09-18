# Node Model

`rlmflow` records every agent run as a typed tree of nodes. One agent's slice of that tree is its transcript, and it alternates between **queries**, **actions**, and **outputs**:

- **Queries** (`UserQuery`) are model-facing turns: a plan, a last-turn prod, or an explicit user message.
- **Actions** (`ActionNode`) are work the engine did on the agent's behalf: execute a block of code.
- **Outputs** (`OutputNode`) are values produced by a model, runtime, or terminal engine step. `LLMOutput` renders as the assistant reply; `ExecOutput` and `ErrorOutput` are observations fed back to the model; `DoneOutput` ends the agent.

`AgentStart` is the agent handle. `LLMOutput` is the model reply itself — there is no `LLMAction` node; `ExecAction` records the code the engine then ran.

Every action is followed by exactly one output (or a `DoneOutput`). This makes each transition auditable: the tree says what the engine decided to do and what happened next.

## Hierarchy

```text
Node
├── AgentStart      (an agent's opening query; also the root of a run)
├── UserQuery
│   ├── PlanQuery
│   ├── FinalQuery
│   └── TruncationSummary
├── ActionNode
│   └── ExecAction
│       └── AppendChild
├── OutputNode
│   ├── LLMOutput
│   ├── ExecOutput
│   ├── ErrorOutput
│   │   └── ReplDead
│   └── DoneOutput
```

`ActionNode` and `OutputNode` group nodes by what the engine executes and what its stages produce. The model "turn" is the `LLMOutput` itself. Planning, final-answer, and truncation are `UserQuery` subclasses the engine commits; a dead REPL is a `ReplDead`. `UserQuery.build_system_prompt` is the only prompt hook, so inheriting `OutputNode` does not make `LLMOutput` prompt-bearing.

An `AgentStart` is both a node in its parent's tree and the handle for a whole agent: it owns that agent's `frontier`, `config`, `sub_agents`, and system prompt table. The root of a run is just the `AgentStart` nobody launched.

## Node Fields

All nodes share:

- `type`: stable serialized discriminator, such as `"llm_output"`;
- `id`: generated node id;
- `content`: the node's text;
- `parent` / `children`: recursive structure;
- `parent_agent`: the `AgentStart` whose transcript this node belongs to;
- `root`: the `AgentStart` at the top of the run;
- `seq`: position in this node's own agent transcript, counting from 0;
- `started_at` / `finished_at`: when the step that produced the node ran, or `None` for a node written from inside a step.

The concrete payloads are:

| Class               | `type`               | Key payload                                               |
| ------------------- | -------------------- | --------------------------------------------------------- |
| `AgentStart`        | `agent_start`        | `content` (the agent's query), `config`, `system_prompts` |
| `UserQuery`         | `user_query`         | `content`                                                 |
| `PlanQuery`         | `plan_query`         | planned next-action prompt before each working turn       |
| `FinalQuery`        | `final_query`        | last-iter prompt (default)                                |
| `TruncationSummary` | `truncation_summary` | keep_n notice (default)                                   |
| `LLMOutput`         | `llm_output`         | `content` (the reply), `code`, `usage`, `prompt_id`       |
| `ExecAction`        | `exec_action`        | `code`                                                    |
| `ExecOutput`        | `exec_output`        | `content` (the REPL's stdout)                             |
| `ErrorOutput`       | `error_output`       | execution error text, `error="exec"`                      |
| `ReplDead`          | `repl_dead`          | crash text plus the cold-REPL note, `error="repl"`        |
| `DoneOutput`        | `done_output`        | `result`, `content`                                       |

`LLMOutput.code` is the block extracted from the reply; `ExecAction.code` is what the engine actually ran. `LLMOutput.prompt_id` keys into `agent.system_prompts`, so the exact system prompt a turn ran under is recoverable with `agent.system_prompt_for(node)`. `DoneOutput` is the only terminal node: an agent is finished when `agent.terminal` is true, which means its frontier is a `DoneOutput`. The only completion tool is `finish(...)`.

## Navigating

An agent's own transcript is a chain; the tree branches only where an agent delegated.

```python
node.next          # the following node in this node's own agent, or None
node.prev          # the previous one, or None at the agent's start
node.walk()        # this node and everything below it, including sub-agents
node.iter_backwards()  # back up this agent's own chain to its AgentStart
agent.transcript() # that chain, start to frontier, in order
agent.frontier     # where the agent is now
agent.sub_agents   # the agents it launched
agent.leaves()     # the frontier of this agent and of every agent below it
```

`node.append(child)` is the only link primitive. It hangs a node off this one and moves the agent's frontier there, so appending anywhere but the frontier raises. It returns `None`; the caller already has the child. Appending an `AgentStart` opens a sub-agent instead: it branches off without moving the parent's frontier.

## Chat Projection

`node.render()` is how a node becomes zero or more chat messages. User-like nodes (`AgentStart`, `UserQuery`, `ExecOutput`, `ErrorOutput`) render as `user`; `LLMOutput` as `assistant`; bookkeeping (`ExecAction`, `DoneOutput`) as an empty list. A `UserQuery` subclass inherits the user turn, and a custom node may return an assistant/user pair or any other ordered message list.

`node.project(keep=...)` walks `prev`, calls each node's canonical `render()`, flattens the results, and stops after `keep` messages. `Flow.build_messages` projects history from `node.prev`, renders only the current frontier through the Flow or prompt profile's `render_fn(runtime, node)`, and prepends the system prompt. It preserves rendered messages in order, including adjacent messages with the same role.

## Normal Flow

A one-turn successful run looks like this:

```text
AgentStart
  -> PlanQuery
  -> LLMOutput(code="finish('answer')")
  -> ExecAction
  -> DoneOutput(result="answer")
```

A multi-turn run loops through LLM and exec halves:

```text
AgentStart
  -> PlanQuery
  -> LLMOutput(code="x = compute()")
  -> ExecAction
  -> ExecOutput(content="...")
  -> PlanQuery
  -> LLMOutput(code="finish({'answer': x})")
  -> ExecAction
  -> DoneOutput(result={"answer": ...})
```

Without `output_schema`, `DoneOutput.result`, `Flow.run()`, and child result handles keep the strict JSON value passed to `finish` (strings stay strings; dictionaries with string keys, lists, finite numbers, booleans, and `None` stay typed). Tuples, sets, bytes, non-finite floats, arbitrary objects, dictionaries with non-string keys, and nested occurrences of those values produce an execution error rather than being stringified. With an explicit schema, a JSON value or JSON-encoded string is parsed and stored as the validated typed value.

Errors are observations too. The next LLM turn sees the traceback and can recover:

```text
LLMOutput(code="1 / 0")
  -> ExecAction
  -> ErrorOutput(error="exec", content="ZeroDivisionError: ...")
  -> PlanQuery
  -> LLMOutput(code="finish(...)")
```

An `ErrorOutput` means the agent's own code raised. A `ReplDead` means the REPL itself died and took the namespace with it; the agent is told that its variables are gone before the next model request.

Every ordinary action turn lands a `PlanQuery` before its `LLMOutput`, including startup and turns after REPL output, errors, or a returned child. Its user content is `Turn N/M:` (plus a first-turn INPUTS safeguard). The system message for that turn is `node.build_system_prompt(flow)` on the `UserQuery`: the inherited protocol, plus the orchestrator addendum when the agent can spawn, or the worker addendum when it cannot. `FinalQuery` replaces `PlanQuery` on the last allowed iteration and wraps the protocol with last-turn submit guidance. `TruncationSummary` records a `keep_n` overflow, and application code may append a generic `UserQuery` explicitly. Successful REPL blocks continue without an explicit model-facing transition call.

## Delegation

Agents delegate with:

```python
finding_schema = {
    "type": "object",
    "properties": {"conclusion": {"type": "string"}},
    "required": ["conclusion"],
    "additionalProperties": False,
}
search = await launch_subagent(
    "Find the evidence",
    model="default",
    name="search",
    inputs={"chunk": chunk},
    output_schema=finding_schema,
)
# Later, when the answer is needed:
result = await search.wait_for_result()
assert set(result) == {"conclusion"}
```

Launching returns a handle without waiting for the child. An ordinary child initially hangs off the `ExecAction` that launched it, and the step lands one `ExecOutput` when the parent's block finishes:

```text
ExecAction
├── AgentStart "Find the evidence"   (root.search, runs to its own DoneOutput)
└── ExecOutput                       (the parent's own sequel)
```

`node.next` skips the sub-agent branches, so the parent's transcript reads as one chain whether or not it delegated. Every child remains beneath the `ExecAction` that launched it. Handle-valued inputs add dataflow edges between those agents without changing ownership. Independent children run concurrently when their calls are passed to `asyncio.gather`. Each child's answer is its `DoneOutput.result`; a child that crashes has the failure recorded as its answer, so its siblings still settle. A result not routed through another agent's inputs also reaches its parent as a `[handoff]` `UserQuery`: a mapping is listed as fields, a scalar as labeled JSON.

A host can set `AgentConfig.child_output_schema` so launched children get a schema without the parent passing `output_schema=` on each call. Launch `output_schema=` still wins. The coding CLI uses `CODING_HANDOFF_SCHEMA` this way; evals do not.

## Streaming Semantics

`flow.run_streaming(root, until=...)` advances the run and yields each node as it lands; the tree is mutated in place. One logical reasoning turn is usually several node transitions:

1. LLM half: `AgentStart | UserQuery | ExecOutput | ErrorOutput -> LLMOutput`.
1. Exec half: `LLMOutput -> ExecAction -> <observation>`.

Each pass, the engine steps every leaf in the tree — the frontier of every agent that has not answered and is not already being stepped — so the parent and its children advance together:

- an agent whose frontier is a `DoneOutput` does nothing unless `resume_on_child_done` appended a `UserQuery`;
- an agent at `LLMOutput` runs its code next;
- an agent at `AgentStart`, `UserQuery`, `ExecOutput`, or `ErrorOutput` calls the model next (a `UserQuery` after `ExecOutput` / `DoneOutput` takes a `PlanQuery` first);
- an agent at `ExecAction` runs that block — but while a block is running, its agent's frontier is that action and the step already owns it, so the leaves that advance are its children's.

## Persistence

`agent.save(path)` writes the tree; `AgentStart.load(path)` reads it back. Serialization writes one shallow record per Node with `parent_id`, then rebuilds the tree iteratively in parent-before-child order, so a loaded run is the same tree with the same ids.

Saved run directories keep a per-agent projection under `agents/<name>/` alongside the flat `graph.json` document. Cross-agent ownership is always the tree's launch structure. Dependency dataflow is derived from stable agent IDs persisted as handle-valued input references and can be rendered as a separate projection over the same nodes.

Walk the tree and switch on `node.type` (or `isinstance`) when inspecting runs:

```python
for node in root.walk():
    if isinstance(node, AgentStart):
        print(node.config.path, "started:", node.content)
    elif isinstance(node, DoneOutput):
        print("result:", node.result)
```
