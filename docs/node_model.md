# Node Model

`rlmflow` records every agent run as a typed tree of nodes. One agent's slice of
that tree is its transcript, and it alternates between **observations** and
**actions**:

- **Observations** are inputs the agent received or observed: its opening query,
  a user query, an LLM reply, REPL output, an error, or a terminal result.
- **Actions** are work the engine did on the agent's behalf: execute a block of
  code.

Every action is followed by exactly one observation. This makes each transition
auditable: the tree says what the engine decided to do and what happened next.

## Hierarchy

```text
Node
├── AgentStart      (an agent's opening query; also the root of a run)
├── UserQuery
├── LLMOutput
├── ExecAction
├── ExecOutput
├── ErrorOutput
└── DoneOutput
```

There are seven concrete node types and one base class. Every node is a
dataclass carrying `content`; there is no separate observation/action base. The
model "turn" is the `LLMOutput` observation itself — there is no `LLMAction`
node; `ExecAction` records the code the engine then ran.

An `AgentStart` is both a node in its parent's tree and the handle for a whole
agent: it owns that agent's `frontier`, `config`, `sub_agents`, and system
prompt table. The root of a run is just the `AgentStart` nobody launched.

## Node Fields

All nodes share:

- `type`: stable serialized discriminator, such as `"llm_output"`;
- `id`: generated node id;
- `content`: the node's text;
- `parent` / `children`: recursive structure;
- `parent_agent`: the `AgentStart` whose transcript this node belongs to;
- `root`: the `AgentStart` at the top of the run;
- `seq`: position in this node's own agent transcript, counting from 0;
- `started_at` / `finished_at`: when the step that produced the node ran, or
  `None` for a node written from inside a step.

The concrete payloads are:

| Class | `type` | Key payload |
| --- | --- | --- |
| `AgentStart` | `agent_start` | `content` (the agent's query), `config`, `system_prompts` |
| `UserQuery` | `user_query` | `content` |
| `LLMOutput` | `llm_output` | `content` (the reply), `code`, `usage`, `prompt_id` |
| `ExecAction` | `exec_action` | `code` |
| `ExecOutput` | `exec_output` | `content` (the REPL's stdout) |
| `ErrorOutput` | `error_output` | `content`, `error` (`"exec"` or `"repl"`) |
| `DoneOutput` | `done_output` | `result`, `content` |

`LLMOutput.code` is the block extracted from the reply; `ExecAction.code` is
what the engine actually ran. `LLMOutput.prompt_id` keys into
`agent.system_prompts`, so the exact system prompt a turn ran under is
recoverable with `agent.system_prompt_for(node)`. `DoneOutput` is the only
terminal node: an agent is finished when `agent.terminal` is true, which means
its frontier is a `DoneOutput`. The tool that lands one is `finish(...)`; `done`
stays bound to the same function so older agent code and saved runs still read.

## Navigating

An agent's own transcript is a chain; the tree branches only where an agent
delegated.

```python
node.next          # the following node in this node's own agent, or None
node.prev          # the previous one, or None at the agent's start
node.walk()        # this node and everything below it, including sub-agents
node.walk(reverse=True)  # back up this agent's own chain to its AgentStart
agent.transcript() # that chain, start to frontier, in order
agent.frontier     # where the agent is now
agent.sub_agents   # the agents it launched
agent.leaves()     # the frontier of this agent and of every agent below it
```

`node.append(child)` is the only link primitive. It hangs a node off this one
and moves the agent's frontier there, so appending anywhere but the frontier
raises. Appending an `AgentStart` opens a sub-agent instead: it branches off
without moving the parent's frontier.

## Normal Flow

A one-turn successful run looks like this:

```text
AgentStart
  -> LLMOutput(code="finish('answer')")
  -> ExecAction
  -> DoneOutput(result="answer")
```

A multi-turn run loops through LLM and exec halves:

```text
AgentStart
  -> LLMOutput(code="x = compute()")
  -> ExecAction
  -> ExecOutput(content="...")
  -> LLMOutput(code="finish(x)")
  -> ExecAction
  -> DoneOutput(result="...")
```

Errors are observations too. The next LLM turn sees the traceback and can
recover:

```text
LLMOutput(code="1 / 0")
  -> ExecAction
  -> ErrorOutput(error="exec", content="ZeroDivisionError: ...")
  -> LLMOutput(code="finish(...)")
```

`error="exec"` means the agent's own code raised. `error="repl"` means the REPL
itself died and took the namespace with it; the agent is told so in the same
node, because its variables are gone.

A turn can also land a `UserQuery` before its `LLMOutput`: the prompt builder's
content for that turn, a nudge when the previous node did not end on a user
turn, or the final-answer prod on the last allowed iteration.

## Delegation

Agents delegate with:

```python
search = await launch_subagent(
    "Find the evidence",
    name="search",
    inputs={"chunk": chunk},
)
# Later, when the answer is needed:
result = await search.wait_for_result()
```

Launching returns a handle without waiting for the child. Children hang off the
`ExecAction` that launched them, and the step lands one `ExecOutput` when the
parent's block finishes:

```text
ExecAction
├── AgentStart "Find the evidence"   (root.search, runs to its own DoneOutput)
└── ExecOutput                       (the parent's own sequel)
```

`node.next` skips the sub-agent branches, so the parent's transcript reads as
one chain whether or not it delegated. Independent children run concurrently
when their calls are passed to `asyncio.gather`. Each child's answer is its
`DoneOutput.result`; a child that crashes has the failure recorded as its
answer, so its siblings still settle.

## Streaming Semantics

`flow.run_streaming(root, until=...)` advances the run and yields each node as
it lands; the tree is mutated in place. One logical reasoning turn is usually
several node transitions:

1. LLM half: `AgentStart | UserQuery | ExecOutput | ErrorOutput -> LLMOutput`.
2. Exec half: `LLMOutput -> ExecAction -> <observation>`.

Each pass, the engine steps every leaf in the tree — the frontier of every agent
that has not answered and is not already being stepped — so the parent and its
children advance together:

- an agent whose frontier is a `DoneOutput` does nothing;
- an agent at `LLMOutput` runs its code next;
- an agent at `AgentStart`, `UserQuery`, `ExecOutput`, or `ErrorOutput` calls the
  model next;
- an agent at `ExecAction` runs that block — but while a block is running, its
  agent's frontier is that action and the step already owns it, so the leaves
  that advance are its children's.

## Persistence

`agent.save(path)` writes the tree; `AgentStart.load(path)` reads it back.
Serialization derives `order` from each node's `seq` and rebuilds the tree from
its `children`, so a loaded run is the same tree with the same ids.

Saved run directories keep a per-agent projection under `agents/<name>/`
alongside the recursive `graph.json` manifest. Cross-agent edges are the tree's
own structure — a child `AgentStart` under an `ExecAction` — so there is no
separate edge object to maintain by hand.

Walk the tree and switch on `node.type` (or `isinstance`) when inspecting runs:

```python
for node in root.walk():
    if isinstance(node, AgentStart):
        print(node.config.path, "started:", node.content)
    elif isinstance(node, DoneOutput):
        print("result:", node.result)
```
