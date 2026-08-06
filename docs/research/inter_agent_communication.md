# `AGENTS`: Tree-Based Agent Discovery

> **Status:** Implemented, unreleased.
>
> **Scope:** A read-only `AGENTS` variable for inspecting the agents in one
> recursive run. This design does not add message passing or control.

## Summary

When enabled, every agent REPL receives an `AGENTS` variable alongside
`INPUTS`:

```python
flow = Flow(..., use_agent_tree=True)

AGENTS.get()
AGENTS.get("researcher")
AGENTS.get_parent()
AGENTS.get_siblings()
AGENTS.get_children()
AGENTS.print_graph()
```

`AGENTS` is a viewer-scoped snapshot of the recursive agent tree. It contains:

- each agent's id, name, path, and depth;
- parent and child relationships;
- current lifecycle status;
- frontier metadata;
- completed results.

Example:

```python
print(AGENTS.get_siblings())
AGENTS.print_graph(show_results=True)
```

```text
root [waiting]
├── researcher [completed] -> "Parser accepts duplicate keys."
├── reviewer [running]
└── writer [running] (you)
```

This should remain a small feature:

- default to disabled;
- build one immutable tree snapshot before each REPL action;
- inject it by value as `AGENTS`;
- implement all lookups and rendering as local, read-only methods;
- refresh it on the next action;
- mention it in the default prompt only when enabled;
- make no graph, scheduler, or persistence changes.

## Why One Variable

A single namespace is cleaner than adding several unrelated global tools:

```python
# Less desirable
get_parent()
get_siblings()
get_children()
get_agent("researcher")
print_agent_graph()

# Proposed
AGENTS.get_parent()
AGENTS.get_siblings()
AGENTS.get_children()
AGENTS.get("researcher")
AGENTS.print_graph()
```

The variable communicates that these operations query one coherent tree. It
also leaves room for additional read-only operations without occupying more
global REPL names.

`AGENTS` mirrors the existing `INPUTS` convention:

```text
INPUTS   task data visible to this agent
AGENTS   recursive run topology visible to this agent
```

## Goals

The first implementation should:

- default to disabled so ordinary runs pay no prompt, snapshot, or serialization
  cost;
- enable with one Flow-level `use_agent_tree=True` option;
- inject `AGENTS` before each `ExecAction` only when enabled;
- expose the whole agent tree for the current root;
- make relative queries default to the current agent;
- include completed results;
- show useful running, waiting, idle, and completed states;
- provide a compact ASCII tree renderer;
- work without mutating the Node tree;
- remain deterministic during one REPL action;
- work with local and remote runtimes through by-value injection;
- reserve the `AGENTS` name.

## Opt-In Configuration

Add one Flow-level flag:

```python
flow = Flow(
    llm,
    use_agent_tree=True,
)
```

Default:

```python
use_agent_tree = False
```

The flag controls the complete feature:

```text
use_agent_tree=False
  no AGENTS variable
  no tree snapshot construction
  no remote serialization
  no default-prompt mention

use_agent_tree=True
  inject AGENTS before every REPL action
  refresh its tree snapshot each action
  include concise default-prompt guidance
```

Do not inject an empty or disabled sentinel. When the feature is off, `AGENTS`
should be absent so `"'AGENTS' in globals()"` accurately reports capability.

Use one option rather than separate `inject_agent_tree` and
`prompt_agent_tree` flags. Exposing a framework variable without telling the
default model about it is rarely useful, while prompting for a missing variable
is incorrect. Callers that replace the system prompt through `PromptProfile`
already own the decision to document available variables there.

Do not auto-enable based on the current number of children. The namespace
should remain stable for the lifetime of a Flow, and the root may need to know
about `AGENTS` before it creates children.

## Non-Goals

`AGENTS` does not:

- send messages;
- wait for status changes;
- subscribe to events;
- cancel, resume, or steer agents;
- expose REPL variables or full transcripts;
- search agents in another root;
- keep agents alive after their stream stops;
- guarantee that a status remains current;
- add a new durable Node type.

It is a read-only observation object, not an agent manager.

## User API

### Current Agent

```python
AGENTS.get() -> AgentInfo
AGENTS.self -> AgentInfo
```

Both return the viewer's own snapshot.

### Lookup

```python
AGENTS.get(selector: str | None = None) -> AgentInfo | None
```

Resolution order:

1. no selector: current agent;
2. exact opaque `AgentStart.id`;
3. exact `AgentConfig.path`;
4. direct relative name among parent, siblings, and children;
5. globally unique name in this root.

If a global name is ambiguous, raise `ValueError` and ask for a path or id. If
there is no match, return `None`.

Examples:

```python
AGENTS.get("researcher")
AGENTS.get("root.researcher")
AGENTS.get("agent_7c...")
```

### Parent

```python
AGENTS.get_parent(agent: str | AgentInfo | None = None) -> AgentInfo | None
```

With no argument, returns the current agent's direct parent. A root returns
`None`.

Passing a selector or an `AgentInfo` queries another visible tree node:

```python
AGENTS.get_parent("root.researcher")
```

### Children

```python
AGENTS.get_children(
    agent: str | AgentInfo | None = None,
) -> list[AgentInfo]
```

Returns direct children in creation order.

### Siblings

```python
AGENTS.get_siblings(
    agent: str | AgentInfo | None = None,
) -> list[AgentInfo]
```

Returns the parent's direct children excluding the selected agent, in creation
order. A root returns an empty list.

Separate roots passed to `parallel_stream(...)` are not siblings. `AGENTS`
contains only the viewer's recursive root.

### All Agents

```python
AGENTS.all() -> list[AgentInfo]
```

Returns the root followed by descendants in tree order.

This is useful for compact filtering:

```python
finished = [agent for agent in AGENTS.all() if agent.status == "completed"]
```

### Graph Rendering

```python
AGENTS.render_graph(
    *,
    show_results: bool = False,
    max_result_chars: int = 160,
) -> str

AGENTS.print_graph(
    *,
    show_results: bool = False,
    max_result_chars: int = 160,
) -> None
```

`render_graph()` returns text for tests or further processing.
`print_graph()` prints the same text because only stdout is shown to the model
after a REPL action.

Default output:

```text
root [waiting]
├── researcher [completed]
├── reviewer [running]
└── writer [running] (you)
```

With results:

```text
root [waiting]
├── researcher [completed] -> "Parser accepts duplicate keys."
├── reviewer [running]
└── writer [running] (you)
```

Result previews must be single-line and bounded. The full normalized result
remains available through `AGENTS.get(...).get_result()`.

## Data Model

Keep the REPL object independent from live `Node` objects:

```python
@dataclass(frozen=True, slots=True)
class AgentFrontier:
    node_id: str
    type: str
    seq: int


@dataclass(frozen=True, slots=True)
class AgentInfo:
    agent_id: str
    name: str
    path: str
    depth: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    status: Literal["running", "waiting", "idle", "completed"]
    frontier: AgentFrontier
    _result: object | None = None

    def get_result(self) -> object | None: ...


@dataclass(frozen=True)
class AgentDirectory:
    viewer_id: str
    root_id: str
    agents: tuple[AgentInfo, ...]
```

`AgentDirectory` builds private lookup maps in `__post_init__` or computes them
from the small tuple. Its public methods are the `AGENTS` API.

The snapshot must contain no:

- `Node`;
- `AgentStart`;
- `Flow`;
- `TaskQueue`;
- `Runtime`;
- asyncio task;
- host callback.

This makes it safe to copy into a remote REPL. Model code cannot mutate the
host graph through the snapshot.

## Building the Tree

`Flow` owns snapshot construction because it can see both durable graph state
and the live queue:

```python
def agent_directory(self, viewer: AgentStart) -> AgentDirectory:
    agents = [
        node
        for node in viewer.root.walk()
        if isinstance(node, AgentStart)
    ]
    return AgentDirectory(
        viewer_id=viewer.id,
        root_id=viewer.root.id,
        agents=tuple(self.agent_info(agent) for agent in agents),
    )
```

Parent identity is derived from the launch edge:

```python
def supervisor_of(agent: AgentStart) -> AgentStart | None:
    if agent.parent is None:
        return None
    return agent.parent.parent_agent
```

`AgentStart.parent_agent` is the agent itself, so it must not be used as the
supervisor relationship.

Children come from `agent.sub_agents`, which already preserves creation order.

## Status Semantics

Status is captured when `AGENTS` is built.

### `completed`

The agent's frontier is `DoneOutput`.

```python
agent.terminal
```

Its normalized `result` is included.

### `waiting`

The agent has an in-flight `ExecAction` with at least one attached unfinished
child.

This identifies the common case where a parent is blocked in
`launch_subagents(...)`.

### `running`

The agent has an in-flight queue task and is not classified as `waiting`.

This includes active LLM and REPL work.

### `idle`

The agent is unfinished and has no in-flight queue task.

Examples include a paused stream, an attached child that has not been
submitted, or a loaded graph that is not being driven.

### Queue Lookup

The first implementation can scan `TaskQueue.running`:

```python
def running_node(queue: TaskQueue | None, agent: AgentStart) -> Node | None:
    if queue is None:
        return None
    return next(
        (
            node
            for node, _task in queue.running.values()
            if node.parent_agent is agent
        ),
        None,
    )
```

This is O(active leaves) per agent and O(tree size × active leaves) for one
snapshot. Agent trees are expected to be small, and one snapshot is built per
REPL action. Do not add another queue index unless profiling justifies it.

## Snapshot Freshness

When `use_agent_tree=True`, `AGENTS` is refreshed before every `ExecAction`, at
the same time as `INPUTS` and framework tools:

```python
repl.seed(self.build_tools(action, repl), agent.config.inputs)
```

`Flow.build_tools(...)` conditionally adds:

```python
namespace = {
    "done": ...,
    "launch_subagents": ...,
    "INPUTS": node.parent_agent.config.inputs,
}
if self.use_agent_tree:
    namespace["AGENTS"] = self.agent_directory(node.parent_agent)
```

The object remains fixed for the duration of that REPL action. If a sibling
finishes while the action is running, repeated calls to
`AGENTS.get_siblings()` in the same block return the original snapshot.

This is intentional:

- no host round trips;
- no races while rendering;
- identical behavior in local and remote REPLs;
- no polling loops that bypass model turns;
- no scheduler changes.

The next agent action receives a fresh snapshot. When the option is disabled,
the directory builder is never called.

## Local and Remote Runtimes

### Local

When enabled, `LocalRepl.seed(...)` stores the immutable `AgentDirectory`
object directly.

### Remote

When enabled, `RemoteRepl.seed(...)` ships `AGENTS` by value through the
existing object injection path. The object contains no live host references.

The current by-value path uses `cloudpickle` for custom class instances. The
shipped subprocess, Docker, and Modal environments must therefore include the
`cloudpickle` extra when `AGENTS` is enabled.

If avoiding that optional dependency is a requirement, add a narrow wire
encoding later:

```text
AgentDirectory -> JSON data -> reconstruct AgentDirectory in sandbox
```

Do not turn each method into a host-proxied tool merely to avoid serialization.
That would make a simple snapshot API live, timing-dependent, and much more
complex.

## Result Normalization

Most results are already strings or schema-parsed JSON values. A
controller-authored `DoneOutput` may contain an arbitrary object.

Normalize results before placing them in `AGENTS`:

```python
def json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)
```

This prevents a result from reintroducing live references or requiring custom
serialization.

`render_graph(show_results=True)` converts the normalized value to a compact
single-line preview and truncates it to `max_result_chars`.

## Prompt Guidance

`AGENTS` is not callable, so the dynamic callable-tool section will not list
it. Do not add it unconditionally to the static REPL documentation.

Add a dynamic prompt section that returns an empty string unless
`flow.use_agent_tree` is true. When enabled, render a concise section beside
the `INPUTS` guidance:

```text
`AGENTS` is a read-only snapshot of this run's agent tree. Use
`AGENTS.get(...)`, `.get_parent()`, `.get_siblings()`, `.get_children()`, or
`.print_graph()`. Call `AgentInfo.get_result()` for a completed agent's result.
The snapshot is refreshed before each REPL action and does not wait, message,
cancel, or steer agents.
```

The prompt should discourage status polling. Calling a snapshot method twice
in one action cannot produce fresher data.

The conditional prompt section and namespace injection must use the same flag.
Tests should prevent either mismatch:

- documented but absent;
- injected but undocumented by the default prompt.

A caller using a fully custom `PromptProfile.system` is responsible for
documenting `AGENTS`, just as it is responsible for documenting the rest of its
custom REPL contract.

## Reserved Namespace

Update:

```python
RESERVED_TOOLS = frozenset({
    "done",
    "launch_subagents",
    "INPUTS",
    "AGENTS",
})
```

The constant may eventually be renamed to `RESERVED_NAMES`, but that rename is
not required for this feature.

`flow.inject("AGENTS", ...)` and `flow.remove_tool("AGENTS")` must be rejected,
matching `INPUTS`.

## Host-Side API

The same immutable types are useful outside a REPL:

```python
directory = flow.agent_directory(agent)
directory.print_graph(show_results=True)
directory.get_siblings()
```

Avoid adding separate `AgentStart.siblings` or `AgentStart.supervisor`
properties initially. `AgentDirectory` centralizes:

- tree relationship derivation;
- status calculation;
- result normalization;
- lookup rules;
- rendering.

Durable Nodes remain unaware of the live queue.

## Persistence and Replay

No persistence format change is required.

`AGENTS` is reconstructed from:

- the loaded Node tree;
- the currently active `Flow.queue`, when present.

For an inactive loaded graph:

- terminal agents are `completed`;
- unfinished agents are `idle`.

Replay receives snapshots like any normal action. Querying them has no side
effects and cannot duplicate graph work.

## Complexity

For `N` agents and `F` active queue tasks:

```text
build snapshot          O(N × F) initially
AGENTS.get(id/path)     O(1)
AGENTS.get_children     O(number of direct children)
AGENTS.get_siblings     O(number of siblings)
AGENTS.all              O(N)
render graph            O(N)
```

If `N × F` becomes material, build one `running_by_agent_id` dictionary during
snapshot construction and reduce the build to O(N + F). This is a local
optimization and does not require changing `TaskQueue`.

## Compatibility

Existing behavior remains unchanged:

- `use_agent_tree` defaults to false;
- default prompts and REPL namespaces remain unchanged unless enabled;
- `launch_subagents(...)` still blocks and returns child results;
- stream output remains Nodes;
- scheduling and cancellation do not change;
- no Node type or run-format version changes;
- `INPUTS` retains its current behavior.

Potential compatibility changes:

- `AGENTS` becomes a reserved injection name;
- remote runtimes need custom-object serialization unless a JSON reconstruction
  path is added.

## Implementation Plan

### 1. Immutable Directory

Add `rlmflow/tools/agents.py` containing:

- `AgentFrontier`;
- `AgentInfo`;
- `AgentDirectory`;
- lookup and relationship methods;
- graph rendering;
- result normalization.

Keep these types independent from `Flow` and live Nodes.

### 2. Snapshot Builder

Add to `Flow`:

```python
agent_directory(viewer: AgentStart) -> AgentDirectory
```

Use the viewer's root tree and the active queue to build one snapshot.

### 3. Opt-In Configuration and Injection

Add `Flow(use_agent_tree: bool = False)`. When true, add `AGENTS` in
`build_tools(...)`; otherwise do not build or inject it. Reserve the name in
either mode and verify that every enabled `exec_step` refreshes it.

### 4. Prompt and Exports

Add a conditional default-prompt section controlled by the same
`use_agent_tree` flag. Export the immutable classes from `rlmflow`.

### 5. Documentation and Example

Add:

- a short section to `docs/control.md`;
- a note in `docs/observability.md`;
- `examples/control/inspect_agents.py`;
- a changelog entry.

The example should have one child inspect siblings and call:

```python
AGENTS.print_graph(show_results=True)
completed = [
    sibling
    for sibling in AGENTS.get_siblings()
    if sibling.status == "completed"
]
```

## Test Plan

Required tests:

1. The option defaults to false.
2. Disabled runs have no `AGENTS` variable, prompt text, snapshot build, or
   serialization.
3. Enabled runs inject `AGENTS` and include the default prompt guidance.
4. `AGENTS.get()` and `AGENTS.self` identify the viewer.
5. Parent, children, and siblings follow the launch tree.
6. Creation order is preserved.
7. Lookup works by id, path, direct relative name, and unique global name.
8. Ambiguous global names raise.
9. A separate parallel root is absent.
10. Completed agents include normalized results.
11. Running, waiting, idle, and completed statuses are classified correctly.
12. The viewer is marked `(you)` in graph output.
13. Graph rendering preserves tree structure and bounds result previews.
14. `AGENTS` contains no live Node, Flow, Runtime, queue, or task.
15. The snapshot does not change during one REPL action.
16. A later action receives a refreshed snapshot.
17. Local and remote REPLs expose the same methods and values.
18. Injecting or removing `AGENTS` is rejected in either mode.
19. A custom system prompt is not modified by the default conditional section.
20. Save/load and replay require no special handling.

## Recommendation

Implement `AGENTS` as an opt-in, by-value tree snapshot, not as a collection of
proxy tools.

The complete data flow is:

```text
Node tree + active TaskQueue
  -> Flow.agent_directory(viewer)
  -> immutable AgentDirectory
  -> REPL variable AGENTS
  -> local get / relationship / render methods
```

That gives agents a clean tree-oriented view of parallel work while keeping the
feature read-only and separate from scheduling, persistence, and communication.
