# Minimal Dynamic Agent Graph

## Thesis

```text
Graph  the full picture. Every durable fact is a node or an edge.
Flow   a driver. It reads the graph, runs steps, and writes results back.
```

`Graph` is a small store: an ordered log, raw lookups, a few explicit writes,
and save/load. It interprets three structural edge kinds only to enforce
invariants. Everything agent-shaped lives in two function modules over that
store:

```text
rlmflow/graph/graph.py   Graph: add_edge, remove_edge, merge, as_of, node, nodes, edges, items, save, load
rlmflow/graph/reads.py   pure queries: history, frontier, terminal, inbox, ready, ...
rlmflow/graph/ops.py     history writes: append, spawn, send, deliver, fork, attach
```

Every feature rlmflow has is a composition of those writes: launching,
dependencies, handoff, injection, fork, attach, merge, and direct messaging.
Each write names one operation and checks that operation's own preconditions;
there is no generic batch writer. Flow writes only through `ops` and schedules
only from `reads`.

## Audit: where the complexity comes from

Today one structural tree, rooted at an `AgentStart`, does six jobs:

| Job | Current mechanism |
|-----|-------------------|
| Transcript order | `parent`/`children`, plus a same-agent filter in `Node.next` |
| Launch ownership | a child `AgentStart` hangs under the launching `ExecAction` |
| Dependency scope | handle arguments restricted to siblings (`delegation.resolve_args`) |
| Fork boundary | `clone_until` copies the whole tree, prunes after one node, and re-IDs everything (about 100 lines) |
| Persistence layout | tree-shaped documents plus per-agent directories |
| Run container | `root._index`, `root.usage`, `root.find_agent`, `_require_run_root` |

Because the tree is the only structure available, live bookkeeping also sits
on nodes: `in_flight`, `pending_handoffs`, `dependency_scheduled`, and
`_index`. Several features then work around the tree:

- `handoff.py` keeps a live mailbox, deduplicates deliveries by searching
  transcripts for `"[handoff path]"` strings, and rebuilds the mailbox on
  restore.
- `AppendChild.attach` rewrites `path` and `depth` across a whole subtree to
  rehome it.
- `delegation.py` sorts children by receipt, tracks `dependency_scheduled`,
  and scans every agent in `ready_dependents`.
- `restore.py` derives replay owners by walking up the tree.
- The shepherd forks into standalone graphs, replays each one by hand, and
  then rehomes them with `append_child`.

None of that is essential behavior. All of it is the cost of encoding many
relations in one tree.

## Records

```python
# rlmflow/graph/graph.py (Edge) and rlmflow/graph/nodes.py (Node)
NEXT, FROM, DEPENDS_ON = "next", "from", "depends_on"
STRUCTURAL = frozenset({NEXT, FROM, DEPENDS_ON})


@dataclass(kw_only=True)
class Node:
    type: ClassVar[str] = "node"
    id: str = field(default_factory=new_node_id)
    agent_id: str = ""            # owner; recipient for an undelivered UserQuery
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    seq: int = -1                 # assigned when the node is logged
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


@dataclass(frozen=True, kw_only=True)
class Edge:
    kind: str
    source: str
    target: str
    data: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_edge_id)
    seq: int = -1
    removed_seq: int | None = None

    @property
    def active(self) -> bool:
        return self.removed_seq is None


@dataclass(kw_only=True)
class AgentStart(Node):
    type: ClassVar[str] = "agent_start"
    args: dict[str, Any] = field(default_factory=dict)        # plain JSON only
    config: AgentConfig = field(default_factory=AgentConfig)  # no path/depth
    system_prompts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_id = self.agent_id or self.id
```

Nodes are plain records. They hold no reference to a graph, parent, children,
or live state.

The existing typed nodes keep their payloads: `UserQuery`, `RepairQuery`,
`FinalQuery`, `TruncationSummary`, `LLMOutput`, `ExecAction`, `AppendChild`,
`DelegationAction`, `TransitionDecision`, `ExecOutput`, `ErrorOutput`,
`ReplDead`, `VerificationFailure`, and `DoneOutput`.

They lose `root`, `parent`, `children`, `parent_agent`, `sub_agents`,
`frontier`, `in_flight`, `pending_handoffs`, `dependency_scheduled`, and
`_index`. `AgentConfig` loses `path` and `depth`, which become derived.

## Structural edge kinds

| Kind | Direction | Meaning |
|------|-----------|---------|
| `next` | earlier node → later node | History order. A branch in history is a fork. |
| `from` | message → sender node | Who sent this message. On an `AgentStart`, that is who launched the agent. |
| `depends_on` | consumer `AgentStart` → producer `AgentStart` | The consumer cannot start until the producer is terminal. If `data["as"]` is set, the producer's result is bound to that name. |

Any other `kind` is application data, such as `peer`, `supervises`,
`reviews`, or `member_of`. Flow does not interpret those kinds.

## `Graph`

```python
# rlmflow/graph/graph.py
class Graph:
    id: str

    # writes (history goes through ops)
    def add_edge(self, edge: Edge) -> Edge: ...           # application edge between existing nodes
    def remove_edge(self, edge: Edge | str) -> Edge: ...  # deactivate an application edge
    def merge(self, other: Graph) -> None: ...            # append another graph's log
    def as_of(self, seq: int) -> Graph: ...               # copy of the graph at a log position

    # raw lookups
    def node(self, id: str) -> Node: ...
    def nodes(self, type: type[Node] = Node, *, agent: str | None = None) -> list[Node]: ...
    def edges(self, *, kind: str | None = None, source: str | None = None,
              target: str | None = None, active: bool | None = True) -> list[Edge]: ...
    def items(self, after: int = -1) -> list[Node | Edge]: ...    # the log, in seq order

    # persistence
    def save(self, path: str | Path) -> Path: ...
    @classmethod
    def load(cls, path: str | Path) -> Graph: ...
```

Its state:

```python
self._nodes: dict[str, Node]
self._edges: dict[str, Edge]
self._out: dict[str, list[Edge]]     # node id -> outgoing edges
self._in: dict[str, list[Edge]]      # node id -> incoming edges
self._log: list[Node | Edge]         # index == seq; a removal re-logs the edge
self._lock: threading.RLock
```

Semantics:

- Each `ops` function holds the graph lock, checks its own preconditions
  against the committed graph (endpoints exist, frontier unchanged, `at`
  landed, no launch cycle, ...), then logs its items with consecutive `seq`
  values. A refused write logs nothing.
- `add_edge` accepts only non-structural kinds between existing nodes.
  `remove_edge` sets `removed_seq` and logs the updated edge. Nothing is
  deleted, and structural edges can't be removed.
- `merge(other)` refuses shared ids, then appends `other`'s log in order.
- `items(after=seq)` is the change stream. It is the only stream; there is
  no event bus.
- `load` indexes the saved log and runs `check_invariants(g)` once, so a
  loaded graph satisfies the same invariants as a live one.

### Invariants (`check_invariants`)

`check_invariants(g)` walks a whole graph and raises `InvariantError` at the
first broken rule; `load` runs it once. Live writes keep the same rules by
checking only what they touch.

1. Node and edge IDs are unique, every edge's endpoints exist, and every node
   is owned by a known agent (an `AgentStart` owns itself).
2. **`next`.** Every non-`AgentStart` node has exactly one incoming `next`
   edge, from the current last node of its own agent. The one exception is an
   undelivered `UserQuery`, which has no incoming `next` yet; delivery adds
   it exactly once. A node has at most one outgoing `next` to its own agent.
   Any number of outgoing `next` edges may point to other agents'
   `AgentStart` nodes; those are forks, and they leave only landed nodes.
   Each agent's history is one chain reachable from its `AgentStart`.
3. **`from`.** An `AgentStart` has at most one `from` edge, and the launcher
   tree is acyclic. A `UserQuery` may have several, because one handoff can
   batch several results.
4. **`depends_on`.** These edges connect distinct `AgentStart`s, are acyclic,
   and are added only before the consumer's first own step.
5. Structural edges are never removed.
6. A node may be annotated (metadata, tool receipts) only while it is its
   agent's frontier.

## `reads`: agent queries

All of these are pure functions of `Graph` lookups. Any caching is private.

```python
# rlmflow/graph/reads.py
def history(g: Graph, agent: AgentStart) -> list[Node]: ...
def frontier(g: Graph, agent: AgentStart) -> Node: ...
def terminal(g: Graph, agent: AgentStart) -> bool: ...
def inbox(g: Graph, agent: AgentStart) -> list[UserQuery]: ...
def ready(g: Graph) -> list[AgentStart]: ...
def launcher(g: Graph, agent: AgentStart) -> AgentStart | None: ...
def children(g: Graph, agent: AgentStart) -> list[AgentStart]: ...
def path(g: Graph, agent: AgentStart) -> str: ...
def depth(g: Graph, agent: AgentStart) -> int: ...
```

| Function | Definition |
|----------|------------|
| `history` | Start at the agent's last own node and follow incoming `next` edges back to an `AgentStart` that has none. For a fork, this walk continues through the base agent's prefix. |
| `frontier` | The last node of `history`. A fork's own `AgentStart` is transparent: an empty fork's frontier is its base node. |
| `terminal` | The frontier is a `DoneOutput`. |
| `inbox` | `UserQuery` nodes addressed to the agent with no incoming `next`, in `seq` order. |
| `ready` | Agents that are not terminal, or have a non-empty inbox, and whose active `depends_on` producers are all terminal. |
| `launcher` | The agent that owns the target of this agent's `from` edge. `None` means top-level. |
| `children` | Agents whose launcher is this agent. |
| `path` | The launcher's path plus `"." + name`. |
| `depth` | The length of the launcher chain. |

`path` and `depth` are derived, so attaching an agent never rewrites configs.
Anything else is a one-liner at the call site:

```python
agents    = g.nodes(AgentStart)
top_level = [a for a in agents if reads.launcher(g, a) is None]
usage     = sum((n.usage for n in g.nodes(LLMOutput)), LLMUsage())
peers     = g.edges(kind="peer", source=a.id)
```

## `ops`: derived writes

Each function is one locked write. The comments list every item it logs.

```python
# rlmflow/graph/ops.py
def append(g, agent, node, *, after=None) -> Node:
    # add(node, Edge(NEXT, last_own(agent), node))

def spawn(g, task, *, by=None, name=None, config=None,
          depends_on=(), inputs=None, **args) -> AgentStart:
    # add(start,
    #     Edge(FROM, start, by)                         if by,
    #     Edge(DEPENDS_ON, start, p)                    for p in depends_on,
    #     Edge(DEPENDS_ON, start, p, {"as": k})         for k, p in inputs)

def send(g, agent, content, *, sender=None, **metadata) -> UserQuery:
    # add(q := UserQuery(content, agent_id=agent.id),
    #     Edge(FROM, q, sender)                         if sender)

def deliver(g, agent) -> list[UserQuery]:
    # add(Edge(NEXT, last_own(agent), q) for q in inbox(agent))

def fork(g, at, *, name=None, by=None, **config) -> AgentStart:
    # add(start := AgentStart(config=copy(owner(at).config, name, **config)),
    #     Edge(NEXT, at, start),
    #     Edge(FROM, start, by)                         if by)

def attach(g, parent, *agents, names=None) -> AppendChild:
    # add(a := AppendChild(code=synthesized, receipts=agent ids),
    #     Edge(NEXT, last_own(parent), a),
    #     Edge(FROM, agent, a)                          for agent in agents)
```

Application topology needs no helper:

```python
g.add_edge(Edge(kind="peer", source=a.id, target=b.id))
g.remove_edge(edge)
```

### `append`

`after` is a compare-and-set guard: the write fails if
`frontier(g, agent) is not after`. Steps always pass the node they started
from. That one check replaces both `AgentBusyError` and the stale-frontier
error.

### `spawn`

With `by=None`, the agent is top-level. A graph holds any number of top-level
agents; there is no root.

### `send` and `deliver`

`send` writes a durable message without linking it into history, so it never
interrupts a running step. The message waits in `inbox`. An agent with a
non-empty inbox is ready, and Flow calls `deliver` before that agent's next
step. Host injection, handoff, gossip, and `ask_agent` all use `send`.

Host code that wants immediate placement on an idle agent calls `append`
with a `UserQuery` instead.

### `fork`

A fork writes two or three items. It copies nothing and remaps no IDs. The new
agent shares the immutable prefix through `at` and owns everything appended
after it.

- `at` may be any landed node except an action (`ExecAction`,
  `DelegationAction`, or `AppendChild`). To rerun an action, fork at the node
  before it; the fork then creates and runs a fresh action.
- Forking at a `DoneOutput` gives a terminal fork; `send` reopens it.
- Agents launched inside the shared prefix remain the original agent's
  children. The fork reads them through the handles that replay returns.
- Forks of forks work without special handling, because `history` follows
  `next` edges transitively.

### `attach`

The synthesized `AppendChild` code is ordinary REPL code:

```python
_handles = [launch_agent(task=...), ...]
print(await asyncio.gather(*_handles))
```

Its receipts resolve to the attached agents, so replay returns the same
handles. An attached agent must not already have a `from` edge. It may be
finished or still running; unfinished agents are ready and run normally.

To bring in a saved run from elsewhere, merge it and then attach it:

```python
other = Graph.load(path)
g.merge(other)
ops.attach(g, parent, *[a for a in other.nodes(AgentStart)
                        if reads.launcher(other, a) is None])
```

IDs are UUIDs, so they collide only when the same run is imported twice. Fork
first if you need a second copy.

## Flow

```python
class Flow:
    def __init__(self, llm, *, tools=None, runtime=None, ...): ...
    async def run_streaming(self, g: Graph, *, until="done") -> AsyncIterator[Node | Edge]: ...
    async def arun(self, g: Graph) -> Graph: ...
    def run(self, g: Graph) -> Graph: ...
```

### Contract

1. **Read.** Schedule `reads.ready(g)` minus the agents already in flight.
2. **Step.** Call `ops.deliver`, then run one step on `reads.frontier`
   through the existing transition policy: model call, code execution,
   delegation, final query, summary, or repair.
3. **Write.** Land the result with `ops.append(..., after=frontier)` and run
   reactions through `ops`.
4. **Stream.** Yield `g.items(after=cursor)`.

```python
async def run_streaming(self, g, *, until="done"):
    running, cursor = {}, len(g.items()) - 1
    while True:
        for agent in reads.ready(g):
            if agent.id not in running:
                ops.deliver(g, agent)
                running[agent.id] = asyncio.create_task(
                    self._step(g, agent, reads.frontier(g, agent)))
        if not running:
            return
        done = await first_completed(running)          # pops from running
        self._react(g, done)
        for item in g.items(after=cursor):
            cursor = item.seq
            yield item
            if stop(until, item, g):
                return

async def _step(self, g, agent, node):
    landed = await self.transitions.produce(self, g, node)
    try:
        ops.append(g, agent, landed, after=node)
    except StaleFrontier:
        pass                                           # a host write won; ready() reschedules
```

### Reactions

These are Flow policy. Each one is a few lines over `ops`.

| Reaction | Trigger | Write |
|----------|---------|-------|
| Handoff | A spawned agent lands `DoneOutput` | `ops.send(g, launcher, rendered, sender=done)`. It is skipped when the launcher is terminal and `resume_on_child_done=False`, when the child was consumed through `depends_on`, or when `done` already has an incoming `from` edge. |
| Budget finalization | LLM usage crosses `max_budget` | `ops.append(g, agent, FinalQuery())` once descendants have finished |
| Ask answer | A target lands `DoneOutput` after a query whose `from` edge points at a waiting sender action | Resolve that sender's waiting tool call |

Handoff deduplication uses an edge instead of a string marker. A child that is
reopened and finishes again is delivered once more.

### Live state

| State | Rebuilt from |
|-------|--------------|
| Model clients, delegation backends, tools, codecs, transitions | constructor arguments |
| REPLs keyed by `(g.id, agent.id)` | replaying the `ExecAction`s in `history` |
| In-flight steps | `reads.ready(g)` |
| Waiting tool calls (`await handle`, `ask_agent`) | the sender's unfinished `ExecAction`, which reruns on replay |

Flow owns no graph structure.

## Agent-facing tools

These run inside a step's REPL. Each one is a single `ops` call (or
`g.add_edge`/`g.remove_edge`), executed on the event loop that drives the graph. Its
receipt (the IDs it wrote) is saved on the current `ExecAction` by call
index. A replayed call returns the receipt instead of writing again, using
the existing saved-call mechanism.

```python
launch_agent(task, *, name=None, schema=None, verify=None, backend=None,
             model=None, context_policy=None, depends_on=None, **inputs) -> AgentHandle
    # ops.spawn(g, task, by=current_action, ...); handle inputs → depends_on with "as"

launch_subagent = launch_agent

fork_agent(handle=None, *, message=None, name=None) -> AgentHandle
    # ops.fork(g, at=latest non-action node of target, by=current_action)
    # then ops.send(g, fork, message); message is required when forking self

send_agent(target, message) -> str
    # ops.send(g, target, message, sender=current_action)

ask_agent(target, message, *, max_iters=None, schema=None) -> Any
    # send_agent, then wait for the target's next DoneOutput

connect_agent(target, kind, **data) -> str       # g.add_edge(Edge(kind, self, target, data))
disconnect_agent(edge_id) -> None                # g.remove_edge(edge_id)

await handle                                     # wait until reads.terminal(target)
AGENTS                                           # read-only directory built from reads
```

`fork_agent()` with no handle forks the caller at the `LLMOutput` that produced
the running action, so it requires `message`; without one, the fork would
rerun the same code block and fork again forever. This gives an agent
best-of-N from inside its REPL:

```python
tries = [fork_agent(message=f"Try approach {k}.") for k in "ABC"]
results = await asyncio.gather(*tries)
```

`ask_agent` rejects an `ephemeral` target, a backend that cannot continue a
session, and any wait that would close a cycle through `depends_on`, open
asks, and handle waits. Its `max_iters` and `schema` apply only to the turns
after that query.

## Agent lifecycle

```python
class ContextPolicy(StrEnum):
    EPHEMERAL = "ephemeral"        # drop live state when terminal; refuse follow-ups
    REHYDRATABLE = "rehydratable"  # drop live state; replay history on follow-up
    RESIDENT = "resident"          # keep live state; replay only after process loss
```

The policy is stored in `AgentStart.config`, so it is a graph fact; Flow
enforces it. The default is `resident`, which matches current behavior.

## Persistence

```text
run/
  graph.json     {"id": ..., "items": [...]}   # the log, in seq order
  blobs/         large argument values, content-addressed
```

Per-agent session files, if kept, are exports derived from `history`.

Old tree-format runs, including everything under `runs/` and `examples/_runs/`,
load through `persistence.upgrade_v1(path)`. That converter does three things:

- `parent_id` becomes a `next` edge, or a `from` edge for a child
  `AgentStart`;
- `$rlmflow_agent` arguments become `depends_on` edges with `as`;
- `Input` descriptions move onto the edge data.

## Rehydration

Python rehydration is required, and it stays a Flow operation:

1. Open an agent's REPL lazily, when its next step is an `ExecAction`.
2. Rerun every landed `ExecAction` in `history`, in order. For a fork, this
   includes the shared prefix. Saved receipts keep launches, forks, sends, and
   edges idempotent.
3. Continue from the frontier.

`reuse_worker` remains a runtime placement detail; its owner is the
launcher's REPL. External backends keep their session ID on
`DelegationAction`. A delivered message to an external agent produces a
follow-up `DelegationAction` that resumes that session.

## Coverage: every current feature

| Feature | Today | After |
|---------|-------|-------|
| Start a run | `start(task)` builds a root | `g.create_agent(task)`, which is `ops.spawn(g, task)` with no launcher |
| Model, exec, and delegation steps | producer, then `anchor.append` | producer, then `ops.append(..., after=)` |
| In-step nudges (`FinalQuery`, `TruncationSummary`, `TransitionDecision`) | `anchor.append` inside the step | `ops.append` from the same step |
| `launch_subagent` | `resolve_child`, then a tree child | `ops.spawn(by=action)` |
| Names, paths, depth, `max_depth` | stored, and rewritten on attach | `reads.path` and `reads.depth` |
| Handle arguments, `blocked`, `resolved_args` | `$rlmflow_agent` markers, sibling scope | `depends_on` edges with `as` |
| `Input(value, description=...)` | args marker | plain values unchanged; a handle's description moves to the edge data |
| `await handle`, `AGENTS`, `render_graph` | waiter plus a tree snapshot | `reads` plus `g.edges` |
| Handoff, `resume_on_child_done` | live mailbox and markers | reaction: `ops.send(..., sender=done)` |
| Host injection, reopening a finished agent | `frontier.append(UserQuery)` | `ops.send`, or `ops.append` when idle |
| Stale append, busy agent | frontier check plus `in_flight` | the `after=` guard |
| `node.fork()` | `clone_until`: copy and re-ID | `ops.fork`: two or three items |
| `append_child`, `AppendChild` | rehome a subtree and rewrite paths | `ops.attach`: one action plus `from` edges |
| Shepherd best-of-N | standalone forks, manual replay, `append_child` | `ops.fork` in one graph, then `ops.attach` |
| Save and load | tree document, agent directories, blobs | the log plus blobs |
| `flow.replay`, `restore=` | walk the tree, find worker owners | replay the `ExecAction`s in `history` |
| Stats, usage, errors | `root._index` | `g.nodes(...)` one-liners |
| Driving several roots | a driving set inside `run_streaming` | `reads.ready(g)` |
| `until=` boundaries | `predicate(node, root)` | `predicate(item, g)` |
| Verification, repair, semantic transitions | typed nodes | unchanged |
| External delegation | `DelegationAction` | unchanged; follow-ups arrive through the inbox |
| RAO credit and export | `sub_agents`, `config.depth`, `config.path` | `reads.children`, `depth`, `path` |
| Collaboration queue | external tool | unchanged |
| Views, TUI, checkpointer | tree walk | `g.items(after=)` plus edges |

## Formations

```python
from rlmflow.graph import Graph, Edge, ops, reads

g = Graph()

# Classic RLM
ops.spawn(g, "Solve it."); flow.run(g)

# Disposable workers (inside a REPL)
launch_agent(task="One question.", context_policy="ephemeral")

# Rootless: several top-level agents
a, b = ops.spawn(g, "Frontend."), ops.spawn(g, "Backend.")
c = ops.spawn(g, "Integrate.", depends_on=[a, b])

# Peers, and supervision that differs from launch provenance
g.add_edge(Edge(kind="peer", source=a.id, target=b.id))
g.add_edge(Edge(kind="peer", source=b.id, target=a.id))
g.add_edge(Edge(kind="supervises", source=c.id, target=a.id))

# Dataflow (inside a REPL)
src = launch_agent(task="Produce data.")
out = launch_agent(task="Summarize.", data=src)

# Best-of-N and merge (the shepherd)
branches = [ops.fork(g, rewind, name=f"branch{i}") for i in range(n)]
for b, plan in zip(branches, plans):
    ops.send(g, b, plan)
ops.attach(g, judge, *branches)   # the judge's next turn sees every result
flow.run(g)                       # the judge writes the merge as ordinary code
```

### Decentralized language models

A decentralized setup is top-level peers, each with its own model, connected
by application edges, communicating through `send_agent` and `ask_agent`. No
agent is a root or a router.

```python
peers = [
    ops.spawn(g, "Own shard {shard}; find contradictions with your peers.",
              name=f"peer{i}", config=AgentConfig(model=m, context_policy="resident"),
              shard=i)
    for i, m in enumerate(["qwen3-4b", "llama-8b", "claude-sonnet", "gpt-oss-20b"])
]
for a, b in zip(peers, peers[1:] + peers[:1]):
    g.add_edge(Edge(kind="peer", source=a.id, target=b.id))
async for item in flow.run_streaming(g):
    render(item)
```

Inside a peer's REPL:

```python
nbrs = [AGENTS.get(e.target) for e in AGENTS.edges(source=AGENTS.self, kind="peer")]
for n in nbrs:
    send_agent(n, f"My claims: {claims}")        # gossip without blocking
reply = ask_agent(nbrs[0], "Do you contradict claim 3?")
connect_agent(AGENTS.get("peer3"), "peer")       # emergent topology
checks = [launch_agent(task=f"Verify {c}", context_policy="ephemeral") for c in claims]
submit(await asyncio.gather(*checks))            # a local answer; messages can reopen it
```

- **Termination is quiescence.** `reads.ready(g)` is empty: every peer is
  terminal and every inbox is empty.
- **A global answer is optional.** Spawn a reader with `depends_on=peers`.
- **Shared world state** (repos, queues, environments) stays in tools.
- **Replay** reconstructs the whole swarm from the graph alone.

Merging, in every formation, is the consuming agent's next code block: select,
vote, splice, or launch a repair agent with `inputs={"a": c0, "b": c3}`. Live
REPL heaps, provider sessions, and environments are not merged directly; the
consuming agent reproduces them by running code.

## Implementation plan

This section is the plan as written before implementation. Line numbers and
paths refer to the code at that time; the package has since been flattened
(`records.py` into `graph.py` and `nodes.py`, `restore.py` into
`runtime/runtime.py`, `boundaries.py` and `execution.py` into
`engine/scheduler.py`, and `Graph.add`/`remove` into the explicit writes
above). See [file_structure.md](file_structure.md) for the current layout.

### File structure

```text
rlmflow/graph/
  records.py      NEW  ~80    Node, Edge, NEXT/FROM/DEPENDS_ON, ids
  graph.py        NEW  ~260   Graph: add, remove, node, nodes, edges, items, save, load, invariants
  reads.py        NEW  ~100   history, frontier, terminal, inbox, ready, launcher, children, path, depth
  ops.py          NEW  ~130   append, spawn, send, deliver, fork, attach
  nodes.py        1124 → ~620  typed payload nodes only (+ ~40 lines of phase-1 shims)
  persistence.py  764  → ~280  record codec, blobs, upgrade_v1 (save/load move into graph.py)
  config.py       197  → ~150  AgentConfig without path/depth; Input kept
  arguments.py    67   unchanged
rlmflow/engine/
  delegation.py   409  → ~170
  handoff.py      166  → ~75
  restore.py      125  → ~65
  boundaries.py   33   predicate signature (node, root) → (item, graph)
  execution.py    268  → ~240  TaskQueue keyed by agent id
  steps.py, transitions.py, completion.py, verification.py, semantic.py
                       take (graph, agent) instead of node.parent_agent
rlmflow/flow.py   679  → ~620
rlmflow/tools/agents.py     430 → ~410
rlmflow/tools/namespace.py  143 → ~135
```

The estimated core change is about 570 lines added and about 1,350 removed,
for a net of about −800 before counting tests.

### `rlmflow/graph/nodes.py` (1124 → ~620)

| Lines | Symbol | Fate |
|-------|--------|------|
| 78–83 | `AgentBusyError` | Deleted. Replaced by `StaleFrontier` in `graph.py`. |
| 182–190 | `RunStats` | Deleted, along with its public export. |
| 194–235 | `RunIndex` (`register`, `snapshot`) | Deleted. Its job moves to the private index inside `Graph`. |
| 248–253 | `Node.root`, `parent_agent`, `children`, `parent` | Deleted. Replaced by the `agent_id` field. |
| 254–256 | `Node.seq` as per-agent position | Becomes graph-global and assigned by `add`. |
| 292–351 | `Node.append` (busy, frontier, and duplicate checks; index registration) | Deleted. Becomes `ops.append` plus the `Graph` invariants. |
| 353–374 | `Node.append_child` | Deleted. Becomes `ops.attach`. |
| 377–396 | `next`, `prev` | Shims in phase 1; deleted in phase 2. |
| 398–409 | `walk` | Deleted. |
| 411–416 | `iter_backwards` | Deleted. Becomes `reads.history`. |
| 466–493 | `project` | Kept. Walks `reads.history(g, agent)`. |
| 495–500 | `subtree_usage` | Deleted. |
| 518–524 | `Node.fork` | Deleted. Becomes `ops.fork`. |
| 538–552 | `AgentStart.sub_agents`, `frontier`, `in_flight`, `pending_handoffs`, `dependency_scheduled`, `_index` | Deleted. |
| 554–563 | `AgentStart.__post_init__` | Shrinks to the message snapshot plus `agent_id = id`. |
| 566–576 | `terminal`, `supervisor`, `result` | Shims, then `reads`. |
| 579–619 | `blocked`, `argument_references`, `resolved_args` | Deleted. Replaced by `depends_on` edges; delegation binds the inputs. |
| 621–631 | `argument_descriptions`, `invocation_arguments` | Kept. Handle descriptions are read from edge data. |
| 633–647 | `leaves`, `transcript` | Deleted. Become `reads.ready` and `reads.history`. |
| 649–656 | `retrieved_agent_ids` | Deleted, along with `ExecAction.mark_agent_retrieved`, the `observe_agent` tool, and `repl_execution_order`. They were written but never read. |
| 658–700 | `_require_run_root`, `usage`, `stats`, `iter_agents`, `find_agent`, `errors`, `rebuild_index` | Deleted. Become `g.nodes(...)` one-liners. |
| 702–721 | `llm_turns`, `record_prompt`, `system_prompt_for`, `latest_system_prompt` | Kept. |
| 723–734 | `AgentStart.save` / `load` | Deleted. Become `Graph.save` / `Graph.load`. |
| 824–835 | Launch-index side effect in `ExecAction.record_tool_call` | Deleted. The receipt is simply saved. |
| 841–859 | `launch_agent_ids`, `launch_call_id_for` | Kept. The receipt key is renamed to `launch_agent`. |
| 889–934 | `AppendChild.child_agents`, `append`, `attach` (subtree rehome, path/depth rewrite) | Deleted. |
| 936–951 | `AppendChild._refresh_code` | Becomes `AppendChild.launching(agents, names)`. |
| 1070–1081 | `start()` | Becomes a shim returning `(graph, agent)`, then is deleted. |

Unchanged: `ChatMessage`, `message_snapshot`, ID helpers, `active_step`,
`record_metadata`, `running_step`, `TurnMode`, `annotate`, the
message-snapshot methods, `timing`, `to_record`, every typed payload class
from `TransitionDecision` through `DoneOutput`, and `build_agent`.

### `rlmflow/graph/persistence.py` (764 → ~280)

| Lines | Symbol | Fate |
|-------|--------|------|
| 167–179 | `to_document` | Deleted. `Graph.save` writes `items` instead. |
| 182–248 | `from_document` (parent-before-child order, one root, `order`/`agent_id` path checks) | Deleted. `Graph.load` replays the log through `add`. |
| 251–354 | `_validate_graph` | About 70 lines deleted: the argument-scope check, the dependency-cycle DFS, and launch uniqueness, which `Graph` invariants now cover. About 25 lines kept: strict-JSON validation of saved tool calls. |
| 357–389 | `_validate_arg_value` | The agent-reference branch is deleted, leaving a plain JSON check. |
| 392–414 | `save`, `load` | Move into `Graph`. |
| 417–512 | `clone_until` | Deleted entirely (96 lines). |
| 515–563 | `_agent_document`, `_summary_document`, `_write_agents` | Deleted. `agents/*/session.jsonl` and `latest.json` are write-only; nothing in `rlmflow` reads them. An opt-in exporter can replace them. |
| 644–648 | `_prune` | Deleted along with `_write_agents`. |
| 651–717 | `_new_node` | Kept, minus `parent_id` handling. |
| new | `upgrade_v1(path)` | About 60 lines; see Persistence. |

Kept: `to_record`, `_node_types`, `_ArgBlobStore`, the encode/resolve
helpers, `_config`, `_epoch`, and the JSON writers.

### `rlmflow/graph/config.py` (197 → ~150)

| Lines | Fate |
|-------|------|
| 14 `AGENT_REF_KEY`, 61–70 `agent_ref_id`, 73–90 `argument_references` | Deleted. Handles become edges. The proxy marker survives only inside `tools/agents.encode_agent_references`. |
| 113–115 `AgentConfig.path`, `depth` | Deleted. |
| 149–162 `child()` | Loses the `path` and `depth` lines. |
| 165–181 `can_spawn`, `at_leaf` | Take `(g, agent)` and use `reads.depth`. |
| new | `context_policy: ContextPolicy = RESIDENT`. |

### `rlmflow/engine/delegation.py` (409 → ~170)

| Lines | Symbol | Fate |
|-------|--------|------|
| 51–129 | `launch_tool` | Kept. Records receipts through `ops.spawn`. |
| 132–166 | `_attach` | Rewritten in about 15 lines around `ops.spawn(by=action, depends_on=..., inputs=...)`. |
| 198–255 | `resolve_args` | About 30 lines deleted: the sibling-scope check (228–234), duplicate-source tracking, and reference markers. About 20 lines remain to split handle inputs from JSON arguments. |
| 258–288 | `resolve_child` | Kept. The depth check uses `reads.depth`. |
| 291–315 | `submit_child`, `submit_leaf` (`dependency_scheduled` gating) | Deleted. Flow schedules from `ready()`. |
| 318–333 | `ready_dependents` | Deleted. |
| 336–370 | `new_child` | Kept, smaller. |
| 373–398 | `_sort_launched_children` | Deleted. `seq` already orders launches. |

### `rlmflow/engine/handoff.py` (166 → ~75)

| Lines | Fate |
|-------|------|
| 24–115: `Handoff`, `marker`, `supervisor`, `delivered`, `enqueue`, `dependency_routed`, `flush`, `apply`, `rebuild_mailboxes` | Deleted. Replaced by about 25 lines: `react(g, done)` calls `ops.send(g, launcher, render(...), sender=done)` and deduplicates by an existing `from` edge. |
| 118–157: `_render`, `_format_result`, `_result_body`, `_jsonish` | Kept. |

### `rlmflow/engine/restore.py` (125 → ~65)

| Lines | Fate |
|-------|------|
| 12–37 `_replay_root`, `_required_agents` | Rewritten to find `reuse_worker` owners through `reads.launcher`. |
| 40–82 `replay` | Same body, but iterates the `ExecAction`s in `reads.history(g, agent)`, which covers fork prefixes. |
| 85–100 `ensure_replayed` | Keyed by `(g.id, agent.id)` instead of `id(agent)`. |
| 103–122 `resume` | Drops the `dependency_scheduled` reset (119–121) and `rebuild_mailboxes`. |

### `rlmflow/flow.py` (679 → ~620)

| Lines | Fate |
|-------|------|
| 431–464 `_drive` | `anchor.append(landed)` and the `frontier.append` calls become `ops.append(..., after=node)`. Catches `StaleFrontier`. |
| 466–498 budget helpers | `root.usage` becomes a graph sum; `sub_agents` becomes `reads.children`. |
| 545–547 `replay(root)` | Becomes `replay(g, agent=None)`. |
| 549–617 `run_streaming` | About 69 lines shrink to about 40: the `ready()` loop with `deliver`, reactions, and `g.items(after=)`. Deletes the root driving set, `driving`, `root_error`, `handoff.apply`, and `ready_dependents`. |
| 619–626 `_submit` | Deleted. |
| 628–661 `arun`, `run`, `start` | Take a `Graph`. The task-string shortcut is kept. |

Unchanged: lines 92–423 (constructor, prompts, LLM clients, structured
queries, `llm_query`, transitions) and 502–541 (tools and backends).

### Other modules

| File | Change |
|------|--------|
| `engine/execution.py` 193–259 `TaskQueue` | `submit`, `join`, and `cancel` are keyed by agent id. `StepResult.is_agent_start` (189–190) is deleted along with the publish-on-spawn special case. |
| `engine/boundaries.py` 10–18 | Boundary predicates take `(item, graph)`. |
| `tools/agents.py` 348–416 `build_agent_directory` | Rebuilt from `reads` and `g.edges`. |
| `tools/agents.py` 153 `AgentDirectory.root_id` | Becomes `top_level_ids`. |
| `tools/agents.py` 237–311 `render_graph` | Renders a forest. Its dependency view reads `depends_on` edges. `AgentInfo` gains `edges` so agents can see peers. |
| `tools/namespace.py` 63–98 | Wait and observe tools look agents up with `g.node` instead of `find_agent`. |
| `tools/builtins.py` 172–181 | Renames the tool to `launch_agent` (keeping the `launch_subagent` alias) and adds `fork_agent`, `send_agent`, `ask_agent`, `connect_agent`, and `disconnect_agent`. |
| `rao/credit.py`, `rao/export.py`, `rao/rollout.py` | `sub_agents`, `config.path`, and `config.depth` become `reads.children`, `path`, and `depth`. The tree `root` becomes a graph plus a top-level agent. `rao/env.py`'s environment fork is a separate concept and is unchanged. |
| `view/*`, `consumers/*` | Walks over `.children` become edge reads. In `figure.py`, `append_child` references read `AppendChild` nodes. `tui.py`'s 19 uses of `.root` become `self.graph`. |
| `rlmflow/__init__.py` | Removes `RunStats`, the public `persistence` module, and `start` (replaced by a shim). Adds `Graph`, `Edge`, `reads`, `ops`, and `ContextPolicy`. Keeps `AppendChild`. |
| Unchanged | `llm.py`, `prompts/`, structured outputs and engines, `runtime/`, the REPL protocol, the `@tool` framework, `collaboration/`, and `backends/`. |

The transition registry stays as Flow policy and can be simplified
separately.

### Compatibility shims

The removed attributes are used widely:

- in the package: about 60 uses of `.parent_agent`, 45 of `.root`, 40 of
  `config.path`, and about 15 each of `.frontier` and `.sub_agents`;
- in tests: about 410 uses across 19 files;
- in examples: 46 files.

In phase 1, `Graph.add` sets a private `node._graph`, and these read-only
properties delegate to `reads`:

```python
node.parent_agent   agent.frontier   agent.result()   agent.terminal
agent.sub_agents    agent.supervisor node.next        node.prev
agent.config.path   agent.config.depth
```

Prompts, views, steps, and most test assertions then work unchanged. Only
write paths need migrating in phase 1. In phase 2, call sites move to `reads`
and the shims and `_graph` are deleted.

## Behaviors preserved

**Agent execution**

- The step loop: the same transition policy and the same node types.
- The turn protocol: `submit`, schema validation, and verifiers, with
  `VerificationFailure` followed by `RepairQuery`.
- Limits: `max_iters`, the run-wide `max_budget`, `FinalQuery`, and waiting
  for descendants before finalizing.
- History limits: `keep_n_messages` and `TruncationSummary`.
- Semantic transitions, including the durable `TransitionDecision` that replay
  reuses.
- The per-agent system-prompt table that maps `prompt_id` to text.
- `llm_query`, its cache, and its per-call annotations.
- External delegation through `DelegationAction`, with resume by `run_id`.

**Delegation**

- Launch options: names and the `child{n}` default, `max_depth`,
  `max_query_chars`, `schema=`, `verify=`, `Input(...)` descriptions, and
  per-child model, backend, and prompt profile.
- Handles: `await handle`, `handle.status`, `handle.result()`, and
  `asyncio.gather`.
- Handle-valued inputs still wait on and bind the producer's result. They now
  also work across launch scopes.
- Handoff: batched delivery, skipping children consumed as inputs, and
  `resume_on_child_done`.
- `AGENTS`: `get`, `get_parent`, `get_children`, `get_siblings`,
  `render_graph`, and the status values.
- `reuse_worker` shared worker sessions.

**Control and branching**

- Host injection reopens a terminal agent.
- Stale appends are rejected.
- Forking from any non-action node, with the original untouched and each fork
  continuing independently.
- Attaching prepared or loaded runs under a parent, using synthesized launch
  code that replays.
- Shepherd best-of-N: fork, run branches in parallel, attach them under the
  judge, and merge.

**Persistence and replay**

- Save and load, including the blob store and strict validation on load.
- Python rehydration that reruns code against saved tool results, in both
  `restore="replay"` and `restore="lazy"` modes.
- Driving several independent agents in one stream.
- All `until=` boundaries: `next`, `idle`, `done`, `finished`, `error`, and
  callables.

**Everything else**

- RAO rollouts, credit, advantages, and export.
- The collaboration queue, all runtimes (local, Docker, Modal, subprocess),
  and the REPL protocol.

## Behaviors that change

| Today | After |
|-------|-------|
| `AgentBusyError` when the host writes during a step | The host write wins; the step result is discarded and the agent rescheduled. Use `send` to queue a message instead. |
| Forking at an `ExecAction` clears its receipts and reruns it | Rejected. Fork at the `LLMOutput` before it. |
| Fork gives everything fresh IDs (`clone_until`) | The fork shares IDs and history; only the new agent's nodes are new. |
| A run has exactly one root | Any number of top-level agents. |
| Handle arguments are limited to siblings | Any agent in the graph. |
| Handoff is deduplicated by a `[handoff path]` string | Deduplicated by a `from` edge, so a reopened child delivers again. |
| `node.seq` is a per-agent position | Global log order. The per-agent position is the index in `history`. |
| Every save writes `agents/*/session.jsonl` and `latest.json` | Removed, or produced by an opt-in exporter. |
| `RunStats` and `root.stats` | One-liners over `g.nodes`. |

## Migration

### API mapping

| Today | After |
|-------|-------|
| `root = start(task)` | `g = Graph(); agent = g.create_agent(task)` |
| `flow.run(root)` | `flow.run(g)` |
| `root.save(p)` / `AgentStart.load(p)` | `g.save(p)` / `Graph.load(p)` |
| `root.iter_agents()` / `find_agent(id)` | `g.nodes(AgentStart)` / `g.node(id)` |
| `agent.sub_agents` / `agent.supervisor` | `reads.children(g, a)` / `reads.launcher(g, a)` |
| `agent.frontier` / `agent.result()` | `reads.frontier(g, a)` / `reads.frontier(g, a).result` |
| `node.append(x)` | `ops.append(g, agent, x, after=node)` |
| `frontier.append(UserQuery(...))` | `ops.send(g, agent, ...)` |
| `node.fork()` | `ops.fork(g, node)` |
| `node.append_child(sub)` | `ops.attach(g, agent, sub)` |
| handle argument | `depends_on` edge with `as` |

`flow.run("task string")` can remain a shortcut that builds a one-agent graph
and returns that agent's result.

### Tests and examples

**Tests.** Roughly 410 call sites need migrating. The heaviest files are
`test_flow.py` (171), `test_handoff.py` (34), `test_graph.py` (34),
`test_graph_scalability.py` (30), `test_concurrency.py` (21), and
`test_rao_textcraft.py` (19).

- With the phase-1 shims, read-only assertions keep working.
- Write calls must be rewritten: `start(...)`, `.frontier.append`, `.fork()`,
  `append_child`, and `.save`/`.load`.
- Some tests need a full rewrite: the handoff-marker tests, the `clone_until`
  re-ID tests, the sibling-scope rejection tests, and the `AgentBusyError`
  tests.

**Examples.** 46 files are affected. The largest changes:

- `examples/graph/01–07`;
- `shepherd/shepherd.py` (lines 327–339 and 581);
- `shepherd/recovery.py` (lines 93–101; the manual replay disappears);
- `control/injection/inject_variants.py:116`.

### Build order

1. Add `records.py`, `graph.py`, and `reads.py` with their own unit tests.
   Nothing else changes.
2. Add `ops.py` and the `nodes.py` shims, so the old attribute names read
   through the graph.
3. Move persistence to the new format and add `upgrade_v1`.
4. Move the engine: delegation, handoff, restore, `flow.run_streaming`, and
   `TaskQueue`.
5. Add the tools: `launch_agent`, `fork_agent`, `send_agent`, `ask_agent`,
   `connect_agent`, and `AGENTS`.
6. Port RAO, views, and consumers.
7. Delete dead code: `RunIndex`, `clone_until`, the mailbox, and the scope
   checks.
8. Port examples and tests, then remove the shims.

Steps 1–3 can land without breaking anything. Step 4 switches behavior over.

## Acceptance tests

1. Existing RLM examples run on `Graph`.
2. A refused write (any `ops` function, `add_edge`, `merge`) logs nothing.
3. A graph with several top-level agents and no root runs to completion.
4. `reads.ready(g)` matches Flow's scheduling with no Flow present.
5. A `depends_on` edge written by `spawn` blocks before the first step;
   `depends_on` cycles are rejected and `peer` cycles are allowed.
6. A named input binds the producer's result in the consumer's REPL.
7. `append(after=)` rejects a stale write; a host write during a step wins,
   and Flow reschedules the agent.
8. A message sent to a running agent waits in its inbox and is delivered
   before that agent's next step, without interrupting the running one.
9. `ops.fork` writes two or three items; `history(fork)` is the base prefix
   plus the fork's own nodes; the original is unchanged.
10. A fork at an `LLMOutput` runs a fresh action; a fork at an action is
    rejected; a fork at a `DoneOutput` is terminal until it receives a
    message.
11. A fork's REPL is rebuilt by replaying the shared prefix, and its receipts
    resolve to the original children.
12. `ops.attach` gives a parent finished and unfinished agents; the parent's
    next step sees every result; replay restores the same handles.
13. The ported shepherd forks, runs, attaches, and merges in one graph.
14. Importing a saved run and attaching it reproduces
    `07_attach_subtree.py`.
15. Handoff deduplicates by `from` edge, survives save and load, and
    redelivers after a child is reopened and finishes again.
16. Replayed tool calls (`launch_agent`, `fork_agent`, `send_agent`,
    `ask_agent`, `connect_agent`) perform no new writes.
17. Save and load round-trips the log, including removed edges, and load
    enforces the invariants.
18. A decentralized ring of peers on different models gossips, rewires,
    and reaches quiescence; replay reproduces it.
19. `run_streaming` yields every item in `seq` order.
20. RAO credit derives ownership and depth from `from` edges.

## Deferred

- Several Flows writing one graph. The `after=` guard and the durable inbox
  keep that feasible later; today, one driving Flow is a documented
  expectation.
- Channels, broadcast, and group chat. `send` plus application edges covers
  the current use cases.
- Generic merge policy. `Graph.merge` imports another log; combining
  results is code the consuming agent writes.

## Public surface

```python
# rlmflow.graph
Graph(add_edge, remove_edge, merge, as_of, node, nodes, edges, items, save, load)
Node, Edge, NEXT, FROM, DEPENDS_ON, typed node classes, ContextPolicy, check_invariants
reads: history, frontier, terminal, inbox, ready, launcher, children, path, depth
ops:   append, spawn, send, deliver, fork, attach

# rlmflow
Flow.run_streaming(g), Flow.arun(g), Flow.run(g)

# agent tools
launch_agent, launch_subagent, fork_agent, send_agent, ask_agent,
connect_agent, disconnect_agent, AGENTS, await handle
```
