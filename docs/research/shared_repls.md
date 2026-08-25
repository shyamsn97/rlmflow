# Shared REPLs: Per-Agent State in a Shared Namespace

> **Status:** Prototyped and validated end to end against the current engine,
> local and remote. Not implemented.
>
> **Scope:** letting several agents share one `Repl` — one Python namespace, one
> process, shared variables — while every piece of per-agent state stays
> correct, with no serialization and no loss of concurrency. Every gap
> identified during design is solved and tested below; nothing is left open.

## Summary

Today `Runtime` owns one `Repl` per agent, keyed by `agent.id`. Agents share a
process, so `sys.modules` and injected objects are shared, but not variable
bindings: an object one agent builds is unreachable by name to any other.

Sharing a `Repl` naively corrupts runs, which is why the obvious reading is
"agents can't share REPLs." That conclusion is wrong. The corruption comes from
per-agent state being stored in places that assume one agent — instance
attributes on the `Repl`, and fixed names in the namespace — and **a namespace
entry does not have to hold a value. It can hold a dispatcher.**

The rule the whole design follows:

> Anything persistent is a dictionary keyed by agent id. Anything per-step is a
> context that dies with the step. The namespace holds only what is genuinely
> shared, plus dispatchers that resolve through the two.

The engine already contains the mechanism. `LocalRepl` scopes stdout with a
`ContextVar`, and every agent step is its own asyncio task, so a `ContextVar` is
already per-agent-step for free:

```python
_stdout_buf: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "minimal_rflow_stdout", default=None
)
```

The same rule carries to remote sandboxes, where the second insight is that
co-tenants share a *namespace*, not a *connection*. Virtual channels
demultiplexed over the one stdio pipe already there give each agent its own
channel into one sandbox, for one additive protocol field and no new transport.

## State Taxonomy

Three tiers, and knowing which tier a thing belongs to answers every design
question about it.

| Tier | Lives in | Lifetime | Contents |
|---|---|---|---|
| Shared | `repl.namespace` | the REPL | variables, functions, classes, imports, `flow.inject` objects |
| Per-agent | `repl.envs[agent_id]` | that agent | `ENV`, `RLMFLOW_*` metadata |
| Per-step | `RunCtx` in a `ContextVar` | one action | stdout buffer, `errored`, `done_result`, current `inputs`, current `action` |

`RunCtx` is the pointer, not the store. It holds a reference into
`repl.envs[agent_id]` rather than a copy, so `ENV` writes persist across an
agent's turns while still being invisible to co-tenants. Nothing needs garbage
collection except the per-agent dict, which is dropped when the agent's REPL
registration is.

## Why The Obvious Approaches Fail

### Naive sharing corrupts silently

Two siblings on one `Repl`, each given different `inputs`, each printing who it
thinks it is:

```text
one thinks : 'one' | stdout: 'I think I am one'
two thinks : 'one' | stdout: ''
```

Agent two reported agent one's answer and lost its own output. Three mechanisms
cause it, needing three different fixes:

1. **Instance-staged run state.** `LocalRepl.capture()` does
   `self._buf = io.StringIO()` and `run()` opens with `self.done_result = None`.
   The second agent to enter replaces the buffer, and `outcome(self.output)`
   reads whichever was set last. `self.errored` is shared the same way.
2. **Seeded names.** `Flow.execute_action` calls `repl.seed(...)` before every
   action, writing `namespace["done"]`, `namespace["launch_subagents"]`, and
   `namespace["INPUTS"]` bound to the current agent. The second agent's seed
   overwrites the first's, so the first then calls the wrong `done` closure and
   reads the wrong `INPUTS`.
3. **Shared `ENV`.** `namespace["ENV"] = self.env`, one dict per `Repl`.

### Per-agent dicts alone fix only the first

Keying `done_result`, `_buf`, and `errored` by agent id addresses (1) and does
nothing for (2): the bug there is not where an answer is stored, it is that the
*name* resolved to the wrong object before the answer existed. Dispatchers are
what fix (2), and the dicts are what they dispatch into.

### A lock works, then deadlocks

Serializing seed+execute per `Repl` does fix all three:

```text
shared Repl, per-Repl lock
   one thinks : 'one' | stdout: 'I think I am one'
   two thinks : 'two' | stdout: 'I think I am two'
```

But it gives up all concurrency between co-tenants, and it deadlocks on
delegation. An agent calling `launch_subagents` is suspended *inside* its `run()`
at the `await`, holding the lock, until its children finish. A co-tenant blocks
for that whole span; a co-tenant that is itself a descendant never runs, so the
parent never returns. Delegation is the normal case here, so a lock is not
viable. The context design has no such problem — proven in
[case B](#delegation) below.

## The Design

```python
@dataclass
class RunCtx:
    agent_id: str
    agent_path: str
    inputs: dict[str, str]
    env: dict[str, Any]        # -> repl.envs[agent_id], NOT a copy
    action: Node
    output_schema: dict[str, Any] | None = None
    buf: io.StringIO = field(default_factory=io.StringIO)
    errored: bool = False
    done_result: Any = None
    done_set: bool = False


_ctx: ContextVar[RunCtx | None] = ContextVar("rlmflow_run", default=None)
```

`Flow.execute_action` sets it around the step. Contextvars propagate down the await
chain within one task, so the context reaches `Runtime.execute` → `Repl.run` →
`exec` without threading an argument through any signature.

The framework names become singletons resolving through it:

```python
class AgentInputs(Mapping):
    def __getitem__(self, key):
        return current().inputs[key]
    def __iter__(self):
        return iter(current().inputs)
    def __len__(self):
        return len(current().inputs)


class AgentEnv(MutableMapping):
    """Per-agent and persistent: writes land in repl.envs[agent_id]."""
    def __getitem__(self, key):
        return current().env[key]
    def __setitem__(self, key, value):
        current().env[key] = value
    def __delitem__(self, key):
        del current().env[key]
    def __iter__(self):
        return iter(current().env)
    def __len__(self):
        return len(current().env)


def done(answer: object) -> None:
    ctx = current()
    schema = ctx.output_schema
    ctx.done_result = (
        str(answer)
        if schema is None
        else parse_structured_output(
            answer if isinstance(answer, str) else json.dumps(answer), schema
        )
    )
    ctx.done_set = True
    raise DoneSignal


async def launch_subagents(specs):
    return await flow.launch_tool(current().action)(specs)
```

`seed()` stops stamping per-agent values into a namespace others are reading,
and `run()` reads its outcome from the context instead of from `self`:

```python
async def run(self, code: str) -> ReplRun:
    ctx = current()
    with self.capture_into(ctx):
        ...exec against self.namespace...
    output = ctx.buf.getvalue().strip()
    if ctx.done_set:
        return ReplRun(output=output, status=ReplStatus.DONE, answer=ctx.done_result)
    return ReplRun(
        output=output, status=ReplStatus.ERROR if ctx.errored else ReplStatus.OK
    )
```

## Validation

Every claim below is output from a prototype run against the unmodified engine.

### Concurrency and isolation

Two siblings sharing one `Repl`, overlapping sleeps, a shared `bag` list:

```text
one result : 'one'   stdout: "I think I am one | bag so far: ['one', 'two']"
two result : 'two'   stdout: "I think I am two | bag so far: ['two']"
shared bag : ['one', 'two']
```

Each read its own `INPUTS`, produced its own stdout, returned its own `done()`
answer — while both mutated the same list and each saw the other's writes at the
moment it looked. No lock, full concurrency.

### Error isolation

```text
one frontier : done_output ''
two frontier : done_output 'two is fine'
two result   : 'two ok' (unaffected by one's raise)
```

### Delegation

The case a lock cannot survive:

```text
one children : ['gc']
grandchild   : 'grandchild ok'
one result   : 'one done'
two result   : 'two done'
shared bag   : ["one saw ['grandchild ok']", 'two ran']
```

`one` launched a grandchild and suspended at the `await` while `two` ran to
completion in the same namespace. Both writes landed, in order.

### Per-agent `ENV` and `RLMFLOW_*` metadata

`ENV` is per-agent and survives across that agent's turns, while the namespace
stays shared:

```text
one outputs : ['one: x = one | ENV[owner] = one | id = root.one | depth = 1',
               'one turn2: ENV[owner] still one | id root.one']
two outputs : ['two: x = one | ENV[owner] = two | id = root.two | depth = 1',
               'two turn2: ENV[owner] still two | id root.two']
distinct ENVs : 2 envs on one shared REPL
agent ids     : ['root.one', 'root.two']
```

Note `x = one` in both: user variables collide, by design (see
[What stays shared](#what-stays-shared)). `ENV`, `INPUTS`, `done`, and the
agent ids did not.

The metadata fix is to stop treating "REPL exists" as "agent is registered".
`Runtime.repl_for` currently calls `update_env(agent_process_env(...))` only
inside its `if repl is None` branch, so a co-tenant inherits the first agent's
identity. Seed per agent instead:

```python
env = repl.env_for(agent.id)
if not env:  # first turn for this agent on this REPL
    env.update(
        agent_process_env(
            agent_id=agent.config.path,
            depth=agent.config.depth,
            parent_agent_id=agent.config.path.rpartition(".")[0] or None,
            max_depth=agent.config.max_depth,
        )
    )
```

### Refcounted close

`close_repl` must not strand co-tenants:

```python
def close_repl(self, value):
    key = value if isinstance(value, str) else _agent(value).id
    repl = self.repls.get(key)
    if repl is None:
        return
    tenants = sum(1 for other in self.repls.values() if other is repl)
    self.repls.pop(key, None)
    self.envs.pop(key, None)
    if tenants <= 1:
        with suppress(Exception):
            repl.close()
```

```text
before close  : 2 tenants
after closing one, two still has its REPL: True
two's bag intact: ['one', 'two']
after closing both, registered: False
```

This matters more for remote than local: `LocalRepl.close()` is a no-op, but
`RemoteRepl.close()` kills the connection, so unrefcounted close leaves the
surviving co-tenant holding a dead stub.

## Remote

The remote case looked like a protocol rework and is not one. The mistake was
assuming co-tenants must share a *connection* because they share a *namespace*.
They don't — sharing is a property of the sandbox process, and the connection is
just how one agent talks to it. **Give each co-tenant its own channel into one
sandbox process.**

The channel need not be a separate pipe. Virtual channels demultiplexed over the
single existing stdio pipe are enough, and are cheaper than a new socket
transport because every backend already provides one duplex byte stream. That is
the design below; a real socket per agent is the fallback if the sandbox ever
needs channels the host does not mint.

### Three things called "stdio", only one of them the problem

Worth separating, because the word covers three unrelated layers here:

| Layer | What it is | Status |
|---|---|---|
| Agent `print()` | captured into a per-step buffer via `_stdout_buf` | already per-agent, works |
| The wire | JSON-line protocol on the sandbox's fd 1 | the deadlock lives here |
| Raw fd 1 writes | subprocesses, `os.write`, C extensions | corrupts the wire today |

The deadlock is not a capture problem. Agent output is already routed per agent
by a `ContextVar` and always has been. The problem is *who reads the pipe* — a
demultiplexing question, not a redirection one. The third layer is a separate
live bug that co-tenancy makes much worse, handled further down.

### What breaks on one shared connection

`ReplServer.serve_stdio` is a `readline` → `await handle` → `write` loop, and
`make_proxy` reads its `ProxyResponse` from *the same stdin*. So a proxy round
trip mid-run consumes whatever arrives next. With two co-tenants that is the
other agent's `RunRequest`, and the failure is exactly as predicted:

```text
KeyError: 'code'          # serve loop parsed a ProxyResponse as a request
one shared channel    : DEADLOCK (parent never returned)
```

The parent parks in a proxied `launch_subagents`; the response meant for it is
swallowed by the serve loop; nobody ever returns. This is the lock model
rejected earlier for local, with the same delegation deadlock.

Give each agent a channel and the same test passes:

```text
one channel per agent : parent='parent-ok' child='child-ok'
```

The rule is that **exactly one thread may read a byte stream.** Today two do:
the serve loop and every parked `make_proxy`. Fix that by making the reader the
sole owner of stdin and routing what it reads:

```python
def reader(self):
    for line in sys.stdin:
        raw = json.loads(line)
        if "cmd" in raw:
            self.route_request(raw.get("repl_id", "root"), raw)   # -> that agent's inbox
        else:
            self.pending.pop(raw["id"]).put(raw)                  # -> the parked proxy
```

Each agent then gets a worker thread running **the existing sequential loop
verbatim**, reading its inbox queue instead of stdin. `make_proxy` blocks on a
queue keyed by the call id it just minted, so it can only ever receive its own
reply. Both routing keys already exist in the protocol: `repl_id` on every
`BaseRequest`, and matching `id` on `ProxyCall`/`ProxyResponse`.

Validated against a real subprocess over a real pipe, with a parent delegating
to a co-tenant child in the same sandbox:

```text
a.output : child returned child-ok | bag ['parent', 'child'] | my env {...'me': 'A'}
a.answer : parent-ok | a.env: {'RLMFLOW_AGENT_ID': 'agent-a', 'me': 'A'}
b.answer : child-ok | b.env: {'RLMFLOW_AGENT_ID': 'agent-b', 'me': 'B'}
VERDICT  : works
```

Shared `bag`, separate `ENV`, delegation across co-tenants, no deadlock, no new
transport.

### The sandbox needs the same fix as the host

One namespace with concurrent co-tenants is the same problem sandbox-side, and
the prototype reproduced it before fixing it. Binding per-agent values per
request — `ENV`, agent identity — is clobbered by whichever co-tenant starts a
block while another is parked in a proxy call:

```text
A: output="agent-b host saw from A (for agent-a) ..."   # A printed B's identity
   env={'RLMFLOW_AGENT_ID': 'agent-a'}                  # A's ENV['me'] went to B
```

The fix is the one already established for local: bind dispatchers once at
startup, never values per request, and let them resolve through a per-channel
`Scope` held in a `ContextVar`. Since each channel is its own thread, each gets
its own context for free.

```text
A: output="agent-a host saw from A (for agent-a) | bag ['a', 'b']"
   answer='answer-A' env={'RLMFLOW_AGENT_ID': 'agent-a', 'me': 'A'}
B: output="agent-b | bag ['a', 'b']"
   answer='answer-B' env={'RLMFLOW_AGENT_ID': 'agent-b', 'me': 'B'}

shared bag     : ['a', 'b']
wall clock     : 0.30s (A alone blocks 0.3s on its proxy)
VERDICT: works | ran concurrently: True
```

Separate stdout, separate `done` answers, separate persistent `ENV`, shared
`bag` — and genuine concurrency, since the pair finish in the time A spends
blocked alone. The server's `RunRequest` handler currently reads
`self.repl.errored` and `self.repl.env`, which move into the scope with
everything else.

### The one field the protocol is missing

Multiplexing costs exactly one additive field. Responses route by an id the host
minted, so those are free. A `ProxyCall` is different: the sandbox mints its id,
so the host has never seen it and cannot tell whose proxy it is. `ProxyCall`
needs `repl_id: str = "root"` for the host to bind the right agent's context
before invoking. `ProxyResponse` does not — the sandbox routes it by the id it
minted.

Defaulted and additive, so old peers in either direction behave exactly as they
do today; no version bump.

### The host must demultiplex too

Symmetrically, one connection shared by several `RemoteRepl`s needs a send lock
and a single reader routing by id — the fix listed below as an independent bug is
a prerequisite here, not an optional extra. One extra requirement the local case
does not have: an inbound `ProxyCall` must be serviced **off** the reader thread,
because servicing it may itself send another agent's request on the same
connection. That is precisely the delegation path, and handling it inline
deadlocks the reader.

### One genuine difference from local

Local co-tenants are asyncio tasks on one loop: cooperatively scheduled, so a
co-tenant can only interleave at an `await`. Remote co-tenants under
thread-per-channel are preemptively scheduled. Both are safe for the state this
design owns, since each scope is touched by one thread, but user code that
mutates a shared structure non-atomically is exposed remotely in a way it is not
locally. Worth saying in the co-tenancy prompt.

### Get the wire off fd 1

Separate from co-tenancy, and the most urgent item in this document. `ReplServer`
takes `self._out = protocol_out or sys.stdout` and relies on grabbing the real
object before `LocalRepl` swaps in its `_Stdout` shim. That shim only intercepts
Python-level writes. Anything below it goes straight onto the wire.

Measured against the real `SubprocessRuntime`, not a prototype:

```text
1 before   : ok   | before state I built up
2 subproc  : dead | REPL execution failed: ValidationError: Invalid JSON ...
3 after    : error| NameError: name 'x' is not defined
4 os.system: dead | REPL execution failed: ValidationError: Invalid JSON ...
6 captured : ok   | got CAPTURED           # capture_output=True is safe
```

`Runtime.execute` catches the parse failure, discards the REPL and opens a fresh
one, so this is a contained outcome rather than a crashed run. What it costs is
still severe:

- **The agent loses its entire Python state.** `x` is gone at step 3.
- **The error is unactionable.** The agent is told about a pydantic
  `ValidationError`, not that its subprocess wrote to stdout, so nothing stops
  it doing the same thing again on the next step.
- **The trigger is ordinary.** `subprocess.run([...])` without
  `capture_output`, `os.system`, and anything shelling out to `git`, `pytest`,
  or `npm` all do it. Every process-isolated runtime is affected; `LocalRuntime`
  is not, since it has no wire.

Under co-tenancy the blast radius widens from one agent's state to the whole
group's shared namespace.

The fix is to stop defending fd 1 and vacate it, at server startup before
anything else runs:

```python
wire = os.fdopen(os.dup(1), "w")   # the real pipe, now unreachable by name
cap_r, cap_w = os.pipe()
os.dup2(cap_w, 1)                  # fd 1 is now a capture pipe
```

A drain thread reads `cap_r`. Now no write to fd 1 by any mechanism can reach
the protocol, and output that is currently lost becomes recoverable:

```text
os.write(1) : survived, answer='x', fd1=['RAW BYTES ON THE WIRE']
subprocess  : survived, answer='y', fd1=[..., 'SUBPROC ON THE WIRE']
```

Subprocess output is captured rather than corrupting the run — strictly better
than today even for a single tenant.

One honest limit: fd 1 is per-process, not per-thread, so the drain thread
cannot attribute bytes to a co-tenant. With one tenant they belong to the
running agent. With co-tenants, raw fd-1 output can only go to a shared sandbox
log unless agent code cooperates by spawning through a scope-aware wrapper.
Python-level `print` is unaffected, since it never reaches fd 1.

### Nobody drains stderr

Worse than the above, and unrelated to sharing. `PopenConnection` opens
`stderr=sp.PIPE` and reads it only in `_exited_error`, after the process is
gone. Nothing drains it during a run, so an agent that writes past the ~64K pipe
buffer blocks forever inside the sandbox.

There is no escape hatch at the default settings. `RemoteRepl.run` waits via
`asyncio.to_thread`, which is not cancellable, so even an outer
`asyncio.wait_for` does not recover — the thread stays blocked and the host
process will not exit.

```text
repl_timeout=None : HUNG (uncancellable; process will not exit)
repl_timeout=5    : dead | TimeoutError: ... did not respond within 5s
```

Two fixes, and the first is available today with no code change:

- **Set `repl_timeout`.** It converts a permanent wedge into a clean `DEAD`
  outcome. Defaulting it to a finite value is a one-line change and strictly
  better than `None`.
- **Drain stderr continuously** into a bounded buffer on a reader thread, and
  surface it with the run output. That removes the wedge rather than timing out
  after it happens, and makes sandbox diagnostics visible instead of only
  readable post-mortem.

### An independent bug, now a prerequisite

Two concurrent calls on a single `RemoteRepl` corrupt the wire at byte level,
because neither `send` nor `recv` is serialized.

```text
1. A: status=error output='<ValidationError: Invalid JSON: key must be a string
      ... input_value='{id":"run""k:re"upt:s is...RLMFLOW_REPLAY":"0"}}\n'>'
1. B: status=error output='<ValidationError: Invalid JSON: trailing characters
      ... input_value='"-2,o"tu,otu""thi A",ore...TH":"2","_ROOT":"1","\n'>'
```

It is reachable today by any concurrent use of one connection, and multiplexing
co-tenants over one pipe requires it. A send lock plus a serialized reader with
request-id matching fixes it:

```python
def call(self, msg):
    with self._send_lock:
        self.connection.send(msg)
    while True:
        with self._recv_lock:
            if msg.id in self._stash:
                return self._check(self._stash.pop(msg.id))
            resp = self.connection.recv()
            if isinstance(resp, ProxyCall):
                self._handle_proxy(resp)
                continue
            if resp.id == msg.id:
                return self._check(resp)
            self._stash[resp.id] = resp
```

```text
current RemoteRepl.call    req-1 got 'result for req-2'   correct: False
with id matching           req-1 got 'result for req-1'   correct: True
```

## What Stays Shared

By design, and the reason to do this at all:

```text
shared        Python variables, functions, classes, imports, open handles
shared        anything flow.inject binds
per-agent     ENV, RLMFLOW_* metadata
per-step      INPUTS view, done, launch_subagents, stdout, errored, done_result
```

User variable names collide, and that is not a bug to fix. Two agents assigning
`x` share one `x`; that is what sharing a namespace means. Co-tenant agents
should be told so in their prompt, and conventions — name-spaced variables, or a
single agreed shared object — are the user's problem exactly as they would be
for two threads sharing globals.

## Scope

The staged plan — including the transport bugs above, which are not caused by
sharing and should ship before it — lives in
[`repl_plan.md`](repl_plan.md), with a defect register, file-level changes, ship
gates, and sequencing. In brief: stage 0 transport correctness, stage 1 local
co-tenancy, stage 2 remote co-tenancy, stage 3 frontier safety.

## Risks

**Debuggability.** A traceback from a shared namespace does not say which agent
wrote the offending binding. `RunCtx` already carries `agent_path`; put it in
error output.

**Prompting.** Agents must be told they share a namespace with named peers or
they will assume `x` is theirs. Most likely to decide whether this is usable in
practice.

**Blast radius.** One agent can rebind a name another depends on, or exhaust
memory for every co-tenant. Co-tenancy must be opt-in per group, never a
default.

**Cancellation.** A cancelled agent leaves half-built bindings in the shared
namespace. Fine for a scratch variable, not for a half-updated shared structure.

## Test Plan

1. Co-tenants read their own `INPUTS` under concurrent, overlapping execution.
2. Co-tenants get their own stdout; nothing crosses.
3. Co-tenants get their own `done()` result, including with differing
   `output_schema`.
4. A raise in one co-tenant leaves the other's `errored` false.
5. A co-tenant may delegate and suspend while another co-tenant runs.
6. A co-tenant may delegate to a child that is itself a co-tenant, no deadlock.
7. Variables and imports are genuinely shared.
8. `ENV` is per-agent and persists across that agent's turns.
9. `RLMFLOW_AGENT_ID` and friends are correct for every co-tenant, not just the
   one that opened the REPL.
10. `close_repl` on one co-tenant leaves the others working; the last one closes.
11. Cancelling one co-tenant does not disturb another's run state.
12. Replay of a graph whose agents shared a REPL rebuilds the same bindings.
13. Remote: two channels into one sandbox keep separate `ENV` and shared globals.
14. Remote: co-tenants run concurrently — two blocks overlap in wall clock.
15. Remote: a proxied `done` on one channel resolves against that channel's
    agent, while another channel is mid-run.
16. Remote: a proxy round trip does not consume another agent's request.
17. Remote: a parent parked in a proxied `launch_subagents` is freed by a child
    co-tenant in the same sandbox.
18. Remote: closing one channel leaves the others and the namespace working.
19. `RemoteRepl` concurrent calls return their own responses and the wire stays
    well-formed JSON, not merely correctly routed.
20. Remote: `os.write(1, ...)`, `os.system('echo ...')`, and an uncaptured
    `subprocess.run` leave the REPL alive with its variables intact, and their
    output is captured rather than lost.
21. Remote: writing several hundred KB to stderr in one block completes instead
    of wedging the sandbox.
22. Remote: a `ProxyCall` is serviced in the scope named by its `repl_id`, not
    the reading thread's, while another agent is mid-run.
23. Non-shared runs behave identically to today, local and remote.

## Recommendation

> **Outcome.** Stage 0 and the two standalone guards shipped. Stage 1 was built,
> passed its tests, and was then reverted: sharing cannot be added without moving
> per-agent state behind an indirection that every single-tenant run also pays,
> which is too much core change for an opt-in feature. The reasoning, and the
> narrower approach to try if it comes back, are in
> [`repl_plan.md`](repl_plan.md#why-sharing-is-shelved). The rest of this section
> is the original recommendation, kept because the analysis still holds.

Land stage 0 first, regardless of whether shared REPLs ever ship — it is the
most valuable thing in this document and the least speculative. An agent that
writes enough to stderr wedges its run permanently and uncancellably at the
default settings; an agent that shells out to anything that prints destroys its
sandbox and every variable in it, and is told only that some JSON failed to
parse. Neither has anything to do with sharing REPLs; both were found looking
for it.

Then stage 1, shipping local co-tenancy on its own. The mechanism is already in
the codebase, the prototype validated every case including delegation and
cross-turn `ENV` persistence, and the result is strictly more capable than the
alternatives: unlike per-agent dicts alone it fixes name binding, and unlike a
lock it keeps concurrency and survives delegation.

Then stage 2, which is smaller than the first analysis suggested. Co-tenants
need their own *channel*, not their own *connection*, and virtual channels
demultiplexed over the existing stdio pipe cost one defaulted protocol field and
a routing reader — no new transport in any backend. The sequential serve loop is
reused verbatim per channel, and the sandbox reuses the context design stage 1
already builds.

Stage 3, the frontier guard, is independent of all of it and can go whenever.
