# Implementation Plan: REPL Correctness and Shared REPLs

> **Covers** every defect found while designing shared REPLs, plus the sharing
> work itself. Companion to `shared_repls.md` (design and validation) and
> `persistent_agents.md` (agent addressing).
>
> **Ordering rule:** nothing that shares a REPL lands before the transport it
> would share is correct. Stage 0 is shippable on its own and should go first
> whether or not sharing is ever built.
>
> **Status:** stage 0, stage 3 and D14 are implemented. Stages 1 and 2 — sharing
> itself — were built once, worked, and were reverted; see
> [Why sharing is shelved](#why-sharing-is-shelved) before picking them up again.

## Defect Register

Everything known to be broken today, independent of whether sharing ships.

| # | Defect | Impact | Fix | Stage | Status |
|---|---|---|---|---|---|
| D1 | Nobody drains the sandbox's `stderr` pipe during a run | Permanent, uncancellable hang past ~64K | B | 0 | done |
| D2 | `repl_timeout` defaults to `None` | Removes the only escape from D1 | backstop | 0 | done |
| D3 | The wire lives on fd 1; subprocesses inherit it | Sandbox destroyed, all agent state lost | B | 0 | done |
| D4 | Parse failure surfaces as a raw `ValidationError` | Agent gets no actionable signal and repeats it | backstop | 0 | done |
| D5 | `RemoteRepl.call` never checks `resp.id` | A desync silently returns another request's response | B | 0 | done |
| D6 | Concurrent calls on one `RemoteRepl` shred the wire | Latent today; blocks stage 2 | B | 0 | done |
| D7 | Per-run state on the `Repl` instance | Blocks local sharing | A | 1 | shelved |
| D8 | `seed()` stamps per-agent values into the namespace | Blocks local sharing | A | 1 | shelved |
| D9 | `close_repl` is unconditional | One co-tenant closing kills the rest | A | 1 | shelved |
| D10 | `env.py` and the `Repl` ABC document per-REPL isolation | Docs contradict the new contract | A | 1 | shelved |
| D11 | The sandbox serve loop and `make_proxy` both read stdin | Deadlock under co-tenancy | B | 2 | shelved |
| D12 | `ProxyCall` carries no scope | Host cannot attribute an inbound proxy | A | 2 | shelved |
| D13 | Framework appends to an agent with a step in flight | `ValueError` escapes and kills the run | guard | 3 | done |
| D14 | `_CWD_LOCK` is held across the whole run | Deadlocks co-tenant threads whenever a working directory is set | A | 1 | done |

Stage 0, D13 and D14 are implemented, with tests in
`tests/test_repl_transport.py`. Everything marked shelved is the sharing feature
itself, which the transport work does not depend on.

D1–D6 are live bugs in shipped code and have nothing to do with sharing.

## Why Sharing Is Shelved

Stage 1 was implemented in full and it worked: co-tenants kept their own
`INPUTS`, `ENV`, stdout, errors and `done` results while genuinely sharing
variables and imports, verified by fifteen tests including concurrent co-tenants.
It was reverted anyway, for a reason worth recording.

Sharing is not additive. Everything per-agent today lives *on the REPL* —
`repl.env` is that agent's env, `repl.done_result` is that agent's answer,
`repl._buf` is that agent's stdout — or gets stamped into the namespace before
each step, which is safe only because there is one tenant. Making room for a
second tenant means moving all of it behind a resolve-at-use indirection, and
that indirection is then on the path of *every* run, including the overwhelming
majority that share nothing. The cost landed as: `repl.env` becomes ambiguous
about which tenant it means, `INPUTS` stops being a dict and becomes a mapping
view, `done_result` leaves the object the ABC says holds it, `Runtime.repls`
stops meaning one REPL per agent, and `close_repl` becomes refcounted. A large
change to core semantics, bought for an opt-in feature nothing yet asks for.

If it is picked up again, the thing to try first is confining it: a `SharedRepl`
and `SharedRuntime` that carry the context mechanism themselves, leaving the
single-tenant classes exactly as they are today. That trades some duplication for
a default path that provably cannot regress, and it makes the feature's cost
visible in one place instead of spread across the runtime.

## The Whole Fix, In Two Mechanisms

There are not fourteen fixes here. There are two mistakes, each made in several
places, and one unrelated guard. Build the two mechanisms and every row above
falls out.

### Mechanism A — resolve per-agent state at use, never bind it at setup

Every one of these bugs is the same shape: something that belongs to *an agent*
is stored on something shared by *all* agents, so whoever touches it last wins.

The rule: **persistent per-agent state lives in a dict keyed by agent id,
transient per-step state lives in a `RunCtx` in a `ContextVar`, and anything the
namespace exposes is a dispatcher that resolves through them.** A namespace
entry does not have to hold a value; it can hold an object that looks up the
caller.

Its corollary, for things the OS makes process-global and no context can split:
**refcount, never mutex.** Count the agents that want the resource, act on the
first in and the last out, and hold a lock only around the counter.

| Fixes | Because |
|---|---|
| D7 | `errored`, `done_result`, the stdout buffer move onto the context |
| D8 | `seed` binds dispatchers once instead of stamping values per request |
| D9 | close is refcounted per registration — last agent out closes |
| D10 | the docs describe the mechanism instead of contradicting it |
| D12 | the scope travels on the message, so the host resolves the same way |
| D14 | the cwd becomes a refcounted scope, not a mutex held across a run |

D9 and D14 are literally the same fix applied to two different shared
resources.

### Mechanism B — exactly one owner per byte stream, and route by id

Every remaining bug is the same shape too: a stream with more than one reader,
more than one writer, or nobody at all.

The rule: **every byte stream gets one dedicated owner that does nothing but
read or write it. Owners never do work; they route messages to whoever is
waiting, by id. Nothing else ever touches the stream.**

| Fixes | Because |
|---|---|
| D1 | stderr is a stream with no owner; give it one and it cannot fill |
| D3 | fd 1 has two writers; `dup` the wire to a private fd so it has one |
| D5 | routing by id is how an owner demultiplexes, so the check is inherent |
| D6 | one writer behind a lock, one reader, so bytes cannot interleave |
| D11 | the serve loop stops competing with `make_proxy` for stdin |

D2 (a finite `repl_timeout`) and D4 (an error naming the real cause) are not
part of the mechanism. They are the backstop for when something outside our
control violates it anyway, which is why they are cheap and worth having.

### The remainder

D13, the frontier guard, belongs to neither and is a few lines on its own.

**So the work is: one context mechanism, one stream-ownership mechanism, one
guard.** The stages below are just the order in which they get applied — B to
the transport first because it is broken today, then A locally, then both
together to make the sandbox multi-tenant.

## What Is And Is Not Agent-Scoped

Every piece of framework state is keyed by agent, either in
`envs[agent_id]` (persistent) or in the `RunCtx` for that agent's step
(transient): `INPUTS`, `ENV`, the `RLMFLOW_*` metadata, stdout, `errored`,
`done` and its result, `output_schema`, and `launch_subagents`. Remotely the
same holds per channel, keyed by `repl_id`, including the `ProxyCall` scope so
the host binds the right agent when servicing a callback.

Three things are not, and it is worth being exact about which is a choice and
which is a constraint.

**Shared on purpose.** The Python namespace — variables, functions, classes,
imports, and anything `flow.inject` binds — plus `sys.modules`. That is the
feature. Two agents assigning `x` share one `x`.

**Process-global, cannot be scoped.** The current working directory,
`os.environ`, `sys.path`, and signal handlers belong to the process, not to a
thread or a task. Remotely, raw writes to fd 1 are in this category too:
Python-level `print` is scoped through the `ContextVar`, but a subprocess's
output cannot be attributed to a co-tenant.

**The working directory is a hard blocker for stage 2 (D14).**
`LocalRepl.capture` holds a module-level `_CWD_LOCK` across the entire run and
`chdir`s into `self.working_directory`. It is an `RLock`, so:

- **Locally it does nothing.** Agents are asyncio tasks on one thread, which
  re-acquires the lock immediately. It provides no mutual exclusion between
  concurrent local agents.
- **Across threads it deadlocks.** Under stage 2 co-tenants are threads. A
  parent parked in a proxied `launch_subagents` holds the lock for the whole
  time it waits, and its child cannot enter `capture` to run:

```text
   child starting
   parent unblocked (child finished in time: False)
   child done after 6.0s
VERDICT: DEADLOCK - child cannot run until the parent gives up
```

The child only ran because the test gave up after six seconds. In the real
system the parent waits for its child indefinitely. With no working directory
configured both finish immediately, so this fires exactly when Subprocess and
Docker runtimes are given a workdir — the normal case.

**The fix is to make the scope refcounted rather than mutually exclusive.** The
lock exists to protect a save/restore of a process-global value, but co-tenants
all share one directory, so there is nothing to arbitrate. First agent in
`chdir`s, last one out restores, and the mutex is held only around the counter,
never across the run:

```python
with _guard:
    if _count == 0:
        _previous, _active = os.getcwd(), target
        os.chdir(target)
    elif os.path.realpath(target) != os.path.realpath(_active):
        raise WorkingDirectoryConflict(...)
    _count += 1
try:
    yield
finally:
    with _guard:
        _count -= 1
        if _count == 0:
            os.chdir(_previous)
```

```text
   child starting
   child done after 0.0s
   parent unblocked (child finished in time: True)
total 0.0s   |   cwd restored: True
```

Two mismatched targets cannot both be live, since `chdir` is process-global.
Today the `RLock` serializes them across threads and silently interleaves them
on one thread; the refcount raises `WorkingDirectoryConflict` instead, which is
the only honest option. For the sandbox specifically there is an even simpler
variant — `chdir` once at server startup and never again, since that process
exists only to serve agents — but the refcount covers `LocalRuntime` too, where
the host's directory must be restored.

Genuinely per-agent working directories remain out of reach for `chdir` in a
shared process. If they are ever required, agent code must resolve paths against
a per-agent root handed to it.

---

## Stage 0 — Transport Correctness
### *Mechanism B applied to the host and the pipes*

Ships alone. No API change, no behaviour change except that things which
currently break stop breaking. **Gate for every later stage.**

### 0.1 Drain stderr (D1, D2)

`rlmflow/runtime/connections.py`, `PopenConnection`:

- Start a daemon reader thread on `proc.stderr` at spawn, appending into a
  bounded `collections.deque(maxlen=...)`. The pipe must never fill.
- Surface the tail in `_exited_error`, and on a `DEAD` outcome so sandbox
  diagnostics stop being post-mortem-only.
- Change `repl_timeout` to default to a finite value rather than `None`. Keep
  `None` accepted as an explicit opt-out.

Why both: draining removes the wedge, the timeout is the backstop for any other
way the sandbox can stop answering. Today neither exists, and
`RemoteRepl.run` waits inside `asyncio.to_thread`, which is not cancellable — so
an outer `wait_for` cannot save it and the host process will not exit.

### 0.2 Move the wire off fd 1 (D3)

`rlmflow/runtime/repl_server.py`, at startup before anything else runs:

```python
wire = os.fdopen(os.dup(1), "w")   # protocol, now unreachable by name
cap_r, cap_w = os.pipe()
os.dup2(cap_w, 1)                  # fd 1 is a capture pipe
os.close(cap_w)
```

- `self._out = wire`.
- Reassign `sys.stdout` so `LocalRepl`'s `_Stdout` shim wraps the capture pipe,
  never the wire.
- Drain `cap_r` on a thread. Single-tenant: attribute to the running agent's
  buffer, so uncaptured subprocess output becomes visible agent output instead
  of being lost. Co-tenant (stage 2): a shared sandbox log, since fd 1 is
  per-process and cannot be attributed.

### 0.3 Make desync loud, then impossible (D4, D5, D6)

`rlmflow/runtime/repl_client.py`:

- `call()` checks `resp.id == msg.id` and raises a distinct
  `ProtocolDesyncError` otherwise. Cheap, and converts a silent wrong answer
  into a visible failure. Do this even though 0.2 removes the main cause.
- Add the send lock, single serialized reader, and per-id stash from
  `shared_repls.md`. Required by stage 2; independently correct now.
- When a message fails to parse, raise an error naming the likely cause
  (non-protocol bytes on stdout) rather than passing a bare pydantic
  `ValidationError` to the agent.

### Stage 0 tests

1. Several hundred KB to `stderr` in one block completes rather than wedging.
2. A sandbox that stops answering yields `DEAD` within `repl_timeout`, and the
   host process exits.
3. `os.write(1, ...)`, `os.system('echo ...')`, and an uncaptured
   `subprocess.run` all leave the REPL alive with its variables intact.
4. That output is captured and attributed to the running agent.
5. `capture_output=True` still works unchanged.
6. A forced out-of-order response raises `ProtocolDesyncError`.
7. Two concurrent calls on one `RemoteRepl` each get their own response, and
   every line on the wire is well-formed JSON.
8. Existing runtime tests pass untouched.

---

## Stage 1 — Local Co-Tenancy
### *Mechanism A applied in-process*

The design and its validation are in `shared_repls.md`. The rule: persistent
state is a dict keyed by agent id, per-step state is a `ContextVar`, and the
namespace holds only what is genuinely shared plus dispatchers.

### 1.1 `rlmflow/runtime/repl.py` (D7, D8, D10)

- Add `RunCtx` and its `ContextVar`, with `agent_id`, `agent_path`, `inputs`,
  `env` (a reference into `envs[agent_id]`, never a copy), `action`,
  `output_schema`, `buf`, `errored`, `done_result`.
- `LocalRepl`: move `_buf`, `errored`, `done_result` off the instance into the
  context. Add `envs: dict[str, dict]` and `env_for(agent_id)`.
- `capture()`, `run()`, `read()` and `Repl.outcome()` all read the context.
  `outcome` currently branches on `self.done_result` and must not.
- `seed()` binds dispatchers once instead of stamping per-agent values.
- Update the `Repl` ABC docstrings, which currently promise `namespace`, `env`,
  `done_result` and `errored` are instance state.

### 1.2 `rlmflow/tools/` (D8)

`AgentInputs` (`Mapping`) and `AgentEnv` (`MutableMapping`) resolving through
the current context.

### 1.3 `rlmflow/flow.py`

- `exec_step` establishes the context for the action it is about to run.
- `build_tools` returns the shared dispatchers rather than per-node closures, so
  `done` and `launch_subagents` resolve the calling agent from the context.

### 1.4 `rlmflow/runtime/runtime.py` (D9)

- Permit one `Repl` to back several agents in `repls`.
- Refcount `close_repl`: the last registration closes, earlier ones detach and
  drop that agent's `envs` entry. Note `Runtime.execute` calls `close_repl` on
  any failure, so under co-tenancy a single agent's dead REPL must not silently
  take out its co-tenants' state — decide explicitly whether a transport failure
  tears down the whole group (it must, for a remote sandbox) and make that
  visible to each affected agent.
- Seed `agent_process_env` per agent through `env_for`, not once per REPL.

### 1.5 Opt-in surface

Off by default, in the style of `use_agent_tree`. A co-tenancy group is named
explicitly; nothing shares implicitly.

### 1.6 Refcount the working directory (D14)

Replace `_CWD_LOCK` with the refcounted scope above: enter and exit only, the
mutex never held across a run, and `WorkingDirectoryConflict` when two live
scopes disagree. Prerequisite for stage 2, where the current lock deadlocks
co-tenant threads. Add a test that a co-tenant runs while another is parked with
a working directory configured.

### 1.7 Documentation of scope (D10)

- `rlmflow/runtime/env.py`'s module docstring promises these values are
  "per-REPL... isolated across concurrent local agents". Reword to per-agent.
- `LocalRepl.__init__`'s comment on `self.env` makes the same claim
  ("Because it is per-REPL it stays isolated across concurrent local agents").
- State plainly that the working directory is process-global and shared by
  co-tenants, rather than letting the lock imply otherwise.

### Stage 1 tests

Per `shared_repls.md`: own `INPUTS`, own stdout, own `done` result including
differing `output_schema`, error isolation, delegation while a co-tenant runs,
delegation to a co-tenant child without deadlock, genuinely shared variables and
imports, per-agent `ENV` persisting across turns, correct `RLMFLOW_*` for every
co-tenant, refcounted close, cancellation isolation, replay fidelity, and
non-shared runs behaving exactly as today.

---

## Stage 2 — Remote Co-Tenancy
### *A and B together, in the sandbox*

Virtual channels demultiplexed over the existing stdio pipe. Requires stage 0
(the host demux) and stage 1 (the context design, reused sandbox-side).

### 2.1 `protocol.py` (D12)

Add `repl_id: str = "root"` to `ProxyCall`. Additive and defaulted, so old peers
in either direction are unaffected. `ProxyResponse` needs nothing — the sandbox
routes it by the id it minted.

### 2.2 `repl_server.py` (D11)

- One reader owns stdin. Requests route to a per-`repl_id` inbox queue;
  `ProxyResponse`s route to the parked proxy by call id.
- One worker thread per scope runs **the existing sequential loop verbatim**
  against its inbox and the shared `LocalRepl`.
- Per-scope state replaces the instance's `errored`, `env`, `done_result`; the
  `RunRequest` handler reads those off the scope.
- Dispatchers bound once at startup. Anything bound per request is clobbered by
  a co-tenant, which the prototype demonstrated before it was fixed.
- Serialize writes.

### 2.3 `repl_client.py`

- Several `RemoteRepl`s over one connection; stamp `repl_id` on requests.
- Service an inbound `ProxyCall` **off** the reader thread, in the scope its
  `repl_id` names. Handling it inline deadlocks, because servicing a delegation
  proxy sends another agent's request on the same connection.

### 2.4 `runtime.py`

Co-tenants get a `RemoteRepl` on a new virtual channel to the same sandbox
rather than a new sandbox.

### Stage 2 tests

Two scopes keeping separate `ENV` and shared globals; co-tenants overlapping in
wall clock; a proxied `done` resolving against its own scope while another
channel is mid-run; a proxy round trip not consuming another agent's request; a
parent parked in a proxied `launch_subagents` freed by a co-tenant child; one
channel closing without disturbing the others; a `ProxyCall` serviced in the
scope named by its `repl_id` rather than the reading thread's.

### Fallback

A real socket per agent, if channels ever need to originate in the sandbox. It
trades 2.1 and the routing for a new connection abstraction in every backend, so
it is more work, not less.

---

## Stage 3 — Frontier Safety
### *Neither mechanism; a standalone guard*

Independent of sharing, and a genuine corruption bug rather than a missing
feature.

### 3.1 Reject appends to a busy agent (D13)

`Node.append` raises `ValueError: <id> is not the frontier of <id>` when the
framework appends to an agent whose step is in flight. Today that escapes
`run_streaming` and kills the run; for a sub-agent it is caught but yields a
bogus `DoneOutput`.

Add an explicit guard that rejects framework appends to an agent with an
in-flight step, with an error naming the agent and the in-flight step, and make
the driver treat it as a handled condition rather than a crash. This turns
silent corruption into a loud, attributable error.

Agent addressing and resumption — `launch_subagents` resuming a finished child
by name, extending `max_iters` on resume, and scoping `run_streaming` to a
sub-agent — are a feature rather than a defect and stay tracked in
`persistent_agents.md`.

---

## Sequencing

```text
Stage 0  transport correctness    ships alone, no dependencies, do first
Stage 3  frontier guard           independent, small, any time
Stage 1  local co-tenancy         depends on nothing but is the design core
Stage 2  remote co-tenancy        depends on stage 0 and stage 1
```

Stage 0 and stage 3 are bug fixes and should not wait on a decision about
sharing. Stage 1 is independently useful and shippable without stage 2. Stage 2
should not start until stage 0 is merged, because it multiplexes the connection
whose framing stage 0 makes safe.

## Ship Gates

- **Stage 0** — the whole existing runtime suite passes, plus its eight tests
  above. No agent-visible API change.
- **Stage 1** — co-tenancy off by default, and a run with it off is byte-for-byte
  what it is today.
- **Stage 2** — the delegation-across-co-tenants case passes against a real
  subprocess, not only in-process.

## Risks

**Debuggability.** A traceback from a shared namespace does not say which agent
wrote the offending binding. `RunCtx` carries `agent_path`; put it in error
output.

**Prompting.** Co-tenants must be told they share a namespace with named peers.
Most likely to decide whether this is usable in practice.

**Blast radius.** One agent can rebind a name another depends on, or exhaust
memory for the group. Opt-in per group, never a default.

**Scheduling differs by runtime.** Local co-tenants are asyncio tasks on one
loop and interleave only at an `await`. Remote co-tenants are threads and are
preemptively scheduled. State this design owns is safe either way, but user code
mutating a shared structure non-atomically is exposed remotely and not locally.

**Cancellation.** A cancelled agent leaves half-built bindings in the shared
namespace. Fine for a scratch variable, not for a half-updated shared structure.
