# Runtimes

A `Runtime` owns lightweight Python worker processes, agent placement, and
cleanup. Every shipped backend uses the same worker and typed JSON-line protocol;
only provisioning differs.

The core installation needs `cloudpickle` for value transfer. `ModalRuntime`
additionally requires `rlmflow[modal]`.

## Shipped runtimes

- `LocalRuntime(working_directory=...)` — a worker subprocess in the current
  Python environment. This is the default.
- `SubprocessRuntime(python=..., env=...)` — a worker using a selected Python
  executable and environment.
- `DockerRuntime(image, ...)` — a worker attached through `docker run -i`.
- `ModalRuntime(...)` — a worker attached through Modal Sandbox stdin/stdout.

Agent Python always crosses a process boundary. This provides fault and state
isolation, but local subprocesses are not security sandboxes.

## Isolated and shared agents

Each root and child gets an isolated worker by default. Pass `reuse_repl=True`
to place a child in its caller's existing worker:

```python
analyst = await launch_subagent(
    "Continue from the live objects prepared above.",
    name="analyst",
    reuse_repl=True,
)
result = await analyst.wait_for_result()
```

Shared agents have separate transcripts, model turns, `INPUTS`, `ENV`,
completion, and output attribution. They use the same interpreter and globals,
so imports, mutable objects, caches, open resources, and object identity are
shared immediately. Worker termination affects every tenant in that worker.

The worker can execute multiple shared tenants concurrently. Per-execution
context routes `INPUTS`, `ENV`, output, completion, and host-tool calls back to
the correct agent. User code can use ordinary top-level `await` and
`asyncio.gather(...)`.

## Working directory and tools

```python
flow = rlmflow.Flow(
    client,
    runtime=rlmflow.LocalRuntime(working_directory="./project"),
    tools=rlmflow.FILE_TOOLS,
)
```

The working directory belongs to the worker. A shared child necessarily inherits
its caller's directory.

Tools marked `@tool(proxy=True)` execute on the host through the worker's private
protocol stream. Other serializable values and functions are copied into the
worker with `cloudpickle`.

## Host and worker state

Agent code runs in a worker process, so the host and the agent share no heap.
Three rules follow, and a tool that ignores them either fails at injection with a
pickling error or quietly updates a copy:

- A function that has to run on the host — because it mutates host state, drives
  the graph, or closes over something unpicklable such as `Flow` — must be marked
  `@tool(proxy=True)`. Everything else is copied into the worker by value.
- State the host reads back travels through `ENV`, which is per-agent and
  round-trips with every execution: whatever agent code writes there is readable
  afterwards with `flow.runtime.get_env_var(agent, name)`.
- `flow.runtime.get_var(agent, name)` copies an object out of the worker, so it
  reads state and never mutates it. A class defined outside an installed package
  needs `cloudpickle.register_pickle_by_value(module)` before injection; without
  it the class crosses by reference and arrives in the worker as a missing
  import.

`examples/shepherd` follows all three: its recovery tools are proxies, the game
publishes each reading the host needs to `ENV`, and the host copies the finished
game out only to write the trace files.

## Stdio and protocol safety

Before user code starts, the worker duplicates stdin and stdout for its private
protocol and makes those descriptors non-inheritable. User stdin is redirected
to `/dev/null`; native writes and subprocess output are redirected to stderr.
Normal Python output is captured per execution. User code therefore cannot read
or corrupt protocol frames.

## Timeouts

User execution is unbounded by default, allowing cells to wait for long-running
subagents and tools. Pass `execution_timeout=<seconds>` to cap one execution:

```python
runtime = rlmflow.LocalRuntime(execution_timeout=3600)
```

A timeout terminates the entire worker because Python cannot safely kill an
arbitrary running thread. `repl_timeout` only bounds startup and control
requests; it does not limit user code.

## Restore behavior

Saved graphs remain the durable state. `restore="replay"` reconstructs unfinished
namespaces before scheduling the run. `restore="lazy"` waits until an agent is
about to execute code, then replays the recorded executions needed by that
worker. `reuse_repl=True` is live sharing, not namespace cloning.

## Docker

```bash
docker build -t rlmflow:local .
```

```python
runtime = rlmflow.DockerRuntime(
    "rlmflow:local",
    mounts={"./project": "/workspace"},
    workdir="/workspace",
    network="none",
    cpus=1.0,
    memory="512m",
)
```

The protocol uses stdin/stdout, so no ports or container-to-host network route
are required.

## Modal

```bash
pip install rlmflow[modal]
```

```python
runtime = rlmflow.ModalRuntime(remote_workdir="/workspace")
```

Modal uses the same worker protocol over Sandbox stdin/stdout; no tunnel or
sandbox-side bridge is involved.

## Custom provisioning

A custom backend only needs a `ReplConnection` with `start`, `send`, `recv`, and
`close`, then wraps it in `WorkerSession` and `WorkerRepl`. Execution, host-tool
RPC, serialization, tenant routing, and sharing semantics stay common across
backends.
