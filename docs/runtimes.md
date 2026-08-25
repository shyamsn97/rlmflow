# Runtimes

A `Runtime` owns lightweight Python worker processes, agent placement, and cleanup. Every shipped backend uses the same worker and typed JSON-line protocol; only provisioning differs.

The core installation needs `cloudpickle` for value transfer. `ModalRuntime` additionally requires `rlmflow[modal]`.

## Shipped runtimes

- `LocalRuntime(working_directory=...)` — a worker subprocess in the current Python environment. This is the default.
- `SubprocessRuntime(python=..., env=...)` — a worker using a selected Python executable and environment.
- `DockerRuntime(image, ...)` — a worker attached through `docker run -i`.
- `ModalRuntime(...)` — a worker attached through Modal Sandbox stdin/stdout.

Agent Python always crosses a process boundary. This provides fault and state isolation, but local subprocesses are not security sandboxes.

## Isolated and shared agents

Each root and child gets an isolated worker by default. Pass `reuse_repl=True` to place a child in its caller's existing worker:

```python
analyst = await launch_subagent(
    "Continue from the live objects prepared above.",
    model="default",
    name="analyst",
    reuse_repl=True,
)
result = await analyst.wait_for_result()
```

Shared agents have separate transcripts, model turns, `INPUTS`, `ENV`, completion, and output attribution. They use the same interpreter and globals, so imports, mutable objects, caches, open resources, and object identity are shared immediately. Worker termination affects every tenant in that worker.

The worker can execute multiple shared tenants concurrently. Per-execution context routes `INPUTS`, `ENV`, output, completion, and host-tool calls back to the correct agent. User code can use ordinary top-level `await` and `asyncio.gather(...)`.

A worker outlives the agent that answered in it: finishing does not close a REPL, so a fan-out holds one worker per child until you close them. Close a run's REPLs with `close_repls=True` on `run`/`arun`/`run_streaming`, one with `flow.runtime.close_repl(agent)`, or everything the Flow owns with `await flow.aclose()` — see [Control](control.md#cleanup).

## Working directory and tools

```python
flow = rlmflow.Flow(
    client,
    runtime=rlmflow.LocalRuntime(working_directory="./project"),
    tools=rlmflow.FILE_TOOLS,
)
```

The working directory belongs to the worker. A shared child necessarily inherits its caller's directory.

Tools marked `@tool(proxy=True)` execute on the host through the worker's private protocol stream. Their arguments travel from worker to host as JSON data only. Other values and functions are copied in the trusted host-to-worker direction with `cloudpickle`.

## Host and worker state

Agent code runs in a worker process, so the host and the agent share no heap. Three rules follow, and a tool that ignores them either fails at injection with a pickling error or quietly updates a copy:

- A function that has to run on the host — because it mutates host state, drives the graph, or closes over something unpicklable such as `Flow` — must be marked `@tool(proxy=True)`. Everything else is copied into the worker by value.
- State the host reads back travels through `ENV`, which is per-agent and round-trips with every execution: whatever agent code writes there is readable afterwards with `flow.runtime.get_env_var(agent, name)`. `ENV` values must be JSON-compatible.
- `flow.runtime.get_var(agent, name)` reads JSON-compatible worker state. Custom class instances and other executable Python objects never deserialize on the host; publish the plain fields the host needs through `ENV` or another variable.

`examples/shepherd` follows all three: its recovery tools are proxies, the game publishes each reading and its frame trace to `ENV`, and the game object itself never leaves the worker.

## Stdio and protocol safety

Before user code starts, the worker duplicates stdin and stdout for its private protocol and makes those descriptors non-inheritable. User stdin is redirected to `/dev/null`; native writes and subprocess output are redirected to stderr. Normal Python output is captured per execution. User code therefore cannot read or corrupt protocol frames.

## Timeouts

User execution is unbounded by default, allowing cells to wait for long-running subagents and tools. Pass `execution_timeout=<seconds>` to cap one execution:

```python
runtime = rlmflow.LocalRuntime(execution_timeout=3600)
```

A timeout terminates the entire worker because Python cannot safely kill an arbitrary running thread. `repl_timeout` only bounds startup and control requests; it does not limit user code.

## Restore behavior

Saved graphs remain the durable state. `restore="replay"` reconstructs unfinished namespaces before scheduling the run. `restore="lazy"` waits until an agent is about to execute code, then replays the recorded executions needed by that worker. `reuse_repl=True` is live sharing, not namespace cloning.

## Docker

```bash
docker build -t rlmflow:local .
```

```python
runtime = rlmflow.DockerRuntime(
    "rlmflow:local",
    mounts={"./project": "/workspace"},
    workdir="/workspace",
)
```

The protocol uses stdin/stdout, so no ports or container-to-host network route are required. The default profile has no network, one CPU, 512 MiB of memory, non-root execution, a read-only root filesystem, dropped capabilities, `no-new-privileges`, and process and `/tmp` limits.

## Modal

```bash
pip install rlmflow[modal]
```

```python
runtime = rlmflow.ModalRuntime(remote_workdir="/workspace")
```

Modal uses the same worker protocol over Sandbox stdin/stdout; no tunnel or sandbox-side bridge is involved.

## Custom provisioning

A custom backend only needs a `ReplConnection` with `start`, `send`, `recv`, and `close`, then wraps it in `WorkerSession` and `WorkerRepl`. Execution, host-tool RPC, serialization, tenant routing, and sharing semantics stay common across backends.
