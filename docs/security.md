# Security

## Trust model

`LocalRuntime` runs agent Python in your process. Same permissions as
your interpreter: filesystem, network, env vars, subprocesses. **Use
it only for code you'd run yourself.**

For untrusted agents, or agents you haven't audited yet, use an
isolated runtime:

- `DockerRuntime` — a fresh container per session.
- `ModalRuntime` — a remote Modal container.
- `E2BRuntime` — a remote E2B sandbox.
- `DaytonaRuntime` — a remote Daytona sandbox.
- Custom `Runtime` — SSH, `kubectl exec`, Firecracker, gVisor, anything.

## Docker isolation knobs

```python
DockerRuntime(
    image="rlmflow:local",
    network="none",           # no outbound traffic
    cpus=1.0,                 # CPU quota
    memory="512m",            # OOM cap
    user="1000:1000",         # non-root
    extra_args=[
        "--read-only",        # read-only rootfs
        "--security-opt", "no-new-privileges",
    ],
    mounts={"./workspace": "/workspace"},
)
```

Mount only what the agent needs. A hostile agent inside the container
can still fill its writable volumes, burn CPU up to the quota, and
call any tool you injected.

## Engine-level caps

Independent of the runtime:

- `max_depth` — recursion limit.
- `max_iters` — LLM calls per agent.
- `max_budget` — total tokens across the subtree.
- `max_output_length` — truncate oversized stdout.
- `workers` — cap concurrent blocking LLM calls on the flow's thread pool.

## Proxied tools

`runtime.register_tool(fn)` and `runtime.register_tools([...])` expose
callables to the agent REPL. Tools marked `@tool(proxy=True)` execute on the
host when called from a remote sandbox; local tools are shipped into the sandbox
when possible. Working-directory-aware tools run relative to
`runtime.working_directory`.

Tools you register are part of the trust boundary. The container can
be sealed off, but any injected tool runs on the host with host
privileges. Keep that surface small and validate arguments.

## Overrides for approval gates

Override `Flow.exec_turn(graph, code, *, replay=False)` to gate, classify, or
rewrite code before it touches the runtime. Short-circuit by appending an
`ErrorOutput` (the next model turn sees it as a normal failure observation)
instead of running the code:

```python
import rlmflow

class ReviewingFlow(rlmflow.Flow):
    async def exec_turn(self, graph, code, *, replay=False):
        if not replay and "rm -rf" in code and input(f"run? {code}\n> ") != "y":
            node = rlmflow.ErrorOutput(
                content="rejected by reviewer",
                output="rejected by reviewer",
                error="rejected",
            )
            self.append_node(graph, node)
            return node
        return await super().exec_turn(graph, code, replay=replay)
```

Wrap the runtime or backend if you want approval at the transport layer.
Subclass `Runtime.open(...)` to return a backend that gates `start(...)` and
`resume(...)` before delegating to the underlying backend.
