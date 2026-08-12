# Security

## Trust model

`LocalRuntime` runs agent Python in a separate worker process, but that
process still has the host user's filesystem, network, environment, and
subprocess permissions. Process separation is not a sandbox. **Use it only for
code you'd run yourself.**

For untrusted agents, or agents you haven't audited yet, use an
isolated runtime:

- `DockerRuntime` — a fresh container per session.
- `ModalRuntime` — a remote Modal container.
- Custom `Runtime` — SSH, `kubectl exec`, Firecracker, gVisor, anything.

`SubprocessRuntime` selects another Python executable or environment but retains
the host user's permissions. It is not a security boundary for untrusted code.

## Docker isolation knobs

```python
DockerRuntime(
    image="rlmflow:local",
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

The Docker worker uses stdin/stdout for its control protocol, so
`network="none"` is supported and requires no published ports.

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

`Flow(tools=[...])` and `flow.add_tool(fn)` expose callables to the agent
worker. Tools marked `@tool(proxy=True)` execute on the host through the
worker's private protocol stream; local tools are copied into the worker when
possible.
Working-directory-aware tools run relative to
`runtime.working_directory`.

Tools you register are part of the trust boundary. The container can
be sealed off, but any injected tool runs on the host with host
privileges. Keep that surface small and validate arguments.

## Overrides for approval gates

Override `Flow.exec_turn(node)` to gate, classify, or rewrite an `ExecAction`
before it touches the runtime. Short-circuit by appending an
`ErrorOutput` (the next model turn sees it as a normal failure observation)
instead of running the code:

```python
import rlmflow

class ReviewingFlow(rlmflow.Flow):
    async def exec_turn(self, action: rlmflow.ExecAction):
        if "rm -rf" in action.code and input(f"run? {action.code}\n> ") != "y":
            rejected = rlmflow.ErrorOutput(
                content="rejected by reviewer",
                output="rejected by reviewer",
                error="rejected",
            )
            return self.append(action, rejected)
        return await super().exec_turn(action)
```

Wrap the runtime or REPL if you want approval at the transport layer.
Subclass `Runtime.open(...)` to return a `Repl` that gates `run(code)` before
delegating to the underlying REPL.
