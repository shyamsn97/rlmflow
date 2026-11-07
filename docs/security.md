# Security

## Trust model

`LocalRuntime` runs agent Python in a separate worker process, but that process still has the host user's filesystem, network, environment, and subprocess permissions. Process separation is not a sandbox. **Use it only for code you'd run yourself.**

For untrusted agents, or agents you haven't audited yet, use an isolated runtime:

- `DockerRuntime` — a fresh container per session.
- `ModalRuntime` — a remote Modal container.
- Custom `Runtime` — SSH, `kubectl exec`, Firecracker, gVisor, anything.

`SubprocessRuntime` selects another Python executable or environment but retains the host user's permissions. It is not a security boundary for untrusted code.

## Docker isolation knobs

```python
DockerRuntime(
    image="rlmflow:local",
    mounts={"./workspace": "/workspace"},
)
```

Those defaults disable networking, cap the worker at one CPU, 512 MiB, and 256 processes, run as `1000:1000`, drop Linux capabilities, enable `no-new-privileges`, use a read-only root filesystem, and provide a bounded `/tmp`. Pass explicit alternatives only when the task requires them; pass `extra_args=[]` to replace the default hardening flags.

The Docker worker uses stdin/stdout for its control protocol, so `network="none"` is supported and requires no published ports.

Mount only what the agent needs. A hostile agent inside the container can still fill its writable volumes, burn CPU up to the quota, and call any tool you injected.

## Engine-level caps

Independent of the runtime:

- `max_depth=1` — recursion limit.
- `max_iters=None` — LLM calls per agent (set a number to cap).
- `max_budget=None` — soft whole-run token threshold; exhaustion allows in-flight calls and one bottom-up final turn per unfinished agent.
- `max_output_length=20_000` — truncate oversized stdout (official RLM's per-block cap).
- `workers` — cap concurrent blocking LLM calls on the flow's thread pool.

## Proxied tools

`Flow(tools=[...])` and `flow.add_tool(fn)` expose callables to the agent worker. Tools execute on the host through the worker's private protocol stream, so a host tool runs with the host's privileges whatever runtime the agent uses; `@tool(where="repl")` tools are copied into the worker instead. Working-directory-aware tools run relative to `runtime.working_directory`. `@tool(..., when=...)` hides a name from that agent's prompt and namespace; it is not a sandbox. The worker can still write files with ordinary Python.

Tools you register are part of the trust boundary. The container can be sealed off, but any injected tool runs on the host with host privileges. Worker-to-host arguments are restricted to JSON data; the host never deserializes worker-controlled Python objects. Keep the tool surface small and still validate the meaning and bounds of every argument.

## Approval gates

There is no built-in approval gate. To review code before it runs, subclass the runtime and check `code` in `execute(agent, code, graph)` before calling `super().execute(...)`; return or raise to refuse. A refused cell is an error the next model turn sees, so the agent can repair its plan.
