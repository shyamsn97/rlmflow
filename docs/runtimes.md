# Runtimes

A `Runtime` is the user-facing object you pass to `Flow(runtime=...)`. It owns:

- the `working_directory` where agent code runs;
- tools registered with `register_tool(...)` / `register_tools(...)`;
- the backend factory that mints one `ReplBackend` per agent.

The old `repl_factory` pattern is gone. The runtime is the factory.

## Protocol

Subclass `Runtime` and implement `open(agent)`:

```python
class Runtime(ABC):
    def __init__(self, working_directory: str | Path | None = None): ...

    @abstractmethod
    def open(self, agent: Graph) -> ReplBackend: ...
```

A `ReplBackend` supports `start(code)`, `resume(value)`, `close()`, and exposes
its `namespace` / `env`. `LocalRuntime` returns an in-process `REPL`.
`DockerRuntime` and sandbox runtimes return `RemoteRepl` backends that speak JSON
with `python -m rlmflow.runtime.repl_server`.

## Shipped runtimes

| Runtime | What it does |
|---|---|
| `LocalRuntime(working_directory=...)` | In-process Python REPL. Defaults to the current process cwd. |
| `DockerRuntime(image, ...)` | Runs `python -m rlmflow.runtime.repl_server` in `docker run -i --rm` and talks over stdio. |
| `ModalRuntime` | Runs the REPL inside a Modal sandbox. |
| `E2BRuntime` | Runs the REPL inside an E2B sandbox. |
| `DaytonaRuntime` | Runs the REPL inside a Daytona sandbox. |

## Working directory and tools

```python
runtime = rlmflow.LocalRuntime(working_directory="./project")
runtime.register_tools(rlmflow.FILE_TOOLS)

agent = rlmflow.Flow(
    rlmflow.OpenAIClient(model="gpt-5"),
    runtime=runtime,
)
```

Relative paths in agent code and in `FILE_TOOLS` resolve inside the runtime's
working directory. The same shape works for Docker and cloud sandboxes.

## Docker

Build the local image once:

```bash
docker build -t rlmflow:local .
```

Then pass a Docker runtime:

```python
from pathlib import Path

import rlmflow

host_project = Path("./project").resolve()
runtime = rlmflow.DockerRuntime(
    "rlmflow:local",
    mounts={host_project: "/workspace"},
    workdir="/workspace",
    network="none",
    cpus=1.0,
    memory="512m",
)
runtime.register_tools(rlmflow.FILE_TOOLS)

agent = rlmflow.Flow(rlmflow.OpenAIClient(model="gpt-5"), runtime=runtime)
```

## Remote sandboxes

Install provider extras as needed:

```bash
pip install rlmflow[modal]
pip install rlmflow[e2b]
pip install rlmflow[daytona]
pip install rlmflow[sandbox]   # all three
```

```python
import rlmflow
from rlmflow.runtime.sandbox.e2b import E2BRuntime

runtime = E2BRuntime(remote_workdir="/workspace")
runtime.register_tools(rlmflow.FILE_TOOLS)
agent = rlmflow.Flow(rlmflow.OpenAIClient(model="gpt-5"), runtime=runtime)
```

See [`examples/sandboxes/`](../examples/sandboxes/) for real-agent examples on
Modal, E2B, and Daytona.

## Writing your own

For in-process behavior, subclass `Runtime` and return a custom `ReplBackend`.
For remote transports, subclass `RemoteRepl` and implement `send(msg)` /
`recv()`; then wrap it in a `Runtime.open(agent)` method.

```python
class MyRuntime(rlmflow.Runtime):
    def open(self, agent: rlmflow.Graph) -> rlmflow.ReplBackend:
        return MyRepl(...)
```
