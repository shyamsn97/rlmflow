# Runtimes

A `Runtime` is the user-facing object you pass to `Flow(runtime=...)`. It owns:

- the `working_directory` where agent code runs;
- tools registered with `register_tool(...)` / `register_tools(...)`;
- the backend factory that mints one `Repl` per agent.

The old `repl_factory` pattern is gone. The runtime is the factory.

## Protocol

Subclass `Runtime` and implement `open(node)`:

```python
class Runtime:
    def __init__(self, working_directory: str | Path | None = None): ...

    def open(self, node: Node) -> Repl:
        raise NotImplementedError
```

A `Repl` supports `run(code)`, `close()`, injection, and access to its
`namespace` / `env`. `LocalRuntime` returns an in-process `LocalRepl`.
`DockerRuntime`, `ModalRuntime`, and `SubprocessRuntime` return `RemoteRepl`
instances that speak JSON with `python -m rlmflow.runtime.repl_server`.

## Shipped runtimes

| Runtime | What it does |
|---|---|
| `LocalRuntime(working_directory=...)` | In-process Python REPL. Defaults to the current process cwd. |
| `SubprocessRuntime(...)` | Runs the REPL in a local Python subprocess. |
| `DockerRuntime(image, ...)` | Runs `python -m rlmflow.runtime.repl_server` in `docker run -i --rm` and talks over stdio. |
| `ModalRuntime` | Runs the REPL inside a Modal sandbox. |

## Working directory and tools

```python
import rlmflow
from rlmflow.clients import OpenAIClient

runtime = rlmflow.LocalRuntime(working_directory="./project")
runtime.register_tools(rlmflow.FILE_TOOLS)

agent = rlmflow.Flow(
    OpenAIClient(model="gpt-5"),
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
from rlmflow.clients import OpenAIClient

host_project = Path("./project").resolve()
runtime = rlmflow.DockerRuntime(
    "rlmflow:local",
    mounts={str(host_project): "/workspace"},
    workdir="/workspace",
    network="none",
    cpus=1.0,
    memory="512m",
)
runtime.register_tools(rlmflow.FILE_TOOLS)

agent = rlmflow.Flow(OpenAIClient(model="gpt-5"), runtime=runtime)
```

## Modal

Install provider extras as needed:

```bash
pip install rlmflow[modal]
```

```python
import rlmflow
from rlmflow.clients import OpenAIClient

runtime = rlmflow.ModalRuntime()
runtime.register_tools(rlmflow.FILE_TOOLS)
agent = rlmflow.Flow(OpenAIClient(model="gpt-5"), runtime=runtime)
```

## Writing your own

For in-process behavior, subclass `Runtime` and return a custom `Repl`.
For remote transports, implement a `ReplConnection`, construct a `RemoteRepl`,
and return it from `Runtime.open(node)`.

```python
class MyRuntime(rlmflow.Runtime):
    def open(self, node: rlmflow.Node) -> rlmflow.Repl:
        return MyRepl(...)
```
