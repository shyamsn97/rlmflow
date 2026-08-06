# Runtimes

A `Runtime` is the user-facing object you pass to `Flow(runtime=...)`. It owns:

- the `working_directory` where agent code runs;
- the backend factory that mints one `Repl` per agent;
- the live REPLs themselves, and their replay and cleanup.

Tools belong to the flow, not the runtime: pass `Flow(tools=[...])`, or add one
later with `flow.add_tool(fn)`.

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
from rlmflow.llm import OpenAIClient

flow = rlmflow.Flow(
    OpenAIClient(model="gpt-5"),
    runtime=rlmflow.LocalRuntime(working_directory="./project"),
    tools=rlmflow.FILE_TOOLS,
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
from rlmflow.llm import OpenAIClient

host_project = Path("./project").resolve()
runtime = rlmflow.DockerRuntime(
    "rlmflow:local",
    mounts={str(host_project): "/workspace"},
    workdir="/workspace",
    network="none",
    cpus=1.0,
    memory="512m",
)

flow = rlmflow.Flow(
    OpenAIClient(model="gpt-5"), runtime=runtime, tools=rlmflow.FILE_TOOLS
)
```

## Modal

Install provider extras as needed:

```bash
pip install rlmflow[modal]
```

```python
import rlmflow
from rlmflow.llm import OpenAIClient

flow = rlmflow.Flow(
    OpenAIClient(model="gpt-5"),
    runtime=rlmflow.ModalRuntime(),
    tools=rlmflow.FILE_TOOLS,
)
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
