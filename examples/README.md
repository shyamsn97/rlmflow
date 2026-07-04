# Examples

Single-file scripts live at the root. Multi-file tasks and API tours live in
named folders. Generated runs go under [`_runs/`](_runs/).

## Scripts

| Script | What it shows |
|---|---|
| [`showcase.py`](showcase.py) | End-to-end `Flow` run + live terminal viz |
| [`drop_in_llm.py`](drop_in_llm.py) | Minimal `FlowLLM` as a drop-in LLM |
| [`llm_query_batched.py`](llm_query_batched.py) | `llm_query_batched` in the REPL |
| [`skills.py`](skills.py) | On-disk skills + dynamic prompt section |
| [`structured_output.py`](structured_output.py) | Root + child `output_schema` validation |
| [`view_demo.py`](view_demo.py) | Lightweight viewer on synthetic minimal graphs |
| [`summarizer.py`](summarizer.py) | Recursive map-reduce summarization |

```bash
python examples/showcase.py --no-viz
python examples/skills.py --model gpt-4o-mini
python examples/summarizer.py --sections 10 --no-viz
```

## Tasks

| Folder | What it shows |
|---|---|
| [`needle/`](needle/) | Needle-in-haystack (in-memory + filesystem variants) |
| [`coding/`](coding/) | Interactive file-editing agent |
| [`autoresearch/`](autoresearch/) | TinyStories autoresearch loop with Modal GPU trials |

## Tours & integrations

| Folder | What it shows |
|---|---|
| [`graph/`](graph/) | Offline Graph API (query, edit, save, fork, render) |
| [`control/`](control/) | Delegation, branching, injection |
| [`sandboxes/`](sandboxes/) | Docker and Modal remote execution |
| [`providers/`](providers/) | DSPy, MCP, Tinker adapters |

---

Most compute examples (`summarizer.py`, `needle/haystack.py`, `showcase.py`,
`coding/agent.py`) share the same flags. Run `--help` on any script for defaults.

| Flag | Default | Meaning |
|---|---|---|
| `--model MODEL` | varies | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI). |
| `--fast-model MODEL` | varies | Optional cheap secondary model registered as `fast` for delegates. |
| `--docker-image IMAGE` | unset | If set, run agent code inside this Docker image via a `DockerRuntime`. Must have `rlmflow` installed. Leaving this unset uses the in-process `LocalRuntime`. |
| `--max-depth N` | `3` | Max delegation depth. |
| `--max-iters N` | `15` | Max LLM turns per agent. |
| `--no-viz` | off | Disable the live terminal visualization. |
| `--out-dir PATH` | `_runs/<example-name>/` | Save the final run here. Defaults use flat example names under [`_runs/`](_runs/). |

## Running under Docker

Build the image once:

```bash
docker build -t rlmflow:local .
```

Then pass `--docker-image rlmflow:local` to any example that supports it:

```bash
python examples/summarizer.py                 --docker-image rlmflow:local
python examples/needle/haystack.py            --docker-image rlmflow:local
python examples/needle/filesystem.py          --docker-image rlmflow:local
python examples/coding/agent.py --workdir ./proj --docker-image rlmflow:local
```

Examples that use file tools register them on the runtime
(`runtime.register_tools(FILE_TOOLS)`) and set `working_directory`, so relative
paths resolve into that directory the same way in local and Docker modes.

For true parallel local REPL/code execution, prefer `SubprocessRuntime`: it runs
one local Python process per agent, so cwd and `RFLOW_*` metadata are isolated per
agent. The in-process `LocalRuntime` is useful for low-overhead debugging, but
code blocks that need cwd/env isolation are serialized inside the host process.
Use `DockerRuntime` or a cloud sandbox runtime when you also need container-level
isolation.

A finished run is saved automatically under `_runs/`; reopen it with:

```bash
python examples/summarizer.py        # saves to examples/_runs/summarizer/
rlmflow view examples/_runs/summarizer
rlmflow render examples/_runs/summarizer -f html -o viewer.html
```

The saved directory holds `graph.json` (and optionally `trace.json` when you
capture a step sequence with `save_trace`).

## Docker And Modal

Remote sandbox examples live under [`sandboxes/`](sandboxes/). They run a small
platformer-building task, so set `OPENAI_API_KEY` plus any provider credentials:

```bash
python examples/sandboxes/docker_agent.py --docker-image rlmflow:local --no-live
python examples/sandboxes/modal_agent.py --model gpt-5 --no-live
```

Install the matching extra first: `rlmflow[modal]` or `rlmflow[sandbox]`.

For fully locked-down local runs, pass a `DockerRuntime`:

```python
from rflow.minimal import DockerRuntime, Flow
from rflow.clients import OpenAIClient

runtime = DockerRuntime("rlmflow:local", working_directory="./proj")
flow = Flow(OpenAIClient(model="gpt-4o"), runtime=runtime)
```

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the
deterministic/offline examples; use `--include-optional`, `--include-live`,
`--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect
the suite. Notebooks are documented separately and are not executed by this
runner.

See each subdirectory README for the examples in that group.
