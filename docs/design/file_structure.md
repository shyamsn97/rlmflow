# File structure

How `rlmflow/` is laid out, and the rules that keep it that way. Each package
is one concept, each file is named for what is in it, and imports only point
down a fixed stack of layers. `from rlmflow import X` works for every public
name; deep imports have one home and no compatibility re-exports.

Per-file contents are in [codebase.md](codebase.md).

## Layers

Imports may only point to the same layer or lower. Imports inside
`TYPE_CHECKING` blocks are exempt. `tests/test_layers.py` enforces this with
`ast`; it also checks that layer-0 modules import nothing from rlmflow.

```text
6  apps        cli/  consumers.py  view/
5  driver      flow.py
4  agent API   tools/  harnesses/
3  engine      engine/  agent_tools/
2  substrate   llm/  runtime/  integrations/
1  data        graph/
0  leaves      utils.py  schema.py
```

- `graph/` is the bottom: usable with no model, no REPL and no `Flow`. Every
  graph write, including edits and invalidation, lives there.
- `llm/` and `runtime/` know nothing about tools or the engine.
- `engine/` is scheduling and workspaces; `agent_tools/` is what an agent's
  REPL calls back into (launch, await, send, `AGENTS`).
- `harnesses/` turn one ready agent into its next node; `tools/` is the tool
  decorator, built-ins and file tools.
- Everything optional sits on top or in `integrations/`.

Two extra checks in the same test:

- `runtime/repl_server.py` runs in the worker process, so it may import only
  `runtime/repl.py`, `schema.py` and `utils.py`.
- `runtime/repl.py` keeps a literal copy of `AGENT_REF_KEY`; the test asserts it
  equals `graph.nodes.AGENT_REF_KEY`.

## Tree

```text
rlmflow/
  __init__.py        public API
  __main__.py
  flow.py            Flow: config, scheduling, result delivery, rerun, streaming
  schema.py          JSON values and the output-schema check
  consumers.py       stream consumers: checkpoint, sync, live trees
  utils.py           code-block parsing, truncation, host-value encoding

  graph/             the data
    graph.py         Graph: agents, nodes, edges, queries, writes, edits, Invalidation
    nodes.py         Agent record, node types, LLMUsage, Step, ids
    config.py        AgentConfig, ContextPolicy, argument walkers
    persistence.py   save/load (format 4), blobs

  llm/               chat clients
    __init__.py      LLMClient protocol, streaming, retries, PooledLLMClient, client_for
    openai.py  anthropic.py

  runtime/           REPL workers
    runtime.py       Runtime, local/subprocess runtimes, REPL rebuild
    repl.py          worker-side REPL contract, agent references
    repl_client.py   host side of a worker process
    repl_server.py   the worker process and its protocol
    docker.py  modal.py

  integrations/
    mcp.py           MCPTools toolkit

  engine/
    scheduler.py     TaskQueue, pools
    workspaces.py    LocalWorkspace, GitWorktrees

  agent_tools/       the REPL's line back to the graph
    tools.py         launch, await, send, run_repl as functions of (flow, step)
    view.py          AgentHandle, AGENTS
    server.py        HTTP server for external harnesses
    cli.py  mcp.py  hooks.py   `rlmflow repl`, the MCP server, Claude Code hooks

  harnesses/         one ready agent -> its next node
    base.py  repl.py  sessions.py  claude_code.py  pi.py  interactive.py  prompts.py

  tools/
    __init__.py      @tool, built-ins (submit, llm_query), toolkits
    files.py         FILE_TOOLS

  view/
    figure.py        graph_svg, step_svg, PNG frames and GIFs
    html.py          save_html, save_svg, open_viewer (Gradio)
    text.py          render_tree / render_forest
    timeline.py      steps, timeline, replay, snapshot

  cli/
    __init__.py      entry point, config and version commands, resolve_options
    run.py  view.py  serve.py  fork.py

  plugin/            the Claude Code plugin (commands, skills)
```

## Placement rules

- A module is split only when it has a separate reason to load: an optional
  SDK (`llm/openai.py`, `runtime/modal.py`, `integrations/*`), a separate
  process (`runtime/repl_server.py`).
- Optional third-party imports for viewers and image export happen inside the
  call, so `view/` imports eagerly without requiring its extras.
- The graph is written only through `Graph` methods (`create_agent`, `spawn`,
  `append`, `send`, `deliver`, `fork`, `add_edge`, `edit`, `delete`, `remove`,
  `replace`), each under the graph lock and checking its own preconditions.

## Moved imports (0.7 flattening)

The 0.7 flattening pass moved these, before core v2 removed several of the targets; `CHANGELOG.md` lists them for callers.

| Old                                                | New                                                                                            |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `rlmflow.graph.records`                            | `rlmflow.graph.graph` (edges); `rlmflow.graph.nodes` (`AGENT_REF_KEY`, `ASK_KEY`, ids)         |
| `rlmflow.graph.usage`                              | `rlmflow.graph.nodes`                                                                          |
| `rlmflow.graph.arguments`                          | `rlmflow.graph.config`                                                                         |
| `rlmflow.models.*`                                 | `rlmflow.llm` / `rlmflow.llm.<provider>`                                                       |
| `rlmflow.structured.*`                             | `rlmflow.structured`                                                                           |
| `rlmflow.structured.instructor`, `.jev`            | `rlmflow.integrations.instructor`, `.jev`                                                      |
| `rlmflow.backends`, `rlmflow.backends.claude_code` | `rlmflow.engine.steps`, `rlmflow.integrations.claude_code`                                     |
| `rlmflow.adapters`                                 | `rlmflow.flow` (`FlowLLM`), `rlmflow.integrations.dspy` (`DSPyFlow`)                           |
| `rlmflow.steps.*`                                  | `rlmflow.engine.steps` (verification: `rlmflow.tools.builtins`, `rlmflow.runtime.repl_server`) |
| `rlmflow.engine.queue`, `.boundaries`              | `rlmflow.engine.scheduler`                                                                     |
| `rlmflow.engine.restore`                           | `rlmflow.runtime.runtime`                                                                      |
| `rlmflow.runtime.env`, `.connections`              | `rlmflow.runtime.runtime`                                                                      |
| `rlmflow.runtime.references`                       | `rlmflow.runtime.repl`                                                                         |
| `rlmflow.runtime.protocol`                         | `rlmflow.runtime.repl_server`                                                                  |
| `rlmflow.tools.decorator`                          | `rlmflow.tools.namespace`                                                                      |
| `rlmflow.tools.completion`                         | `rlmflow.tools.builtins`                                                                       |
| `rlmflow.tools.graph_tools`                        | `rlmflow.tools.messaging` (`fork_agent`: `rlmflow.tools.launch`)                               |
| `rlmflow.tools.directory`                          | `rlmflow.tools.agents_view`                                                                    |
| `rlmflow.tools.filesystem`, `.coding`              | `rlmflow.tools.files`                                                                          |
| `rlmflow.prompts.{text,prompts,sections,messages}` | `rlmflow.prompts` (turn-node text: `rlmflow.graph.nodes`)                                      |
| `rlmflow.utils.*`                                  | `rlmflow.utils`                                                                                |
| `rlmflow.consumers.*`                              | `rlmflow.consumers` (`FlowTUI`: `rlmflow.cli.tui`)                                             |
| `rlmflow.cli.{app,config,options}`                 | `rlmflow.cli`                                                                                  |
| `rlmflow.view.images`                              | `rlmflow.view.figure`                                                                          |
| `rlmflow.view.app`                                 | `rlmflow.view.html`                                                                            |
