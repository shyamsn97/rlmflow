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
6  apps        cli/  consumers.py  view/  rao/
5  driver      flow.py
4  engine      engine/
3  agent API   tools/  prompts.py
2  substrate   llm/  runtime/  integrations/
1  data        graph/
0  leaves      utils.py  structured.py
```

- `graph/` is the bottom: usable with no model, no REPL and no `Flow`.
- `llm/` and `runtime/` know nothing about tools or the engine.
- `tools/` and `prompts.py` are what an agent calls and sees.
- `engine/` is how `Flow` turns one node into the next, and when it stops.
- Everything optional sits on top or in `integrations/`.

Two extra checks in the same test:

- `runtime/repl_server.py` runs in the worker process, so it may import only
  `runtime/repl.py`, `structured.py` and `utils.py`.
- `runtime/repl.py` keeps a literal copy of `AGENT_REF_KEY`; the test asserts it
  equals `graph.nodes.AGENT_REF_KEY`.

## Tree

```text
rlmflow/
  __init__.py        public API
  __main__.py
  flow.py            Flow, FlowLLM, messages_to_query
  structured.py      output codec + structured engines
  prompts.py         prompt prose, Prompt sections, system prompt, message assembly
  consumers.py       stream consumers: checkpoint, sync, live trees
  utils.py           code-block parsing, truncation, host-value encoding

  graph/             the data
    graph.py         Graph (history writes and agent queries), Edge, edge kinds, check_invariants, errors
    nodes.py         node types, LLMUsage, Step context, payload keys, ids
    config.py        AgentConfig, ContextPolicy, Input, argument walkers
    persistence.py   save/load, blobs, v1 upgrade

  llm/               chat clients
    __init__.py      LLMClient protocol, streaming, retries, PooledLLMClient, client_for
    openai.py  anthropic.py  tinker.py

  runtime/           REPL workers
    runtime.py       Runtime, local/subprocess runtimes, connections, ENV, REPL rebuild
    repl.py          worker-side REPL contract, agent references, tool metadata
    repl_client.py   host side of a worker process
    repl_server.py   the worker process and its protocol, verifier fingerprints
    docker.py  modal.py

  integrations/      third-party glue
    claude_code.py   ClaudeCodeBackend
    dspy.py          DSPyFlow
    instructor.py  jev.py

  tools/             what agents call
    namespace.py     @tool, tool docs, RESERVED_TOOLS, REPL namespace building
    builtins.py      submit, llm_query, completion contract, run_verifier
    launch.py        launch_agent, fork_agent
    messaging.py     send/ask/connect/disconnect, wait-cycle checks
    agents_view.py   AGENTS, AgentHandle, resolve_agent
    files.py         FILE_TOOLS, CODING_HANDOFF_SCHEMA


  engine/            steps
    steps.py         @transition, resolve_transition, choose_transition,
                     DEFAULT_TRANSITIONS, AgentBackend
    scheduler.py     TaskQueue, pools, resolve_boundary
    handoff.py       result delivery to reporters

  view/
    figure.py        graph_svg, step_svg, PNG frames and GIFs
    html.py          save_html, save_svg, open_viewer (Gradio)
    text.py          render_tree / render_forest
    timeline.py      steps, timeline, replay, snapshot

  cli/
    __init__.py      entry point, config and version commands, resolve_options
    run.py  view.py
    serve.py         rlmflow serve: a Claude Code session as the root agent
    fork.py          rlmflow fork
    tui.py           FlowTUI

  rao/               rollout, env, credit, export, trainer
```

## Placement rules

- A module is split only when it has a separate reason to load: an optional
  SDK (`llm/openai.py`, `runtime/modal.py`, `integrations/*`), a separate
  process (`runtime/repl_server.py`).
- Optional third-party imports for viewers and image export happen inside the
  call, so `view/` imports eagerly without requiring its extras.
- Names say what the thing does: `resolve_transition`, `resolve_boundary`,
  `resolve_agent`, `resolve_options`, `rebuild_repl`, `run_verifier`.
- History is written only through `Graph`'s history writes (`spawn`,
  `append`, `send`, `deliver`, `fork`, `attach`) and its edge methods
  (`add_edge`, `remove_edge`, `merge`); each checks its own preconditions. Loading a saved log runs `check_invariants` once.

## Moved imports

The flattening pass moved these; `CHANGELOG.md` lists them for callers.

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
