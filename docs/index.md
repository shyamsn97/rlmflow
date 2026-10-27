# rlmflow docs

## Start here

- [Usage](usage.md): run a task, configs, models, tools, runtimes, streaming, forks, Claude Code and pi, the CLI.
- [Agents](agents.md): what an agent does in a run: its REPL tools, launching and waiting, messages, Claude Code sessions, forks.
- [Command line](cli.md): every `rlmflow` command and flag, and how settings resolve.

## Guides

- [Runtimes](runtimes.md): where agent code runs: Local, subprocess, Docker, Modal, or your own.
- [Security](security.md): the trust model, Docker isolation, caps, proxied tools.
- [Streaming and scheduling](streaming.md): `run_streaming(..., until=...)`, boundaries, cancellation.
- [Control](control.md): limits, multi-turn runs, save and resume, forks, custom tools.
- [Node injection](injections.md): add nodes between streaming calls and continue the run.
- [Observability](observability.md): querying the graph, consumers, reading a saved run.
- [Node model](node_model.md): node types, their data, and edges.
- [Prompt customization](prompt_customization.md): prompts, sections and profiles.
- [Skills](skills.md): `SKILL.md` files in prompts.
- [Benchmarks](benchmarks.md): the evaluation suites and how to run them.

## Examples and context

- [Shepherd walkthrough](shepherd.md): a meta-agent rewinds a failed run and forks recovery plans.
- [Needle](https://github.com/shyamsn97/rlmflow/tree/main/examples/needle) and [Boids coding](https://github.com/shyamsn97/rlmflow/tree/main/examples/coding): long-context search and a multi-file coding run.
- [Positioning](positioning.md): rlmflow next to other agent frameworks.
- [Blog post](https://shyamsn97.github.io/blog/rlmflow/).

## How it's built

- [Internals](internals.md): the Flow, replay, forks and persistence.
- [Scope](design/scope.md): what the library provides and what stays in user code.
- [File structure](design/file_structure.md) and [codebase](design/codebase.md): the package layout, its import rules, and every module.

## History

Design records kept for context. They describe code as it was, and many of the names in them no longer exist.

- [Colocation review](history/colocate.md), [Flow extract](history/flow_extract.md), [`Node.append`](history/node_append.md), [Step table](history/step_table.md), [Structured outputs](history/structured_outputs.md).
- [Agent-directed transitions](history/self_evolving_transitions.md) and [graph-native context evolution](history/graph_context_evolution.md): unimplemented proposals.

## Research notes

These records explain experiments and design lineage. They are not API reference; dated sketches can use names that the current beta removed.

- [`AGENTS` tree discovery](research/inter_agent_communication.md) — opt-in REPL view for querying and rendering agent status and results.
- [Persistent agents](research/persistent_agents.md) — reopening a finished child with a follow-up question, why peer messaging deadlocks, and the engine constraints either would have to survive.
- [Colocated agents](runtimes.md#isolated-and-colocated-agents) — one worker for several agents with isolated namespaces and per-agent invocation arguments, `submit`, stdout, and errors.
- [Historical REPL correctness and sharing plan](research/repl_plan.md) — the staged plan: the transport bugs found along the way, then local and remote co-tenancy.
- [RAO implementation plan](research/rao_implementation_plan.md) — the design record behind the now-implemented but still experimental `rlmflow.rao` module.
- [Speculative programmatic tool calling](research/spec_ptc.md) — implement Zhang's sPTC algorithm in `rlmflow.speculate` (shadow REPL, claim-or-run futures) without depending on the spec-ptc package.
- [Shepherd run analysis](research/shepherd_run_analysis.md) — a concrete rewind/replay/fan-out walkthrough from a saved eight-branch Sokoban run, including outcomes, parallelism, process boundaries, persistence, and the terminal-worker lifecycle bug exposed by the artifacts.
- [DeLM vs. rlmflow](research/delm_vs_rlmflow.md) — how DeLM-style coordination could sit on top of recursive execution graphs.
- [Cursor self-driving codebases](research/cursor_self_driving_codebases.md) — Wilson Lin's multi-agent coding harness vs rlmflow: same recursive tree, different join.
- [Procedural Graphs](research/procedural_graphs.md) — Lu et al. 2026: a reusable *what-to-do* prior vs rlmflow's *what-happened* tree. Localized guidance + gated topology evolution; not a second Node type.
- [Runtime harness optimization](research/runtime_harness.md) — ACE / Continual Harness / GEPA / RHO mapped onto the live Node tree, with copy-paste sketches against `Flow` (not weight training).
- [JAZ vs. rlmflow](research/jaz_vs_rlmflow.md) — measured prompt, data-grounding, and orchestration analysis from the GPT-5-mini fan-out pilot, with controlled ablations.
