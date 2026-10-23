# rlmflow docs

Pick the doc that matches what you're trying to do.

## Flagship examples

- [Needle](https://github.com/shyamsn97/rlmflow/tree/main/examples/needle) — recursive decomposition over a million-line REPL input, with a saved graph that can be rendered as SVG, HTML, or GIF.
- [Shepherd recovery](shepherd.md) — a live meta-agent inspects a failed Sokoban run, rewinds its graph, forks eight recovery plans, and selects the best result.
- [Boids coding](https://github.com/shyamsn97/rlmflow/tree/main/examples/coding) — a coding agent chooses the architecture and graph for a dependency-free browser simulation from product-level constraints.

Together they show practical recursive execution at three scales: decompose long-context work, construct a multi-module application, then intervene in and recover the retained graph itself. The [benchmark suite](benchmarks.md) separately evaluates long-context work with OOLONG and persistent visual interaction with MazeBench.

## Get oriented

- [Blog post](https://shyamsn97.github.io/blog/rlmflow/) — long-form pitch. Why recursive agents, why graphs over flat traces, and walkthroughs.
- [Positioning](positioning.md) — when to use rlmflow vs rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Shepherd walkthrough](shepherd.md) — the flagship worked run and its reproducible visual artifacts.

## Use rlmflow

- [Command line](cli.md) — `rlmflow tui` for the coding dashboard, `run` for headless execution, `view` and `render` for a saved run, and how settings resolve.
- [Control](control.md) — streaming loop, per-agent limits, multi-turn runs, save/load resume, forks, direct arguments, opt-in `AGENTS`, `launch_subagent`, custom tools.
- [Streaming and scheduling](streaming.md) — detailed guide to `run_streaming(..., until=...)`, `TaskQueue`, transitions, delegation, parallel roots, boundaries, cancellation, and Pool/Runtime placement.
- [Skills](skills.md) — workspace `SKILL.md` files, always-on skills, query-selected skills, child-only skills, run-memory skills.
- [Node injection](injections.md) — append controller Nodes between streaming calls, then continue the same root.
- [Observability](observability.md) — querying the Node tree, run layout, stream consumers, and reading a saved run.
- [Node model](node_model.md) — the typed node hierarchy, its transitions, and how delegation is recorded.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes (Local / subprocess / Docker / Modal), writing your own.
- [Prompt customization](prompt_customization.md) — `Prompt`, Flow default factories, node prompt/message hooks, and prompt profiles.
- [Structured outputs](structured_outputs.md) — output codecs, structured engines, explicit behavior composition, and Jev.
- [Benchmarks](benchmarks.md) — benchmark harness and runnable sets.
- [Security](security.md) — trust model, Docker isolation knobs, engine-level caps, proxied tools, approval gates.
- [Example smoke runner](https://github.com/shyamsn97/rlmflow/blob/main/examples/run_examples.py) — run the offline examples and opt into optional, live, sandbox, or manual checks.

## Extend rlmflow

- [**Scope**](design/scope.md) — what the library provides (Graph, Flow, coding agents), what stays in user code, and the rule that decides.
- [**Internals**](internals.md) — Node structure, Flow transitions, the task queue and pools, Runtime identity, replay, forks, and persistence.
- [`Node.append`](node_append.md) — `append` is inplace and returns `None`; the step returns the created node.
- [Flow extract](flow_extract.md) — `Flow` stays the facade; restore, builtin factories, and child launch become `(flow, ...)` functions.
- [Step table](step_table.md) — automatic transition functions and semantic guards.
- [Historical colocation review](colocate.md) — the design audit that led to colocated default policy; its named-transition path was later removed.

## Historical and experimental research notes

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
