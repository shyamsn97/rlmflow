# rlmflow docs

Pick the doc that matches what you're trying to do.

## Get oriented

- [Blog post](https://shyamsn97.github.io/blog/rlmflow/) — long-form pitch. Why recursive agents, why graphs over flat traces, and walkthroughs.
- [Positioning](positioning.md) — when to use rlmflow vs rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Shepherd walkthrough](shepherd.md) — one worked run: a meta-agent reverts a jammed Sokoban worker's graph and steers eight recoveries in parallel.

## Use rlmflow

- [Command line](cli.md) — `rlmflow run` for a coding agent in the dashboard, `view` and `render` for a saved run, and how settings resolve.
- [Control](control.md) — streaming loop, per-agent limits, multi-turn runs, save/load resume, forks, `INPUTS`, opt-in `AGENTS`, `launch_subagent`, custom tools.
- [Streaming and scheduling](streaming.md) — detailed guide to `run_streaming(..., until=...)`, `TaskQueue`, transitions, delegation, parallel roots, boundaries, cancellation, and Pool/Runtime placement.
- [Skills](skills.md) — workspace `SKILL.md` files, always-on skills, query-selected skills, child-only skills, run-memory skills.
- [Node injection](injections.md) — append controller Nodes between streaming calls, then continue the same root.
- [Observability](observability.md) — querying the Node tree, run layout, stream consumers, and reading a saved run.
- [Node model](node_model.md) — the typed node hierarchy, its transitions, and how delegation is recorded.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes (Local / subprocess / Docker / Modal), writing your own.
- [Prompt customization](prompt_customization.md) — `PromptBuilder`, `Flow.build_system_prompt` / `UserQuery.build_system_prompt`, `PromptProfile` / `prompt_profile` / `prompt_router`.
- [Benchmarks](benchmarks.md) — benchmark harness and runnable sets.
- [Prime Agent benchmark strategy](prime_agent_benchmark_strategy.md) — paper-derived task selection, comparison arms, accounting, and run order.
- [Delegation problem set](delegation_problems.md) — 16 items to A/B `max_depth=0` vs `1`: local and query controls, foreground/background spawn, fresh-context verify. Not wired yet; spec only.
- [Security](security.md) — trust model, Docker isolation knobs, engine-level caps, proxied tools, approval gates.
- [Example smoke runner](https://github.com/shyamsn97/rlmflow/blob/main/examples/run_examples.py) — run the offline examples and opt into optional, live, sandbox, or manual checks.

## Extend rlmflow

- [**Internals**](internals.md) — Node structure, Flow transitions, the task queue and pools, Runtime identity, replay, forks, and persistence.
- [Step table](step_table.md) — `transitions = Transitions()`, then `@Flow.transitions.on(...)` functions.
- [`Node.append`](node_append.md) — `append` is inplace and returns `None`; the step returns the created node.
- [Flow extract](flow_extract.md) — `Flow` stays the facade; restore, builtin factories, and child launch become `(flow, ...)` functions.
- [Step table](step_table.md) — automatic producers, named choices, and direct `transition("name")` routing.
- [Historical colocation review](colocate.md) — the design audit that led to colocated default policy and direct named transitions.

## Historical and experimental research notes

These records explain experiments and design lineage. They are not API reference; dated sketches can use names that the current beta removed.

- [`AGENTS` tree discovery](research/inter_agent_communication.md) — opt-in REPL view for querying and rendering agent status and results.
- [Persistent agents](research/persistent_agents.md) — reopening a finished child with a follow-up question, why peer messaging deadlocks, and the engine constraints either would have to survive.
- [Shared REPLs](runtimes.md#isolated-and-shared-agents) — one namespace for several agents, with `INPUTS`, `finish`, stdout, and error state dispatched per agent through a `ContextVar`.
- [Historical REPL correctness and sharing plan](research/repl_plan.md) — the staged plan: the transport bugs found along the way, then local and remote co-tenancy.
- [RAO implementation plan](research/rao_implementation_plan.md) — the design record behind the now-implemented but still experimental `rlmflow.rao` module.
- [Speculative programmatic tool calling](research/spec_ptc.md) — implement Zhang's sPTC algorithm in `rlmflow.speculate` (shadow REPL, claim-or-run futures) without depending on the spec-ptc package.
- [Shepherd run analysis](research/shepherd_run_analysis.md) — a concrete rewind/replay/fan-out walkthrough from a saved eight-branch Sokoban run, including outcomes, parallelism, process boundaries, persistence, and the terminal-worker lifecycle bug exposed by the artifacts.
- [DeLM vs. rlmflow](research/delm_vs_rlmflow.md) — how DeLM-style coordination could sit on top of recursive execution graphs.
- [Cursor self-driving codebases](research/cursor_self_driving_codebases.md) — Wilson Lin's multi-agent coding harness vs rlmflow: same recursive tree, different join.
- [Procedural Graphs](research/procedural_graphs.md) — Lu et al. 2026: a reusable *what-to-do* prior vs rlmflow's *what-happened* tree. Localized guidance + gated topology evolution; not a second Node type.
- [Runtime harness optimization](research/runtime_harness.md) — ACE / Continual Harness / GEPA / RHO mapped onto the live Node tree, with copy-paste sketches against `Flow` (not weight training).
