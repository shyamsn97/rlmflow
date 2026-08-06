# rlmflow docs

Pick the doc that matches what you're trying to do.

## Get oriented

- [Blog post](https://shyamsn97.github.io/blog/rlmflow/) — long-form pitch.
  Why recursive agents, why graphs over flat traces, and walkthroughs.
- [Positioning](positioning.md) — when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.

## Use rlmflow

- [Control](control.md) — streaming loop, per-agent limits, multi-turn runs,
  save/load resume, forks, `INPUTS`, opt-in `AGENTS`, `launch_subagents`,
  custom tools.
- [Streaming and scheduling](streaming.md) — detailed guide to
  `run_streaming(..., until=...)`, `TaskQueue`, transitions, delegation,
  parallel roots, boundaries, cancellation, and Pool/Runtime placement.
- [Skills](skills.md) — workspace `SKILL.md` files, always-on skills,
  query-selected skills, child-only skills, run-memory skills.
- [Node injection](injections.md) — append controller Nodes between streaming
  calls, then continue the same root.
- [Observability](observability.md) — querying the Node tree, run layout,
  stream consumers, and reading a saved run.
- [Node model](node_model.md) — the seven node types, the transitions between
  them, and how delegation is recorded.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes
  (Local / subprocess / Docker / Modal), writing your own.
- [Prompt customization](prompt_customization.md) — `SystemPromptBuilder`
  sections, `PromptProfile` / `prompt_profile` / `prompt_router`, full replacement.
- [Security](security.md) — trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Example smoke runner](../examples/run_examples.py) — run the offline
  examples and opt into optional, live, sandbox, or manual checks.

## Extend rlmflow

- [**Internals**](internals.md) — Node structure, Flow transitions, the task
  queue and pools, Runtime identity, replay, forks, and persistence.

## Research Notes

- [`AGENTS` tree discovery](research/inter_agent_communication.md) — opt-in REPL
  view for querying and rendering agent status and results.
- [RAO implementation plan](research/rao_implementation_plan.md) — how to
  implement Recursive Agent Optimization as a first-class `rlmflow.rao` module
  over `Flow` rollouts.
- [DeLM vs. rlmflow](research/delm_vs_rlmflow.md) — how DeLM-style
  coordination could sit on top of recursive execution graphs.
