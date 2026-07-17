# rlmflow docs

Pick the doc that matches what you're trying to do.

## Get oriented

- [Blog post](https://shyamsn97.github.io/blog/rflow/) — long-form pitch.
  Why recursive agents, why graphs over flat traces, and walkthroughs.
- [Positioning](positioning.md) — when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.

## Use rlmflow

- [Control](control.md) — streaming loop, save/load resume, rewind,
  forks, `INPUTS`, delegation via `launch_subagents`,
  inline-first strategy, custom tools.
- [Streaming and scheduling](streaming.md) — detailed guide to
  `run_streaming(..., until=...)`, per-agent queues, boundaries, and delegation.
- [Skills](skills.md) — workspace `SKILL.md` files, always-on skills,
  query-selected skills, child-only skills, run-memory skills.
- [Node injection](injections.md) — append typed controller events to a
  running graph, then continue with `agent.run(graph=graph)` / `agent.run_streaming(graph=graph)`.
- [Observability](observability.md) — querying the `Graph`,
  run layout, the live terminal tree, the Gradio viewer, and static
  image/step exports.
- [Node model](node_model.md) — typed graph state taxonomy, action /
  observation alternation, delegation wait/resume flow.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes
  (Local / Docker / Modal / E2B / Daytona), writing your own.
- [Prompt customization](prompt_customization.md) — `PromptBuilder`
  sections, callable dynamic sections, deriving from the default prompt,
  full replacement.
- [Security](security.md) — trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Example smoke runner](../examples/run_examples.py) — run the offline
  examples and opt into optional, live, sandbox, or manual checks.

## Extend rlmflow

- [**Internals**](internals.md) — the `Flow`/`Graph`/`Run` split, the two phases
  of a run, the per-agent scheduler, exec turns, delegation, REPL lifecycle,
  runtime backends, graph persistence, and extension seams.

## Research Notes

- [RAO implementation plan](research/rao_implementation_plan.md) — how to
  implement Recursive Agent Optimization as a first-class `rflow.rao` module
  over `Flow` rollouts.
- [DeLM vs. rlmflow](research/delm_vs_rlmflow.md) — how DeLM-style
  coordination could sit on top of recursive execution graphs.
