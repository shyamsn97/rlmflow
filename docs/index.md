# rlmflow docs

Pick the doc that matches what you're trying to do.

## Get oriented

- [Blog post](https://shyamsn97.github.io/blog/rlmflow/) — long-form pitch.
  Why recursive agents, why graphs over flat traces, and walkthroughs.
- [Positioning](positioning.md) — when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.

## Use rlmflow

- [Control](control.md) — streaming loop, save/load resume, rewind,
  forks, `INPUTS`, cold `launch_subagents` and warm `launch_branches`, custom
  tools.
- [Streaming and scheduling](streaming.md) — detailed guide to
  `run_streaming(..., until=...)`, structured concurrency, boundaries, and delegation.
- [Skills](skills.md) — workspace `SKILL.md` files, always-on skills,
  query-selected skills, child-only skills, run-memory skills.
- [Node injection](injections.md) — append controller Nodes between streaming
  calls, then continue the same root.
- [Observability](observability.md) — querying the Node tree,
  run layout, `LiveGraphTree` / `LiveTreeRenderer`, the Gradio viewer, and static
  image/step exports.
- [Node model](node_model.md) — typed graph state taxonomy, action /
  observation alternation, delegation wait/resume flow.
- [Runtimes](runtimes.md) — `Runtime` protocol, shipped runtimes
  (Local / subprocess / Docker / Modal), writing your own.
- [Prompt customization](prompt_customization.md) — `SystemPromptBuilder`
  sections, `PromptProfile` / `prompt_profile` / `prompt_router`, full replacement.
- [Security](security.md) — trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Example smoke runner](../examples/run_examples.py) — run the offline
  examples and opt into optional, live, sandbox, or manual checks.

## Extend rlmflow

- [**Internals**](internals.md) — Node structure, Flow transitions, structured
  concurrency, publication, Runtime identity, surgery, and persistence.

## Research Notes

- [RAO implementation plan](research/rao_implementation_plan.md) — how to
  implement Recursive Agent Optimization as a first-class `rlmflow.rao` module
  over `Flow` rollouts.
- [DeLM vs. rlmflow](research/delm_vs_rlmflow.md) — how DeLM-style
  coordination could sit on top of recursive execution graphs.
