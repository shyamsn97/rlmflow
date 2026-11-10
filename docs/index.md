# rlmflow docs

## Start here

- [Usage](usage.md): run a task, configs, models, tools, runtimes, streaming, forks, Claude Code and pi, the CLI.
- [Agents](agents.md): what an agent does in a run: its REPL tools, launching and waiting, messages, Claude Code sessions, forks.
- [Command line](cli.md): every `rlmflow` command and flag, and how settings resolve.

## Guides

- [Runtimes](runtimes.md): where agent code runs: Local, subprocess, Docker, Modal, or your own.
- [Security](security.md): the trust model, Docker isolation, caps, proxied tools.
- [Streaming and turns](streaming.md): `run_streaming(..., until=...)`, turns, boundaries, cancellation.
- [Control](control.md): limits, multi-turn runs, save and resume, forks, custom tools.
- [Node injection](injections.md): add messages and edits between streaming calls and continue the run.
- [Observability](observability.md): querying the graph, consumers, reading a saved run.
- [The graph](node_model.md): nodes, agents, edges, editing, and the saved format.
- [Prompt customization](prompt_customization.md): each harness's prompt sections, and overriding them from a config or a harness subclass.
- [Skills](skills.md): `SKILL.md` files in prompts.

## Examples and context

- [Shepherd walkthrough](shepherd.md): a meta-agent rewinds a failed run and forks recovery plans.
- [Needle](https://github.com/shyamsn97/rlmflow/tree/main/examples/needle) and [Boids coding](https://github.com/shyamsn97/rlmflow/tree/main/examples/coding): long-context search and a multi-file coding run.
- [Example results](example_results.md): real runs of the examples, with tables, figures and caveats.
- [Positioning](positioning.md): rlmflow next to other agent frameworks.
- [Blog post](https://shyamsn97.github.io/blog/rlmflow/).

## How it's built

- [Internals](internals.md): the Flow, replay, forks and persistence.
