# When to use rlmflow

A guide for choosing between rlmflow and other agent frameworks.

## The one-line pitch

rlmflow is a Python library for building **recursive agents as controllable execution graphs**.

It gives an LLM a REPL and recursive delegation tools, then persists every query, action, observation, child call, wait, resume, and result as typed nodes you can inspect, step, visualize, fork, replay, resume, and branch into new workspaces.

## Decision matrix

| You want to... | Use |
|----------------|-----|
| Build a recursive agent where sub-agents spawn sub-agents | **rlmflow** |
| Inspect the actual execution graph produced by a recursive agent | **rlmflow** |
| Persist/fork/rewind agent execution mid-run | **rlmflow** |
| Run a one-off coding task from your terminal | Codex CLI, Claude Code, Aider |
| Build a graph of heterogeneous agents with complex routing | LangGraph |
| Assemble a team of role-based agents (researcher + writer + reviewer) | CrewAI |
| Optimize prompts/programs against a metric | DSPy |
| Run SWE-bench evaluations with minimal setup | SWE-agent, OpenHands |
| Pair-program with git integration and auto-commits | Aider |

## rlmflow vs specific alternatives

### vs rlm-minimal

[rlm-minimal](https://github.com/alexzhang13/rlm-minimal) is the reference implementation rlmflow grew from. It proves the core recursive-agent idea in a single file.

**Choose rlm-minimal** when you want the simplest possible recursive agent to study or fork.

**Choose rlmflow** when you need:
- Step-level control (pause, inspect, resume)
- Typed graph state with save/load persistence, fork, and time-travel
- Parallel child execution with a thread pool
- `agents/` persistence for per-agent state logs
- `INPUTS` payloads for per-agent task data
- Interactive visualization (Gradio viewer)
- Token tracking and budgets
- Multiple LLM backends and model routing

### vs ypi

[ypi](https://github.com/rawwerks/ypi) is a shell-first recursive agent built on Pi. We borrowed prompt patterns and session design from it.

**Choose ypi** when you want a polished CLI coding agent with jj workspace isolation per child.

**Choose rlmflow** when you want an **embeddable Python library** with:
- Typed config and Pydantic state (not env vars and subprocess orchestration)
- First-class typed graph state you can serialize/replay
- Fine-grained stepping visible to your own code
- Custom runtimes, tools, and prompt builders

### vs LangGraph

[LangGraph](https://github.com/langchain-ai/langgraph) is a graph DSL with channel-based state, durable checkpointing (SQLite/Postgres), and a huge ecosystem.

**Choose LangGraph** when you need:
- Complex graph topologies beyond recursive trees
- Production-grade durable checkpointing to databases
- Large ecosystem of prebuilt integrations
- Enterprise features (LangSmith tracing, deployment)

**Choose rlmflow** when:
- Your problem is naturally recursive (one agent type, spawning copies of itself)
- You want the mental model to be "LLM + REPL + delegation" not "graph nodes + channels"
- You want the graph to emerge from model delegation instead of being hand-authored
- You want session/context/runtime separation in plain Python
- You prefer a small Python library over a large orchestration platform

### vs CrewAI

[CrewAI](https://github.com/crewAIInc/crewAI) builds teams of role-based agents with Flows for deterministic orchestration.

**Choose CrewAI** when you want pre-defined roles (researcher, writer, reviewer) collaborating on a task, with MCP and enterprise tracing.

**Choose rlmflow** when your agents are homogeneous (same prompt, different subtask) and the tree structure emerges from the problem, not from hand-designed roles.

### vs AutoGen

[AutoGen](https://github.com/microsoft/autogen) is an actor-based multi-agent framework with event-driven messaging.

**Choose AutoGen** when you need async multi-agent messaging at scale, OpenTelemetry, or Microsoft ecosystem integration.

**Choose rlmflow** when you want a direct Node step/stream API where the entire
recursive execution is one inspectable tree.

### vs SWE-agent / OpenHands

These are task-specific harnesses for coding agents with Docker sandboxes and benchmark infrastructure.

**Choose SWE-agent/OpenHands** when you're running SWE-bench or need a ready-made coding harness.

**Choose rlmflow** when you're building a general recursive agent (not just code patching) and want full control over the execution loop.

### vs Aider

[Aider](https://github.com/paul-gauthier/aider) is a git-first pair programmer — the best single-agent coding UX.

**Choose Aider** for daily coding tasks with automatic git commits and rollbacks.

**Choose rlmflow** when you need agents to recursively decompose and parallelize work across a tree.

## What rlmflow is NOT

- **Not a product** — it's a library. You bring your own LLM keys, runtime, and UI.
- **Not a sandbox** — `LocalRuntime` runs code in your process. Use `ModalRuntime` or a custom `Runtime` for isolation.
- **Not an integration platform** — we don't ship 100 tool connectors. Register your own tools with `@tool` + `Flow(tools=[...])`, or use the built-in filesystem tools.
- **Not a prompt optimizer** — we provide sensible defaults, but prompt tuning is your job (or pair with DSPy).

## Who should use rlmflow

- **Researchers** studying recursive/hierarchical agent architectures who want inspectable, replayable execution traces.
- **Engineers** building agents that naturally decompose work (code gen, search, summarization) and want step-level control.
- **Teams** that need save/load persistence, fork, and replay for debugging complex multi-agent runs.
- **RL practitioners** who want a gym-style `step()` loop over agent execution.
