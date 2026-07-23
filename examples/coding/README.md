# Coding agent

Interactive coding agent with the full-screen `FlowTUI` consumer (query/context
inputs, live chat bubbles, and side tabs for the execution tree / agents /
counts / waiting / errors / latest). The example owns the
`flow.run_streaming(...)` loop and feeds events into the TUI.

```bash
pip install -e ".[openai,tui]"
python examples/coding/agent.py --workdir ./myproject
python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
```

Pass `--cli` for the plain REPL + live tree instead of the TUI.

Default working directory: `examples/_runs/coding/`.
