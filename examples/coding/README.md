# Coding agent

Interactive coding agent as a plain REPL. The example owns the
`flow.run_streaming(...)` loop: it prints one line per node the run lands
(agent path and node type), checkpoints the tree under `<workdir>/graph` as it
goes, and prints the final answer when the root is done.

```bash
pip install -e ".[openai]"
python examples/coding/agent.py --workdir ./myproject
python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
```

Default working directory: `examples/_runs/coding/`.
