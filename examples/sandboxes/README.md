# Sandbox example

[`platformer.py`](platformer.py) runs one platformer-building `Flow` task with
every agent's Python running in a sandbox: a Docker container or a Modal
Sandbox. Each runtime lazily opens one REPL per agent. Set `OPENAI_API_KEY`
first.

## Docker

```bash
docker build -t rlmflow:local .
pip install -e ".[openai]"
python examples/sandboxes/platformer.py --runtime docker --docker-image rlmflow:local
```

## Modal

```bash
pip install -e ".[openai,modal]"
modal setup
python examples/sandboxes/platformer.py --runtime modal --model gpt-5
```

The Modal run builds its image from this checkout, installs `rlmflow` inside
it, and runs each agent's REPL under `/workspace` (`--remote-workdir`).
`--app-name`, `--sandbox-timeout` and `--repl-timeout` tune the sandbox.

Runs are saved under `examples/_runs/sandbox-<runtime>/`.
