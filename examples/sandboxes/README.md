# Sandbox Examples

These examples run one platformer-building `Flow` task with Python execution
isolated in Docker or Modal. They use `OpenAIClient`, so set `OPENAI_API_KEY`
before running them.

Each runtime lazily opens one REPL per agent.

## Docker

```bash
docker build -t rlmflow:local .
pip install -e ".[openai]"
export OPENAI_API_KEY=...
python examples/sandboxes/docker_agent.py --docker-image rlmflow:local
```

The generated project and run data are written under
`examples/_runs/sandbox-docker/`.

## Modal

```bash
pip install -e ".[openai,modal]"
export OPENAI_API_KEY=...
modal setup
python examples/sandboxes/modal_agent.py --model gpt-5
```

The Modal example builds its image from this checkout, installs `rlmflow`
inside it, and runs each agent REPL under `/workspace`.

```bash
python examples/sandboxes/modal_agent.py \
  --app-name rlmflow-dev \
  --sandbox-timeout 600 \
  --remote-workdir /workspace
```
