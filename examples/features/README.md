# Features

One script per feature. Run them from the repository root.

| Script | What it shows | Needs |
| --- | --- | --- |
| [`skills.py`](skills.py) | Skills as files in a live prompt section, and an `add_skill` tool the agent extends itself with | a key, or `--scripted` offline |
| [`structured_output.py`](structured_output.py) | `output_schema` on the root and its children | a key |
| [`mcp_tools.py`](mcp_tools.py) | Every tool on an MCP server, as REPL tools | a key, `.[openai,mcp]` |
| [`sandboxes.py`](sandboxes.py) | Every agent's code in a Docker container or a Modal sandbox | a key, Docker or Modal |

## MCP

`mcp.py` starts [`mcp_weather_server.py`](mcp_weather_server.py), a FastMCP server backed by the real Open-Meteo API, over stdio. It registers the server's tools on the `Flow`, delegates Seattle and Austin forecasts to child agents, and combines the packing advice.

```bash
pip install -e ".[openai,mcp]"
python examples/features/mcp_tools.py --model gpt-5-mini --out-dir /tmp/mcp-run
```

Run `python examples/features/mcp_weather_server.py` to start the server on its own, for debugging with your own MCP client.

## Sandboxes

`sandboxes.py` builds a small platformer with every agent's Python in a sandbox. Each runtime opens one REPL per agent, when the agent first runs code.

```bash
docker build -t rlmflow:local .
python examples/features/sandboxes.py --runtime docker --docker-image rlmflow:local

pip install -e ".[openai,modal]" && modal setup
python examples/features/sandboxes.py --runtime modal --model gpt-5
```

The Modal run builds its image from this checkout, installs `rlmflow` inside it, and runs each agent's REPL under `/workspace` (`--remote-workdir`). `--app-name`, `--sandbox-timeout` and `--repl-timeout` tune the sandbox. Runs are saved under `examples/_runs/sandbox-<runtime>/`.
