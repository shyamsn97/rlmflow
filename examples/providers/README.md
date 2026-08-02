# Providers

Examples that connect `Flow` to another model, tool, or inference provider.

Run commands below from the repository root after installing the matching extra.

## DSPy

Use a `Flow` agent as the LM behind a DSPy program.

```bash
export OPENAI_API_KEY=...
pip install -e ".[openai,dspy]"
python examples/providers/dspy_drop_in.py
```

## MCP Weather

Starts a local FastMCP weather server backed by the real Open-Meteo API, registers
its tools on a `Flow` subclass, delegates Seattle/Austin forecasts to child agents, and
combines the packing advice.

```bash
export OPENAI_API_KEY=...
pip install -e ".[openai,mcp]"
python examples/providers/mcp_weather.py
```

`mcp_weather.py` starts `mcp_weather_server.py` for you over stdio. To run the
FastMCP server directly for debugging:

```bash
pip install -e ".[mcp]"
python examples/providers/mcp_weather_server.py
```

That command starts the server on stdio, which is what MCP clients expect. If you
want to inspect it with your own MCP client, point the client at:

```bash
python examples/providers/mcp_weather_server.py
```

Useful flags:

```bash
python examples/providers/mcp_weather.py --model gpt-5-mini
python examples/providers/mcp_weather.py --out-dir /tmp/mcp-weather-run
```

The run calls Open-Meteo over the network. Pass `--out-dir` to save the final
`graph.json`.

## Tinker

Run `Flow` with Tinker inference.

```bash
export TINKER_API_KEY=...
pip install -e ".[tinker]"
python examples/providers/tinker_agent.py
```

Optional model flags:

```bash
python examples/providers/tinker_agent.py --base-model Qwen/Qwen3-8B
python examples/providers/tinker_agent.py --model-path tinker://run/weights/checkpoint
```
