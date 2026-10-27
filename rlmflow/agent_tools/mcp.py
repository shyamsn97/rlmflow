"""The agent tools as a stdio MCP server; ``repl`` is the ``python`` tool.

A harness that speaks MCP launches ``python -m rlmflow.agent_tools.mcp`` with the
turn's ``RLMFLOW_*`` environment; the Claude Code plugin launches ``rlmflow mcp``,
which finds ``rlmflow serve`` through ``.rlmflow/serve.json``. Every tool is one request to the same
agent-tools server the CLI uses, so a ``python`` call lands a ``code`` node.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from rlmflow.agent_tools.cli import REPL_TIMEOUT, request


def mcp_config(env: dict[str, str]) -> dict[str, Any]:
    """An ``--mcp-config`` document that starts this server with ``env``."""
    command = [sys.executable, "-m", "rlmflow.agent_tools.mcp"]
    return {"mcpServers": {"rlmflow": {"command": command[0], "args": command[1:], "env": env}}}


def build_server() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer

    server = MCPServer("rlmflow")

    async def call(tool: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        return await asyncio.to_thread(request, tool, body, timeout=timeout)

    @server.tool()
    async def python(code: str) -> str:
        """Run Python in your persistent REPL. Variables persist; only printed output comes back."""
        ran = await call("repl", {"code": code}, REPL_TIMEOUT)
        return ran["output"] + ("\n[submitted]" if ran["submitted"] else "")

    @server.tool()
    async def spawn(task: str, config: str | None = None, name: str | None = None) -> str:
        """Start an agent on ``task`` with a config name (default: yours). Returns its id."""
        return (await call("spawn", {"task": task, "config": config, "name": name}))["id"]

    @server.tool()
    async def wait(agents: list[str] | None = None, timeout: float = 60.0) -> str:
        """Wait up to a minute for agents; returns each result or "running"."""
        waited = await call("wait", {"agents": agents or [], "timeout": timeout}, timeout + 30)
        text = json.dumps(waited["results"], indent=2, default=str)
        return f"{text}\n{waited['note']}" if waited.get("note") else text

    @server.tool()
    async def send(agent: str, text: str) -> str:
        """Put a message in another agent's inbox."""
        return (await call("send", {"agent": agent, "text": text}))["id"]

    @server.tool()
    async def done(result: Any) -> str:
        """Record your final result. Call it once, when you're finished."""
        await call("done", {"result": result})
        return "[submitted]"

    @server.tool()
    async def agents() -> str:
        """The run's agent tree with each agent's config, status and result."""
        return (await call("agents", {}))["text"]

    return server


def main() -> int:
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
