"""The agent's REPL as a stdio MCP server with one tool, ``python``.

A harness that speaks MCP launches ``python -m rlmflow.agent_tools.mcp`` with the
turn's ``RLMFLOW_*`` environment; the Claude Code plugin launches ``rlmflow mcp``,
which finds ``rlmflow serve`` through ``.rlmflow/serve.json``. ``python`` is one
request to the agent-tools server ``rlmflow repl`` uses, so each call lands a ``code`` node.
"""

from __future__ import annotations

import asyncio
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

    return server


def main() -> int:
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
