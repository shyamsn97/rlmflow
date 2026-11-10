"""Expose an MCP server's tools to agents.

    flow = Flow(configs=[...], tools=[MCPTools("python", "server.py")])

The Flow starts the server when a run begins and stops it when the run ends.
Requires ``pip install "rlmflow[mcp]"``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from rlmflow.tools import tool_from_schema


class MCPTools:
    """Every tool on an MCP stdio server, as a ``Toolkit``.

    ``prefix`` is prepended to each tool name, to keep two servers' tools apart.
    """

    def __init__(
        self,
        command: str,
        *args: str,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        prefix: str = "",
    ) -> None:
        self.command = command
        self.args = list(args)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self.prefix = prefix
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    async def open(self) -> list[Any]:
        try:
            from mcp import StdioServerParameters
        except ImportError as exc:
            raise ImportError(
                'MCPTools requires the `mcp` package: pip install "rlmflow[mcp]"'
            ) from exc
        params = StdioServerParameters(
            command=self.command, args=self.args, env=self.env, cwd=self.cwd
        )
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._serve(params, ready))
        session = await ready
        listing = await session.list_tools()
        return [self._tool(session, spec) for spec in listing.tools]

    async def close(self) -> None:
        if self._task is None:
            return
        assert self._stop is not None
        self._stop.set()
        task, self._task = self._task, None
        await task

    async def _serve(self, params: Any, ready: asyncio.Future) -> None:
        # The stdio and session contexts must exit in the task that entered them.
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        assert self._stop is not None
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if ready.done():
                raise
            ready.set_exception(exc)

    def _tool(self, session: Any, spec: Any) -> Any:
        async def call(arguments: dict[str, Any]) -> Any:
            result = await session.call_tool(spec.name, arguments)
            if _field(result, "is_error", "isError"):
                raise RuntimeError(_text(result) or f"MCP tool {spec.name} failed")
            structured = _field(result, "structured_content", "structuredContent")
            if structured is not None:
                return structured
            text = _text(result)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        return tool_from_schema(
            self.prefix + spec.name,
            spec.description or f"Call MCP tool {spec.name}.",
            _field(spec, "input_schema", "inputSchema"),
            call,
        )


def _field(value: Any, name: str, legacy: str) -> Any:
    # mcp 2.x renamed the 1.x camelCase fields.
    return getattr(value, name, getattr(value, legacy, None))


def _text(result: Any) -> str:
    parts = []
    for item in result.content or []:
        text = getattr(item, "text", None)
        parts.append(text if text is not None else json.dumps(item.model_dump()))
    return "\n".join(parts)


__all__ = ["MCPTools"]
