"""Use Flow with a real MCP weather server.

Requires MCP and a live LLM client:

    export OPENAI_API_KEY=...
    pip install -e ".[openai,mcp]"
    python examples/providers/mcp_weather.py
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from rlmflow import (
    AgentConfig,
    AgentStart,
    Flow,
    LocalRuntime,
    tool,
)
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream  # noqa: E402

DEFAULT_QUERY = """I will be in Seattle today, Austin 3 days after that, and San Francisco 5 days after that. Check the weather and tell me what to pack for each city.
"""


class MCPStdioClient:
    """Small synchronous wrapper around an MCP stdio client session."""

    def __init__(self, command: str, args: list[str]) -> None:
        self.command = command
        self.args = args
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-weather", daemon=True)
        self._queue: asyncio.Queue[tuple[str, Any, Future]] | None = None
        self._startup_error: BaseException | None = None
        self._session: Any = None
        self._closed = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._worker())
        self._loop.run_forever()

    def start(self) -> MCPStdioClient:
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self._thread.join(timeout=5)
            self._loop.close()
            raise self._startup_error
        return self

    async def _worker(self) -> None:
        close_future: Future | None = None
        current_future: Future | None = None
        try:
            from mcp import (  # type: ignore[reportMissingImports]
                ClientSession,
                StdioServerParameters,
            )
            from mcp.client.stdio import (
                stdio_client,
            )  # type: ignore[reportMissingImports]
        except ImportError:  # pragma: no cover - exercised by example smoke skips
            self._startup_error = RuntimeError(
                "The MCP weather example requires the `mcp` package. "
                'Install with `pip install -e ".[mcp]"`.'
            )
            self._ready.set()
            self._loop.call_soon(self._loop.stop)
            return

        params = StdioServerParameters(command=self.command, args=self.args)
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    self._session = session
                    await session.initialize()
                    self._queue = asyncio.Queue()
                    self._ready.set()

                    while True:
                        op, payload, future = await self._queue.get()
                        current_future = future
                        if op == "close":
                            close_future = future
                            break
                        try:
                            if op == "list_tools":
                                future.set_result(await session.list_tools())
                            elif op == "call_tool":
                                name, arguments = payload
                                future.set_result(await session.call_tool(name, arguments))
                            else:
                                future.set_exception(RuntimeError(f"Unknown MCP client op: {op}"))
                        except BaseException as exc:  # noqa: BLE001
                            future.set_exception(exc)
                        finally:
                            current_future = None
        except BaseException as exc:  # noqa: BLE001
            if self._queue is None:
                self._startup_error = exc
                self._ready.set()
            elif current_future is not None and not current_future.done():
                current_future.set_exception(exc)
            elif close_future is not None and not close_future.done():
                close_future.set_exception(exc)
        else:
            if close_future is not None and not close_future.done():
                close_future.set_result(None)
        finally:
            self._session = None
            self._loop.call_soon(self._loop.stop)

    def list_tools(self) -> list[Any]:
        result = self._submit("list_tools")
        return list(getattr(result, "tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = self._submit("call_tool", (name, arguments))
        if getattr(result, "isError", False):
            raise RuntimeError(_tool_result_text(result))
        return _tool_result_payload(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._queue is not None:
                self._submit("close")
        finally:
            self._thread.join(timeout=5)
            self._loop.close()

    def _submit(self, op: str, payload: Any = None) -> Any:
        if self._startup_error is not None:
            raise self._startup_error
        if self._queue is None:
            raise RuntimeError("MCP client is not started")
        future: Future = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (op, payload, future))
        return future.result(timeout=30)


def _tool_result_text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        elif hasattr(item, "model_dump"):
            parts.append(json.dumps(item.model_dump()))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _tool_result_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    text = _tool_result_text(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _schema_for(spec: Any) -> dict[str, Any]:
    schema = getattr(spec, "inputSchema", None)
    if schema is None:
        schema = getattr(spec, "input_schema", None)
    return schema or {}


def _annotation_for(json_type: str | list[str] | None) -> object:
    if isinstance(json_type, list):
        json_type = next((item for item in json_type if item != "null"), None)
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(json_type or "", inspect.Signature.empty)


def _signature_from_schema(schema: dict[str, Any]) -> inspect.Signature:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    for name, prop in properties.items():
        default = inspect.Parameter.empty if name in required else prop.get("default", None)
        params.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=_annotation_for(prop.get("type")),
            )
        )
    return inspect.Signature(params)


def _description_for(spec: Any) -> str:
    description = getattr(spec, "description", None) or f"Call MCP tool {spec.name}."
    schema = _schema_for(spec)
    if schema:
        description += "\n\nMCP input schema:\n" + json.dumps(schema, indent=2, sort_keys=True)
    return description


def make_mcp_tool(client: MCPStdioClient, spec: Any):
    @tool(_description_for(spec), name=spec.name)
    def mcp_tool(**kwargs):
        return client.call_tool(spec.name, kwargs)

    mcp_tool.__name__ = spec.name
    mcp_tool.__signature__ = _signature_from_schema(_schema_for(spec))  # type: ignore[attr-defined]
    return mcp_tool


def mcp_tools(client: MCPStdioClient) -> list[Any]:
    """Build the MCP server's tools as callables, each named by its MCP spec."""
    return [make_mcp_tool(client, spec) for spec in client.list_tools()]


async def run_until_done(flow: Flow, root: AgentStart, out_dir: Path) -> AgentStart:
    async for node in checkpoint_stream(flow.run_streaming(root), out_dir):
        print(f"{node.parent_agent.config.path}  {node.type}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow + MCP weather example")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "mcp-weather"),
        help="Save the final run here (default: examples/_runs/mcp-weather/).",
    )
    args = parser.parse_args()

    server_script = Path(__file__).with_name("mcp_weather_server.py")
    mcp_client = MCPStdioClient(sys.executable, [str(server_script)]).start()

    try:
        # Expose the MCP-backed tools to every agent by handing them to the Flow
        # (each is already named by its MCP spec).
        flow = Flow(
            client_for(args.model),
            runtime=LocalRuntime(),
            tools=mcp_tools(mcp_client),
            root_config=AgentConfig(max_depth=args.max_depth, max_iters=args.max_iters),
        )
        root = flow.start(args.query)
        out_dir = Path(args.out_dir)
        asyncio.run(run_until_done(flow, root, out_dir))

        print(f"\n{'=' * 60}\nWEATHER PACKING RECOMMENDATION\n{'=' * 60}")
        print(root.result())
        print(f"\nRun saved to {out_dir}")

        flow.runtime.close_repls()
    finally:
        mcp_client.close()


if __name__ == "__main__":
    main()
