"""A localhost HTTP endpoint that CLI agents reach through ``rlmflow repl``.

Each run gets a token the first time its ``Repl.environ`` is read. A request
with it acts as that run at its latest node, and only while its agent is
taking a step: ``/repl`` runs the code as a new node. ``rlmflow serve`` also
gets a host token for ``/prompt``, ``/stop`` and ``/shutdown``, which it serves
itself, and ends a ``Person``'s step with ``end_step``.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rlmflow.graph.nodes import Node

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph

REASONS = {
    200: "OK",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    409: "Conflict",
    500: "Internal Server Error",
}


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Host(Protocol):
    """What serves ``/prompt``, ``/stop`` and ``/shutdown``."""

    async def prompt(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def stop(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def shutdown(self, body: dict[str, Any]) -> dict[str, Any]: ...


def _root_key(node: Node) -> tuple[Node, tuple[str, str]]:
    root = node.root
    return root, (root._graph().id, root.id)


class AgentToolsServer:
    """Serves the agent tools for every run of one ``Flow``."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow
        self.url: str | None = None
        self._server: asyncio.Server | None = None
        #: token -> (graph, run id), and each run's token by (graph id, run id).
        self._runs: dict[str, tuple[Graph, str]] = {}
        self._tokens: dict[tuple[str, str], str] = {}
        #: A ``Person`` step waiting for ``/stop``, by (graph id, run id).
        self._waiting: dict[tuple[str, str], asyncio.Future[Node]] = {}
        self.host: Host | None = None
        self.host_token: str | None = None

    async def start(self) -> str:
        if self._server is None:
            self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
            port = self._server.sockets[0].getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            self._bin = tempfile.TemporaryDirectory(prefix="rlmflow-bin-")
            shim = Path(self._bin.name) / "rlmflow"
            package_root = shlex.quote(str(Path(__file__).resolve().parents[2]))
            shim.write_text(
                f"#!/bin/sh\nPYTHONPATH={package_root}${{PYTHONPATH:+:$PYTHONPATH}} "
                f'exec {shlex.quote(sys.executable)} -m rlmflow "$@"\n'
            )
            shim.chmod(0o755)
        return self.url

    def token(self, run: Node) -> str:
        """``run``'s token, made on first use."""
        root, key = _root_key(run)
        if key not in self._tokens:
            token = secrets.token_urlsafe(24)
            self._tokens[key] = token
            self._runs[token] = (root._graph(), root.id)
        return self._tokens[key]

    def environ(self, run: Node) -> dict[str, str]:
        """The ``RLMFLOW_*`` environment a CLI agent's process gets for ``run``."""
        if self.url is None:
            raise RuntimeError("the agent-tools server isn't started; await start() first")
        return {
            "RLMFLOW_URL": self.url,
            "RLMFLOW_AGENT": run.root.id,
            "RLMFLOW_TOKEN": self.token(run),
            # `rlmflow` on the session's PATH runs this Python, installed or not.
            "PATH": f"{self._bin.name}{os.pathsep}{os.environ.get('PATH', '')}",
        }

    async def turn_ended(self, run: Node) -> Node:
        """Wait for ``end_step(run, ...)``: what a ``Person``'s session reported when its turn ended."""
        _, key = _root_key(run)
        future = self._waiting[key] = asyncio.get_running_loop().create_future()
        try:
            return await future
        finally:
            if self._waiting.get(key) is future:
                del self._waiting[key]

    def waiting(self, run: Node) -> bool:
        """Whether ``run``'s step is open, waiting for its session's turn to end."""
        _, key = _root_key(run)
        future = self._waiting.get(key)
        return future is not None and not future.done()

    def end_step(self, run: Node, node: Node) -> bool:
        """End ``run``'s open step with ``node``; whether one was open."""
        _, key = _root_key(run)
        future = self._waiting.get(key)
        if future is None or future.done():
            return False
        future.set_result(node)
        return True

    def serve(self, host: Host) -> str:
        """Route ``/prompt``, ``/stop`` and ``/shutdown`` to ``host``; returns their token."""
        self.host, self.host_token = host, secrets.token_urlsafe(24)
        return self.host_token

    def close(self) -> None:
        self._runs.clear()
        self._tokens.clear()
        for future in self._waiting.values():
            if not future.done():
                future.cancel()
        self._waiting.clear()
        if self._server is not None:
            self._server.close()
            self._server = None
            self.url = None
            self._bin.cleanup()

    # -- the tools ----------------------------------------------------------

    async def repl(self, run: Node, body: dict[str, Any]) -> dict[str, Any]:
        """Run ``code`` as a new node after the run's last node."""
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError(400, "repl needs non-empty code")
        g = run._graph()
        node = self.flow.append(g, run, Node("assistant", code=code))
        await self.flow.repl(run).exec(node)
        submitted = node.result is not None
        return {
            "output": node.output,
            "error": node.error,
            "submitted": submitted,
            "result": node.result if submitted else None,
        }

    # -- HTTP -----------------------------------------------------------------

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            status, payload = await self._respond(reader)
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        data = json.dumps(payload, default=str).encode()
        head = (
            f"HTTP/1.1 {status} {REASONS[status]}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n"
        )
        try:
            writer.write(head.encode() + data)
            await writer.drain()
        except ConnectionError:
            pass  # the caller hung up; there is no one to tell
        finally:
            writer.close()

    async def _respond(self, reader: asyncio.StreamReader) -> tuple[int, dict[str, Any]]:
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
        method, path = head[0].split(" ")[:2]
        headers = {
            name.strip().lower(): value.strip()
            for name, value in (line.split(":", 1) for line in head[1:] if ":" in line)
        }
        raw = await reader.readexactly(int(headers.get("content-length") or 0))
        try:
            token = headers.get("authorization", "").removeprefix("Bearer ")
            name = path.strip("/")
            if method != "POST" or name not in (*ROUTES, *HOST_ROUTES):
                raise RequestError(404, f"no tool at {method} {path}")
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise RequestError(400, "the request body must be a JSON object")
            if name in HOST_ROUTES:
                if self.host is None or not token or token != self.host_token:
                    raise RequestError(401, "this route takes the serve token")
                return 200, await getattr(self.host, name)(body)
            return 200, await getattr(self, name)(self._run_for(token), body)
        except RequestError as exc:
            return exc.status, {"error": str(exc)}
        except (ValueError, TypeError, KeyError, RuntimeError, LookupError) as exc:
            return 400, {"error": str(exc)}

    def _run_for(self, token: str) -> Node:
        found = self._runs.get(token)
        if found is None:
            raise RequestError(401, "this token was never valid, or the server restarted")
        g, run_id = found
        if run_id not in g:
            raise RequestError(401, "this token's run was removed")
        run = g.agent_root(run_id)
        if not self.flow.stepping(run):
            raise RequestError(409, "this run's turn has ended; it isn't taking a step now")
        return run


ROUTES = ("repl",)
HOST_ROUTES = ("prompt", "stop", "shutdown")


__all__ = ["HOST_ROUTES", "ROUTES", "AgentToolsServer", "Host"]
