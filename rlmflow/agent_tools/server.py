"""A localhost HTTP endpoint that external harnesses reach through ``rlmflow``.

A harness opens a turn for the agent it is stepping and starts its session with
``turn.env``. Each request carries the turn's token, so it acts as that agent at
its latest node; the token stops working when the turn ends. ``/repl`` runs a
cell and lands it as a ``code`` node, which streams like any other step.

``rlmflow serve`` also hands out an agent token (``attach``), which lasts across
turns and acts for whichever turn of that agent is open, and a host token for
``/prompt``, ``/stop`` and ``/shutdown``, which it serves itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rlmflow.agent_tools import tools
from rlmflow.graph.nodes import Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph
    from rlmflow.harnesses.sessions import TurnResult

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


@dataclass
class Turn:
    """One session turn of one agent: its token and what it submitted.

    ``ended`` is for harnesses whose turn is closed from outside, by ``/stop``.
    """

    step: Step
    token: str
    env: dict[str, str] = field(default_factory=dict)
    submitted: bool = False
    result: Any = None
    ended: asyncio.Future[TurnResult] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )

    @property
    def graph(self) -> Graph:
        return self.step.graph

    @property
    def agent(self) -> Node:
        return self.step.agent


class Host(Protocol):
    """What serves ``/prompt``, ``/stop`` and ``/shutdown``."""

    async def prompt(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def stop(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def shutdown(self, body: dict[str, Any]) -> dict[str, Any]: ...


class AgentToolsServer:
    """Serves the agent tools for every open turn of one ``Flow``."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow
        self.url: str | None = None
        self._server: asyncio.Server | None = None
        self._turns: dict[str, Turn] = {}
        self._agent_tokens: dict[str, str] = {}
        self._open: dict[str, Turn] = {}
        self._cell_locks: dict[str, asyncio.Lock] = {}
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

    @asynccontextmanager
    async def turn(self, step: Step) -> AsyncIterator[Turn]:
        """Open a turn for ``step.agent``; ``turn.env`` is the session's ``RLMFLOW_*`` environment."""
        url = await self.start()
        token = secrets.token_urlsafe(24)
        turn = Turn(step, token)
        turn.env = {
            "RLMFLOW_URL": url,
            "RLMFLOW_AGENT": step.agent.id,
            "RLMFLOW_TOKEN": token,
            # `rlmflow` on the session's PATH runs this Python, installed or not.
            "PATH": f"{self._bin.name}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        self._turns[token] = turn
        self._open[step.agent.id] = turn
        try:
            yield turn
        finally:
            self._turns.pop(token, None)
            if self._open.get(step.agent.id) is turn:
                del self._open[step.agent.id]
            if not turn.ended.done():
                turn.ended.cancel()

    def open_turn(self, agent: Node | str) -> Turn | None:
        """The turn ``agent`` has open, if any."""
        return self._open.get(agent if isinstance(agent, str) else agent.id)

    def attach(self, agent: Node) -> str:
        """A token that acts for whichever turn of ``agent`` is open."""
        token = secrets.token_urlsafe(24)
        self._agent_tokens[token] = agent.id
        return token

    def serve(self, host: Host) -> str:
        """Route ``/prompt``, ``/stop`` and ``/shutdown`` to ``host``; returns their token."""
        self.host, self.host_token = host, secrets.token_urlsafe(24)
        return self.host_token

    def close(self) -> None:
        self._turns.clear()
        self._open.clear()
        self._agent_tokens.clear()
        if self._server is not None:
            self._server.close()
            self._server = None
            self.url = None
            self._bin.cleanup()

    # -- the tools ----------------------------------------------------------

    def _step(self, turn: Turn) -> Step:
        return Step(turn.graph, turn.agent, self.flow.anchor(turn.graph, turn.agent))

    async def repl(self, turn: Turn, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError(400, "repl needs non-empty code")
        lock = self._cell_locks.setdefault(turn.agent.id, asyncio.Lock())
        async with lock:
            ran = await tools.repl(self.flow, self._step(turn), code)
        if ran["submitted"]:
            turn.submitted, turn.result = True, ran["result"]
        return ran

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
            return 200, await getattr(self, name)(self._turn_for(token), body)
        except RequestError as exc:
            return exc.status, {"error": str(exc)}
        except (ValueError, TypeError, KeyError, RuntimeError, LookupError) as exc:
            return 400, {"error": str(exc)}

    def _turn_for(self, token: str) -> Turn:
        turn = self._turns.get(token)
        if turn is not None:
            return turn
        agent = self._agent_tokens.get(token)
        if agent is None:
            raise RequestError(401, "this token's turn has ended, or it was never valid")
        turn = self._open.get(agent)
        if turn is None:
            raise RequestError(409, "no open turn; send a prompt first")
        return turn


ROUTES = ("repl",)
HOST_ROUTES = ("prompt", "stop", "shutdown")


__all__ = ["HOST_ROUTES", "ROUTES", "AgentToolsServer", "Host", "Turn"]
