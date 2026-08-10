"""Sandbox-side server for the remote REPL protocol."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import threading
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from rlmflow.runtime.protocol import (
    CapabilitiesRequest,
    CapabilityMap,
    InjectImportRequest,
    InjectProxyRequest,
    InjectRequest,
    InjectSourceRequest,
    PingRequest,
    ProxyCall,
    ProxyResponse,
    RemoveRequest,
    ReplRequest,
    ReplResponse,
    RetrieveRequest,
    RunRequest,
    SetEnvRequest,
    dump_message,
    parse_host_message,
    parse_request,
)
from rlmflow.runtime.repl import DoneSignal, LocalRepl, ReplStatus
from rlmflow.tools import tool
from rlmflow.utils.serial import (
    CLOUDPICKLE,
    cloudpickle_available,
    decode_object,
    encode_object,
    is_json_safe,
)

#: How many trailing lines of escaped fd-1 output to keep per run.
FD_CAPTURE_LINES = 200


class WireGuard:
    """Give the protocol its own file descriptor, so agent code cannot reach it.

    The wire is fd 1 by default, and ``LocalRepl``'s ``_Stdout`` shim only
    intercepts Python-level writes. A subprocess inheriting fd 1, an
    ``os.write(1, ...)``, or a C extension writes straight onto the protocol and
    destroys the connection. So: ``dup`` the real pipe to a descriptor nothing
    can name, then point fd 1 at a pipe we own and drain. After this, fd 1 is a
    capture channel and the wire has exactly one writer.
    """

    def __init__(self) -> None:
        self.wire: TextIO | None = None
        self.escaped: deque[str] = deque(maxlen=FD_CAPTURE_LINES)
        self._reader: threading.Thread | None = None
        self._marker: str | None = None
        self._seen = threading.Event()
        self._sync_n = 0

    def install(self) -> TextIO:
        wire_fd = os.dup(1)
        self.wire = os.fdopen(wire_fd, "w", buffering=1)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.close(write_fd)
        # Rebind sys.stdout to the capture pipe so Python-level writes that miss
        # the ``_Stdout`` shim (a bare ``sys.__stdout__.write``) land here too.
        sys.stdout = os.fdopen(os.dup(1), "w", buffering=1)
        self._reader = threading.Thread(
            target=self._pump, args=(read_fd,), daemon=True, name="fd1 capture"
        )
        self._reader.start()
        return self.wire

    def _pump(self, read_fd: int) -> None:
        with suppress(Exception), os.fdopen(read_fd, "r") as stream:
            for line in stream:
                text = line.rstrip("\n")
                if self._marker is not None and text == self._marker:
                    self._seen.set()
                    continue
                self.escaped.append(text)

    def take(self, *, timeout: float = 0.5) -> str:
        """Drain what escaped to fd 1 since the last call.

        A subprocess's bytes are in the pipe by the time it exits, but our reader
        may not have consumed them yet. Writing a marker and waiting for the
        reader to reach it makes "everything that escaped during this run" exact
        rather than a race against a sleep.
        """
        if self._reader is not None:
            self._sync_n += 1
            self._marker = f"__rlmflow_fd1_sync_{self._sync_n}__"
            self._seen.clear()
            with suppress(Exception):
                os.write(1, (self._marker + "\n").encode())
                self._seen.wait(timeout)
            self._marker = None
        lines, self.escaped = list(self.escaped), deque(maxlen=FD_CAPTURE_LINES)
        return "\n".join(lines)


class ReplServer:
    """One deployed protocol server around one stateful REPL."""

    def __init__(
        self,
        *,
        workdir: str | Path | None = None,
        protocol_in: TextIO | None = None,
        protocol_out: TextIO | None = None,
        wire_guard: WireGuard | None = None,
    ) -> None:
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        self.wire_guard = wire_guard
        self.repl = LocalRepl(working_directory=workdir)
        self.capabilities = CapabilityMap(cloudpickle=cloudpickle_available())
        self._next_proxy_id = 0
        # Requests that arrived while a proxy call was waiting for its answer.
        # They cannot be handled at that point (the only thread is inside the
        # block), so they queue here for the serve loop to take next.
        self._held: deque[ReplRequest] = deque()

    def write(self, msg: ReplResponse | ProxyCall) -> None:
        self._out.write(dump_message(msg) + "\n")
        self._out.flush()

    def _proxy_id(self, name: str) -> str:
        self._next_proxy_id += 1
        return f"proxy-{self._next_proxy_id}-{name}"

    def _with_escaped(self, output: str) -> str:
        """Fold anything that escaped to fd 1 into the run's own output.

        Without the guard this text would have corrupted the protocol; with it,
        a subprocess that prints is simply visible to the agent.
        """
        if self.wire_guard is None:
            return output
        escaped = self.wire_guard.take()
        if not escaped:
            return output
        return f"{output}\n{escaped}".strip() if output else escaped

    def _proxy_response(self) -> ProxyResponse:
        """Read past anything that is not the answer to the call parked here.

        The host may send an ordinary request at any time, including while agent
        code sits inside a proxy call. Parsing every line as a ``ProxyResponse``
        would fail the tool for a reason unrelated to the agent's code, so a
        request is held instead and answered once the block finishes.
        """
        while True:
            line = self._in.readline()
            if not line:
                raise RuntimeError("the host closed the connection mid-proxy-call")
            msg = parse_host_message(line)
            if isinstance(msg, ProxyResponse):
                return msg
            self._held.append(msg)

    def make_proxy(self, name: str, *, is_async: bool = False):
        def proxy(*args: object, **kwargs: object) -> object:
            proxy_id = self._proxy_id(name)
            self.write(ProxyCall(id=proxy_id, proxy=name, args=list(args), kwargs=kwargs))
            resp = self._proxy_response()
            if resp.done:
                if name in {"finish", "done"} and args:
                    print(f"[{name}] {args[0]}")
                raise DoneSignal()
            if not resp.ok or resp.error is not None:
                raise RuntimeError(resp.error or "proxy call failed")
            return resp.value

        if not is_async:
            return proxy

        # Awaitable in the sandbox for async proxy tools (e.g. launch_subagents,
        # llm_query_batched); the body is a synchronous host round-trip.
        async def async_proxy(*args: object, **kwargs: object) -> object:
            return proxy(*args, **kwargs)

        return async_proxy

    async def handle(self, msg: ReplRequest) -> ReplResponse:
        if isinstance(msg, PingRequest):
            return ReplResponse(id=msg.id)
        if isinstance(msg, CapabilitiesRequest):
            return ReplResponse(id=msg.id, capabilities=self.capabilities)
        if isinstance(msg, RunRequest):
            # The answer travels back over the proxied completion tool, not this response,
            # so the client classifies the run on its side.
            run = await self.repl.run(msg.code)
            return ReplResponse(
                id=msg.id,
                output=self._with_escaped(run.output),
                errored=run.status is ReplStatus.ERROR,
                env=dict(self.repl.env),
            )
        if isinstance(msg, InjectRequest):
            value = decode_object(msg.value) if msg.encoding == CLOUDPICKLE else msg.value
            self.repl.namespace[msg.name] = value
            return ReplResponse(id=msg.id)
        if isinstance(msg, RetrieveRequest):
            if msg.name not in self.repl.namespace:
                return ReplResponse(id=msg.id, ok=False, error=f"no variable named {msg.name!r}")
            value = self.repl.namespace[msg.name]
            # Send plain data as JSON; wrap anything else as a cloudpickle blob.
            if is_json_safe(value):
                return ReplResponse(id=msg.id, value=value)
            return ReplResponse(id=msg.id, value=encode_object(value), value_encoding=CLOUDPICKLE)
        if isinstance(msg, RemoveRequest):
            self.repl.namespace.pop(msg.name, None)
            return ReplResponse(id=msg.id)
        if isinstance(msg, SetEnvRequest):
            self.repl.env.update(msg.values)
            return ReplResponse(id=msg.id)
        if isinstance(msg, InjectProxyRequest):
            self.repl.namespace[msg.name] = self.make_proxy(msg.name, is_async=msg.is_async)
            return ReplResponse(id=msg.id)
        if isinstance(msg, InjectImportRequest):
            target = importlib.import_module(msg.module)
            for part in msg.qualname.split("."):
                target = getattr(target, part)
            self.repl.namespace[msg.name] = target
            return ReplResponse(id=msg.id)
        if isinstance(msg, InjectSourceRequest):
            scope = dict(self.repl.namespace)
            scope.setdefault("tool", tool)
            exec(msg.source, scope)  # noqa: S102 - trusted host-shipped source
            self.repl.namespace[msg.name] = scope[msg.func_name]
            return ReplResponse(id=msg.id)
        return ReplResponse(id=msg.id, ok=False, error=f"unknown command: {msg.cmd!r}")

    async def serve_stdio(self) -> None:
        while True:
            try:
                if self._held:
                    request = self._held.popleft()
                else:
                    line = self._in.readline()
                    if not line:
                        return
                    request = parse_request(line)
                resp = await self.handle(request)
            except ValidationError as exc:
                resp = ReplResponse(
                    id="unknown",
                    ok=False,
                    error=f"ValidationError: {exc}",
                    errored=True,
                )
            except Exception as exc:  # noqa: BLE001
                resp = ReplResponse(
                    id="unknown",
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    errored=True,
                )
            self.write(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="rlmflow remote REPL")
    parser.add_argument("--workdir")
    args = parser.parse_args()
    # Vacate fd 1 before anything can write to it, so the protocol has exactly
    # one writer for the life of the process.
    guard = WireGuard()
    wire = guard.install()
    server = ReplServer(workdir=args.workdir, protocol_out=wire, wire_guard=guard)
    asyncio.run(server.serve_stdio())


if __name__ == "__main__":
    main()


__all__ = ["FD_CAPTURE_LINES", "ReplServer", "WireGuard", "main"]
