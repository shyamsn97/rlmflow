"""Sandbox-side JSON-line REPL server for minimal runtimes."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from typing import TextIO

from pydantic import ValidationError

from rflow.minimal.protocol import (
    CapabilitiesRequest,
    CapabilityMap,
    InjectImportRequest,
    InjectProxyRequest,
    InjectRequest,
    InjectSourceRequest,
    PingRequest,
    ProxyCall,
    ProxyResponse,
    ReplRequest,
    ReplResponse,
    RunRequest,
    SetEnvRequest,
    dump_message,
    parse_request,
)
from rflow.minimal.repl import DoneSignal, Repl
from rflow.minimal.tools import tool


class RemoteServer:
    def __init__(
        self,
        protocol_in: TextIO | None = None,
        protocol_out: TextIO | None = None,
    ) -> None:
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        self.repl = Repl()
        self.capabilities = CapabilityMap()
        self._next_proxy_id = 0

    def write(self, msg: ReplResponse | ProxyCall) -> None:
        self._out.write(dump_message(msg) + "\n")
        self._out.flush()

    def _proxy_id(self, name: str) -> str:
        self._next_proxy_id += 1
        return f"proxy-{self._next_proxy_id}-{name}"

    def make_proxy(self, name: str):
        def proxy(*args: object, **kwargs: object) -> object:
            proxy_id = self._proxy_id(name)
            self.write(ProxyCall(id=proxy_id, proxy=name, args=list(args), kwargs=kwargs))
            resp = ProxyResponse.model_validate_json(self._in.readline())
            if resp.done:
                if name == "done" and args:
                    print(f"[done] {args[0]}")
                raise DoneSignal()
            if not resp.ok or resp.error is not None:
                raise RuntimeError(resp.error or "proxy call failed")
            return resp.value

        return proxy

    async def handle(self, msg: ReplRequest) -> ReplResponse:
        if isinstance(msg, PingRequest):
            return ReplResponse(id=msg.id)
        if isinstance(msg, CapabilitiesRequest):
            return ReplResponse(id=msg.id, capabilities=self.capabilities)
        if isinstance(msg, RunRequest):
            output = await self.repl.run(msg.code)
            return ReplResponse(id=msg.id, output=output, errored=self.repl.errored)
        if isinstance(msg, InjectRequest):
            self.repl.namespace[msg.name] = msg.value
            return ReplResponse(id=msg.id)
        if isinstance(msg, SetEnvRequest):
            os.environ.update({str(k): str(v) for k, v in msg.values.items()})
            return ReplResponse(id=msg.id)
        if isinstance(msg, InjectProxyRequest):
            self.repl.namespace[msg.name] = self.make_proxy(msg.name)
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

    async def serve(self) -> None:
        while True:
            line = self._in.readline()
            if not line:
                return
            try:
                resp = await self.handle(parse_request(line))
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
    parser = argparse.ArgumentParser(description="minimal rflow remote REPL")
    parser.add_argument("--workdir")
    args = parser.parse_args()
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        os.chdir(args.workdir)
    asyncio.run(RemoteServer().serve())


if __name__ == "__main__":
    main()


__all__ = ["RemoteServer", "main"]
