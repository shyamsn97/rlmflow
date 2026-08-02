"""Sandbox-side server for the minimal remote REPL protocol."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
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
    parse_request,
)
from rlmflow.runtime.repl import DoneSignal, LocalRepl
from rlmflow.tools import tool
from rlmflow.utils.serial import (
    CLOUDPICKLE,
    cloudpickle_available,
    decode_object,
    encode_object,
    is_json_safe,
)


class ReplServer:
    """One deployed protocol server around one stateful REPL."""

    def __init__(
        self,
        *,
        workdir: str | Path | None = None,
        protocol_in: TextIO | None = None,
        protocol_out: TextIO | None = None,
    ) -> None:
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        self.repl = LocalRepl(working_directory=workdir)
        self.capabilities = CapabilityMap(cloudpickle=cloudpickle_available())
        self._next_proxy_id = 0

    def write(self, msg: ReplResponse | ProxyCall) -> None:
        self._out.write(dump_message(msg) + "\n")
        self._out.flush()

    def _proxy_id(self, name: str) -> str:
        self._next_proxy_id += 1
        return f"proxy-{self._next_proxy_id}-{name}"

    def make_proxy(self, name: str, *, is_async: bool = False):
        def proxy(*args: object, **kwargs: object) -> object:
            proxy_id = self._proxy_id(name)
            self.write(
                ProxyCall(id=proxy_id, proxy=name, args=list(args), kwargs=kwargs)
            )
            resp = ProxyResponse.model_validate_json(self._in.readline())
            if resp.done:
                if name == "done" and args:
                    print(f"[done] {args[0]}")
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
            # The answer travels back over the proxied ``done``, not this response,
            # so the client classifies the run on its side.
            run = await self.repl.run(msg.code)
            return ReplResponse(
                id=msg.id,
                output=run.output,
                errored=self.repl.errored,
                env=dict(self.repl.env),
            )
        if isinstance(msg, InjectRequest):
            value = (
                decode_object(msg.value) if msg.encoding == CLOUDPICKLE else msg.value
            )
            self.repl.namespace[msg.name] = value
            return ReplResponse(id=msg.id)
        if isinstance(msg, RetrieveRequest):
            if msg.name not in self.repl.namespace:
                return ReplResponse(
                    id=msg.id, ok=False, error=f"no variable named {msg.name!r}"
                )
            value = self.repl.namespace[msg.name]
            # Send plain data as JSON; wrap anything else as a cloudpickle blob.
            if is_json_safe(value):
                return ReplResponse(id=msg.id, value=value)
            return ReplResponse(
                id=msg.id, value=encode_object(value), value_encoding=CLOUDPICKLE
            )
        if isinstance(msg, RemoveRequest):
            self.repl.namespace.pop(msg.name, None)
            return ReplResponse(id=msg.id)
        if isinstance(msg, SetEnvRequest):
            self.repl.env.update(msg.values)
            return ReplResponse(id=msg.id)
        if isinstance(msg, InjectProxyRequest):
            self.repl.namespace[msg.name] = self.make_proxy(
                msg.name, is_async=msg.is_async
            )
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
    parser = argparse.ArgumentParser(description="minimal rlmflow remote REPL")
    parser.add_argument("--workdir")
    args = parser.parse_args()
    asyncio.run(ReplServer(workdir=args.workdir).serve_stdio())


if __name__ == "__main__":
    main()


__all__ = ["ReplServer", "main"]
