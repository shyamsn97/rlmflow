"""Sandbox-side lightweight Python worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from rlmflow.runtime.protocol import (
    InjectProxyRequest,
    InjectRequest,
    PingRequest,
    ProxyCall,
    ProxyResponse,
    RemoveRequest,
    ReplRequest,
    ReplResponse,
    RetrieveRequest,
    RunRequest,
    WireModel,
    dump_message,
    parse_host_message,
)
from rlmflow.runtime.repl import (
    CurrentObject,
    DoneSignal,
    LocalRepl,
    ReplStatus,
    TransitionSignal,
    base_namespace,
    current_binding,
)
from rlmflow.tools.agents import AGENT_OBSERVE_TOOL, AGENT_WAIT_TOOL
from rlmflow.utils.serial import decode_host_value


class ReplServer:
    """One worker process with a shared heap and concurrent agent tenants."""

    def __init__(
        self,
        *,
        workdir: str | Path | None = None,
        protocol_in: TextIO | None = None,
        protocol_out: TextIO | None = None,
        preimports: Sequence[str] | None = None,
    ) -> None:
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        self._workdir = workdir
        # Tenants share this namespace, so preimports are paid for once per worker.
        self._namespace: dict[str, object] = base_namespace(preimports)
        self._tenants: dict[str, LocalRepl] = {}
        self._pending: dict[str, Future[ProxyResponse]] = {}
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._proxy_counter = 0

    def _tenant(self, tenant_id: str) -> LocalRepl:
        with self._state_lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                tenant = LocalRepl(self._workdir, namespace=self._namespace)
                self._tenants[tenant_id] = tenant
            return tenant

    def send(self, message: WireModel) -> None:
        with self._send_lock:
            self._out.write(dump_message(message) + "\n")
            self._out.flush()

    def _next_proxy_id(self, name: str) -> str:
        with self._state_lock:
            self._proxy_counter += 1
            return f"proxy-{self._proxy_counter}-{name}"

    def make_proxy(self, name: str, *, is_async: bool):
        def begin(args: tuple[object, ...], kwargs: dict[str, object]):
            binding = current_binding()
            if name == "finish":
                answer = None
                if len(args) == 1 and not kwargs:
                    answer = args[0]
                elif not args and set(kwargs) == {"answer"}:
                    answer = kwargs["answer"]
                if binding["structured_output"]:
                    json.dumps(answer, allow_nan=False)
                elif len(args) == 1 and not kwargs:
                    args = (str(args[0]),)
                elif not args and set(kwargs) == {"answer"}:
                    kwargs = {"answer": str(kwargs["answer"])}
            counts = binding.setdefault("_rpc_counts", {})
            call_id = counts.get(name, 0)
            counts[name] = call_id + 1
            request_id = self._next_proxy_id(name)
            future: Future[ProxyResponse] = Future()
            with self._state_lock:
                self._pending[request_id] = future
            self.send(
                ProxyCall(
                    id=request_id,
                    run_id=binding["run_id"],
                    tenant_id=binding["tenant_id"],
                    proxy=name,
                    call_id=call_id,
                    args=list(args),
                    kwargs=kwargs,
                )
            )
            return request_id, future

        def finish(request_id: str, future: Future[ProxyResponse]) -> object:
            try:
                response = future.result()
            finally:
                with self._state_lock:
                    self._pending.pop(request_id, None)
            if response.done:
                raise DoneSignal()
            if response.transition is not None:
                raise TransitionSignal(response.transition)
            if not response.ok or response.error is not None:
                raise RuntimeError(response.error or f"{name} failed")
            return decode_host_value(response.value) if response.value is not None else None

        if is_async:

            async def async_proxy(*args: object, **kwargs: object) -> object:
                request_id, future = begin(args, kwargs)
                try:
                    response = await asyncio.wrap_future(future)
                finally:
                    with self._state_lock:
                        self._pending.pop(request_id, None)
                if response.done:
                    raise DoneSignal()
                if response.transition is not None:
                    raise TransitionSignal(response.transition)
                if not response.ok or response.error is not None:
                    raise RuntimeError(response.error or f"{name} failed")
                return decode_host_value(response.value) if response.value is not None else None

            return async_proxy

        def proxy(*args: object, **kwargs: object) -> object:
            return finish(*begin(args, kwargs))

        return proxy

    def handle_control(self, message: ReplRequest) -> None:
        if isinstance(message, PingRequest):
            self.send(ReplResponse(id=message.id))
            return
        if isinstance(message, RunRequest):
            threading.Thread(
                target=self._run,
                args=(message,),
                daemon=True,
                name=f"rlmflow-run-{message.id}",
            ).start()
            return
        try:
            if isinstance(message, InjectRequest):
                self._namespace[message.name] = decode_host_value(message.value)
                response = ReplResponse(id=message.id)
            elif isinstance(message, RetrieveRequest):
                if message.name not in self._namespace:
                    response = ReplResponse(
                        id=message.id,
                        ok=False,
                        error=f"no variable named {message.name!r}",
                    )
                else:
                    response = ReplResponse(
                        id=message.id,
                        value=self._namespace[message.name],
                    )
            elif isinstance(message, RemoveRequest):
                self._namespace.pop(message.name, None)
                response = ReplResponse(id=message.id)
            elif isinstance(message, InjectProxyRequest):
                self._namespace[message.name] = self.make_proxy(
                    message.name,
                    is_async=message.is_async,
                )
                response = ReplResponse(id=message.id)
            else:
                response = ReplResponse(id=message.id, ok=False, error="unknown command")
        except BaseException as exc:  # noqa: BLE001 - control errors cross the wire
            response = ReplResponse(
                id=message.id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.send(response)

    def _run(self, message: RunRequest) -> None:
        tenant = self._tenant(message.tenant_id)
        binding = decode_host_value(message.binding)
        tenant.inputs = binding["inputs"]
        tenant.env = binding["env"]
        wait_agent = self._namespace.get(AGENT_WAIT_TOOL)
        if wait_agent is not None:
            binding["wait_agent"] = wait_agent
        observe_agent = self._namespace.get(AGENT_OBSERVE_TOOL)
        if observe_agent is not None:
            binding["observe_agent"] = observe_agent
        if binding.get("expose_agents"):
            self._namespace.setdefault("AGENTS", CurrentObject("agents"))
        try:
            run = asyncio.run(tenant.run(message.code, binding=binding))
            self.send(
                ReplResponse(
                    id=message.id,
                    output=run.output,
                    errored=run.status is ReplStatus.ERROR,
                    env=binding["env"],
                    transition=run.transition,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - keep the worker dispatcher alive
            self.send(
                ReplResponse(
                    id=message.id,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    def resolve_proxy(self, response: ProxyResponse) -> None:
        with self._state_lock:
            future = self._pending.get(response.id)
        if future is not None and not future.done():
            future.set_result(response)

    def serve_stdio(self) -> None:
        for line in self._in:
            try:
                message = parse_host_message(line)
                if isinstance(message, ProxyResponse):
                    self.resolve_proxy(message)
                else:
                    self.handle_control(message)
            except ValidationError as exc:
                self.send(
                    ReplResponse(
                        id="unknown",
                        ok=False,
                        error=f"ValidationError: {exc}",
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="rlmflow Python worker")
    parser.add_argument("--workdir")
    parser.add_argument(
        "--preimport",
        help=(
            "comma-separated modules to bind in the REPL namespace, overriding the "
            "defaults; pass an empty value to bind none"
        ),
    )
    args = parser.parse_args()
    preimports = None if args.preimport is None else [n for n in args.preimport.split(",") if n]

    # Claim private protocol descriptors before user code can inherit stdio.
    wire_in_fd = os.dup(0)
    wire_out_fd = os.dup(1)
    os.set_inheritable(wire_in_fd, False)
    os.set_inheritable(wire_out_fd, False)
    wire_in = os.fdopen(wire_in_fd, "r")
    wire_out = os.fdopen(wire_out_fd, "w", buffering=1)

    # User stdin is empty. Native stdout and subprocess output become stderr
    # diagnostics; normal Python output is captured per run.
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    ReplServer(
        workdir=args.workdir,
        protocol_in=wire_in,
        protocol_out=wire_out,
        preimports=preimports,
    ).serve_stdio()


if __name__ == "__main__":
    main()


__all__ = ["ReplServer", "main"]
