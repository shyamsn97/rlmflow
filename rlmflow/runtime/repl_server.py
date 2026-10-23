"""The sandbox-side Python worker and the typed JSON-line protocol it speaks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Annotated, Any, Literal, TextIO, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from rlmflow.runtime.repl import (
    AGENT_WAIT_TOOL,
    CurrentObject,
    DoneSignal,
    LocalRepl,
    ReplStatus,
    base_namespace,
    current_binding,
    encode_agent_references,
)
from rlmflow.structured import DEFAULT_OUTPUT_CODEC, as_finish_value
from rlmflow.utils import decode_host_value


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseRequest(WireModel):
    id: str
    cmd: str
    tenant_id: str


class PingRequest(BaseRequest):
    cmd: Literal["ping"] = "ping"


class RunRequest(BaseRequest):
    cmd: Literal["run"] = "run"
    code: str
    binding: str
    restoring: bool = False


class InjectRequest(BaseRequest):
    cmd: Literal["inject"] = "inject"
    name: str
    value: str


class RetrieveRequest(BaseRequest):
    cmd: Literal["retrieve"] = "retrieve"
    name: str


class RemoveRequest(BaseRequest):
    cmd: Literal["remove"] = "remove"
    name: str


class InjectProxyRequest(BaseRequest):
    cmd: Literal["inject_proxy"] = "inject_proxy"
    name: str
    is_async: bool = False


class VerifyRequest(BaseRequest):
    cmd: Literal["verify"] = "verify"
    verifier_id: str
    candidate: JsonValue


ReplRequest = Annotated[
    PingRequest
    | RunRequest
    | InjectRequest
    | RetrieveRequest
    | RemoveRequest
    | InjectProxyRequest
    | VerifyRequest,
    Field(discriminator="cmd"),
]


class ReplResponse(WireModel):
    id: str
    ok: bool = True
    output: str = ""
    errored: bool = False
    error: str | None = None
    value: JsonValue = None
    env: dict[str, JsonValue] | None = None


class ProxyCall(WireModel):
    id: str
    run_id: str
    tenant_id: str
    proxy: str
    call_id: int
    restoring: bool = False
    args: list[JsonValue]
    kwargs: dict[str, JsonValue]


class ProxyResponse(WireModel):
    id: str
    ok: bool = True
    value: str | None = None
    done: bool = False
    error: str | None = None


_REQUEST = TypeAdapter(ReplRequest)


def parse_host_message(data: str) -> ReplRequest | ProxyResponse:
    raw = TypeAdapter(dict[str, Any]).validate_json(data)
    if "cmd" in raw:
        return _REQUEST.validate_python(raw)
    return ProxyResponse.model_validate(raw)


def parse_client_message(data: str) -> ReplResponse | ProxyCall:
    raw = TypeAdapter(dict[str, Any]).validate_json(data)
    if "proxy" in raw:
        return ProxyCall.model_validate(raw)
    return ReplResponse.model_validate(raw)


def dump_message(message: WireModel) -> str:
    return message.model_dump_json(exclude_none=True)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def fingerprint(verify: Any) -> str:
    """Stable identity of a verifier callable's code.

    The host imports this too: a replayed launch must hash to the same value on
    whichever side registered the verifier.
    """
    code = getattr(verify, "__code__", None)
    return _digest(
        {
            "qualname": getattr(verify, "__qualname__", repr(verify)),
            "code": code.co_code.hex() if code is not None else repr(verify),
            "constants": repr(code.co_consts) if code is not None else "",
        }
    )


def schema_hash(schema: object) -> str | None:
    return _digest(schema) if schema is not None else None


class ReplServer:
    """One worker process with isolated, concurrent agent namespaces."""

    def __init__(
        self,
        *,
        protocol_in: TextIO | None = None,
        protocol_out: TextIO | None = None,
        preimports: Sequence[str] | None = None,
    ) -> None:
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        # Preimports are paid for once, then shallow-copied into each tenant namespace.
        self._namespace: dict[str, object] = base_namespace(preimports)
        self._tenants: dict[str, LocalRepl] = {}
        self._pending: dict[str, Future[ProxyResponse]] = {}
        self._verifiers: dict[tuple[str, str], object] = {}
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._proxy_counter = 0

    def _tenant(self, tenant_id: str) -> LocalRepl:
        with self._state_lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                tenant = LocalRepl(namespace=dict(self._namespace))
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
            if name in {"finish", "submit"}:
                if len(args) == 1 and not kwargs:
                    args = (as_finish_value(args[0]),)
                elif not args and set(kwargs) == {"answer"}:
                    kwargs = {"answer": as_finish_value(kwargs["answer"])}
            counts = binding.setdefault("_rpc_counts", {})
            call_id = counts.get(name, 0)
            counts[name] = call_id + 1
            if name in {"launch_agent", "launch_subagent"}:
                schema = kwargs.get("schema")
                if schema is not None:
                    kwargs["schema"] = DEFAULT_OUTPUT_CODEC.normalize(schema)
                verifier = kwargs.get("verify")
                if verifier is not None:
                    if not callable(verifier):
                        raise TypeError(f"{name} verify must be callable")
                    verifier_id = f"{binding['step_id']}:{name}:{call_id}"
                    with self._state_lock:
                        self._verifiers[(binding["tenant_id"], verifier_id)] = verifier
                    kwargs["verify"] = {
                        "$rlmflow_verifier": verifier_id,
                        "fingerprint": fingerprint(verifier),
                    }
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
                    restoring=bool(binding.get("restoring")),
                    args=cast(list[object], encode_agent_references(list(args))),
                    kwargs=cast(
                        dict[str, object],
                        encode_agent_references(kwargs),
                    ),
                )
            )
            return request_id, future

        def forget(request_id: str) -> None:
            with self._state_lock:
                self._pending.pop(request_id, None)

        def unwrap(response: ProxyResponse) -> object:
            if response.done:
                raise DoneSignal()
            if not response.ok or response.error is not None:
                raise RuntimeError(response.error or f"{name} failed")
            return decode_host_value(response.value) if response.value is not None else None

        if is_async:

            async def async_proxy(*args: object, **kwargs: object) -> object:
                request_id, future = begin(args, kwargs)
                try:
                    response = await asyncio.wrap_future(future)
                finally:
                    forget(request_id)
                return unwrap(response)

            return async_proxy

        def proxy(*args: object, **kwargs: object) -> object:
            request_id, future = begin(args, kwargs)
            try:
                response = future.result()
            finally:
                forget(request_id)
            return unwrap(response)

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
            tenant = self._tenant(message.tenant_id)
            if isinstance(message, InjectRequest):
                tenant.inject(message.name, decode_host_value(message.value))
                response = ReplResponse(id=message.id)
            elif isinstance(message, RetrieveRequest):
                if message.name not in tenant.namespace:
                    response = ReplResponse(
                        id=message.id,
                        ok=False,
                        error=f"no variable named {message.name!r}",
                    )
                else:
                    response = ReplResponse(
                        id=message.id,
                        value=tenant.namespace[message.name],
                    )
            elif isinstance(message, RemoveRequest):
                tenant.remove_tool(message.name)
                response = ReplResponse(id=message.id)
            elif isinstance(message, InjectProxyRequest):
                tenant.inject(
                    message.name,
                    self.make_proxy(message.name, is_async=message.is_async),
                )
                response = ReplResponse(id=message.id)
            elif isinstance(message, VerifyRequest):
                with self._state_lock:
                    verifier = self._verifiers.get((message.tenant_id, message.verifier_id))
                if verifier is None:
                    raise RuntimeError(f"verifier {message.verifier_id!r} is not registered")
                feedback = verifier(message.candidate)
                if inspect.isawaitable(feedback):
                    feedback = asyncio.run(feedback)
                if feedback is not None and not isinstance(feedback, str):
                    raise TypeError("verify must return str or None")
                response = ReplResponse(id=message.id, value=feedback)
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
        binding["restoring"] = message.restoring
        tenant.seed({}, binding["args"])
        tenant.env = binding["env"]
        wait_agent = tenant.namespace.get(AGENT_WAIT_TOOL)
        if wait_agent is not None:
            binding["wait_agent"] = wait_agent
        if binding.get("expose_agents"):
            tenant.namespace.setdefault("AGENTS", CurrentObject("agents"))
        try:
            run = asyncio.run(tenant.run(message.code, binding=binding))
            self.send(
                ReplResponse(
                    id=message.id,
                    output=run.output,
                    errored=run.status is ReplStatus.ERROR,
                    env=binding["env"],
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
        protocol_in=wire_in,
        protocol_out=wire_out,
        preimports=preimports,
    ).serve_stdio()


if __name__ == "__main__":
    main()


__all__ = [
    "InjectProxyRequest",
    "InjectRequest",
    "PingRequest",
    "ProxyCall",
    "ProxyResponse",
    "RemoveRequest",
    "ReplRequest",
    "ReplResponse",
    "ReplServer",
    "RetrieveRequest",
    "RunRequest",
    "VerifyRequest",
    "WireModel",
    "dump_message",
    "fingerprint",
    "main",
    "parse_client_message",
    "parse_host_message",
    "schema_hash",
]
