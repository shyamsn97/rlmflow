"""Host-side client for lightweight Python workers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
import uuid
from collections.abc import Callable
from contextvars import Context, ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from rlmflow.graph.nodes import ExecAction, Node, active_step, running_step
from rlmflow.runtime.protocol import (
    InjectProxyRequest,
    InjectRequest,
    PingRequest,
    ProxyCall,
    ProxyResponse,
    RemoveRequest,
    ReplResponse,
    RetrieveRequest,
    RunRequest,
    WireModel,
)
from rlmflow.runtime.repl import MISSING_REPL_NOTE, DoneSignal, Repl, ReplRun, ReplStatus
from rlmflow.tools import get_tool_metadata
from rlmflow.tools.agents import AGENTS_BINDING
from rlmflow.utils.serial import encode_host_value

_NO_ANSWER = object()
_RPC_CALL_ID: ContextVar[int] = ContextVar("rlmflow_rpc_call_id", default=0)


def current_rpc_call_id() -> int:
    """Ordinal of the current proxied call within one execution."""
    return _RPC_CALL_ID.get()


class ReplConnection(Protocol):
    def start(self) -> None: ...

    def send(self, message: WireModel) -> None: ...

    def recv(self) -> ReplResponse | ProxyCall: ...

    def close(self, *, force: bool = False) -> None: ...

    def stderr_tail(self) -> str: ...


class ProtocolDesyncError(RuntimeError):
    """A worker response matched no outstanding request."""


@dataclass
class _RunState:
    tenant: WorkerRepl
    loop: asyncio.AbstractEventLoop
    step: Node | None
    proxied: dict[str, Callable[..., object]]
    answer: Any = _NO_ANSWER


class WorkerSession:
    """One worker connection shared by one or more logical agent tenants."""

    def __init__(
        self,
        connection: ReplConnection,
        *,
        timeout: float = 300.0,
        execution_timeout: float | None = None,
    ) -> None:
        self.connection = connection
        self.timeout = timeout
        self.execution_timeout = execution_timeout
        self.tenants: dict[str, WorkerRepl] = {}
        self.runs: dict[str, _RunState] = {}
        self.pending: dict[str, concurrent.futures.Future[ReplResponse]] = {}
        self.closed = False
        self._started = False
        self._counter = 0
        self._execution_order = 0
        self._start_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._failure: BaseException | None = None

    def add_tenant(self, tenant: WorkerRepl) -> None:
        with self._state_lock:
            if self.closed:
                raise RuntimeError("worker session is closed")
            self.tenants[tenant.tenant_id] = tenant

    def release(self, tenant: WorkerRepl) -> None:
        with self._state_lock:
            self.tenants.pop(tenant.tenant_id, None)
            empty = not self.tenants
        if empty:
            self.close()

    def _next_id(self, kind: str) -> str:
        with self._state_lock:
            self._counter += 1
            return f"{kind}-{self._counter}"

    def ensure_started(self) -> None:
        with self._start_lock:
            if self.closed:
                raise RuntimeError("worker session is closed")
            if self._started:
                if self._failure is not None:
                    raise RuntimeError(f"worker failed: {self._failure}")
                return
            self.connection.start()
            self._reader = threading.Thread(
                target=self._read_messages,
                daemon=True,
                name=f"rlmflow-worker-reader-{id(self):x}",
            )
            self._reader.start()
            self._started = True
        probe = PingRequest(id=self._next_id("ping"), tenant_id="__session__")
        self.request(probe, timeout=self.timeout, start=False)

    def request(
        self,
        message: WireModel,
        *,
        timeout: float | None,
        start: bool = True,
    ) -> ReplResponse:
        if start:
            self.ensure_started()
        future: concurrent.futures.Future[ReplResponse] = concurrent.futures.Future()
        with self._state_lock:
            if self.closed:
                raise RuntimeError("worker session is closed")
            self.pending[message.id] = future
        try:
            with self._send_lock:
                self.connection.send(message)
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self.close(force=True)
            kind = getattr(message, "cmd", "request")
            if kind == "run":
                raise TimeoutError(f"REPL execution exceeded {timeout}s") from None
            raise TimeoutError(f"REPL {kind} did not respond within {timeout}s") from None
        finally:
            with self._state_lock:
                self.pending.pop(message.id, None)

    def _read_messages(self) -> None:
        try:
            while not self.closed:
                message = self.connection.recv()
                if isinstance(message, ProxyCall):
                    threading.Thread(
                        target=self._handle_proxy,
                        args=(message,),
                        daemon=True,
                        name=f"rlmflow-host-call-{message.id}",
                    ).start()
                    continue
                with self._state_lock:
                    future = self.pending.get(message.id)
                if future is None:
                    raise ProtocolDesyncError(
                        f"response {message.id!r} matches no outstanding request"
                    )
                if not future.done():
                    future.set_result(message)
        except BaseException as exc:  # noqa: BLE001 - reader death fails the session
            if not self.closed:
                self._fail(exc)

    def _fail(self, error: BaseException) -> None:
        with self._state_lock:
            self._failure = error
            futures = tuple(self.pending.values())
        for future in futures:
            if not future.done():
                future.set_exception(error)

    def _handle_proxy(self, message: ProxyCall) -> None:
        with self._state_lock:
            state = self.runs.get(message.run_id)
        if state is None:
            self._send_proxy_error(message, "RPC belongs to no active execution")
            return
        try:

            def invoke() -> Any:
                token = _RPC_CALL_ID.set(message.call_id)
                try:
                    fn = state.proxied[message.proxy]
                    value = fn(*message.args, **message.kwargs)
                    if inspect.isawaitable(value):
                        value = self._await_host(value, state.loop)
                    return value
                finally:
                    _RPC_CALL_ID.reset(token)

            def in_step() -> Any:
                if state.step is None:
                    return invoke()
                with running_step(state.step):
                    return invoke()

            value = Context().run(in_step)
        except DoneSignal as signal:
            state.answer = signal.answer
            response = ProxyResponse(id=message.id, done=True)
        except BaseException as exc:  # noqa: BLE001 - host tool failures cross the wire
            response = ProxyResponse(
                id=message.id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            response = ProxyResponse(id=message.id, value=encode_host_value(value))
        with self._send_lock:
            self.connection.send(response)

    def _send_proxy_error(self, message: ProxyCall, error: str) -> None:
        with self._send_lock:
            self.connection.send(ProxyResponse(id=message.id, ok=False, error=error))

    @staticmethod
    def _await_host(value: Any, loop: asyncio.AbstractEventLoop) -> Any:
        async def wait() -> Any:
            return await value

        return asyncio.run_coroutine_threadsafe(wait(), loop).result()

    def execute(self, tenant: WorkerRepl, code: str, state: _RunState) -> ReplRun:
        self.ensure_started()
        run_id = uuid.uuid4().hex
        with self._state_lock:
            self._execution_order += 1
            if isinstance(state.step, ExecAction):
                state.step.repl_execution_order = self._execution_order
            self.runs[run_id] = state
        binding: dict[str, Any] = {
            "run_id": run_id,
            "tenant_id": tenant.tenant_id,
            "inputs": dict(tenant.inputs),
            "env": dict(tenant.env),
            "structured_output": tenant.structured_output,
        }
        agents = tenant.namespace.get(AGENTS_BINDING)
        if agents is not None:
            binding["agents"] = agents
            binding["expose_agents"] = "AGENTS" in tenant.namespace
        request = RunRequest(
            id=run_id,
            tenant_id=tenant.tenant_id,
            code=code,
            binding=encode_host_value(binding),
        )
        try:
            response = self.request(request, timeout=self.execution_timeout)
            if not response.ok:
                raise RuntimeError(response.error or "worker execution failed")
            if response.env is not None:
                tenant.env = dict(response.env)
            if state.answer is not _NO_ANSWER:
                return ReplRun(
                    output=response.output,
                    status=ReplStatus.DONE,
                    answer=state.answer,
                )
            status = ReplStatus.ERROR if response.errored else ReplStatus.OK
            return ReplRun(output=response.output, status=status)
        finally:
            with self._state_lock:
                self.runs.pop(run_id, None)

    def close(self, *, force: bool = False) -> None:
        with self._state_lock:
            if self.closed:
                return
            self.closed = True
            futures = tuple(self.pending.values())
        error = RuntimeError("worker session is closed")
        for future in futures:
            if not future.done():
                future.set_exception(error)
        self.connection.close(force=force)


class WorkerRepl(Repl):
    """One logical agent tenant in a lightweight Python worker."""

    def __init__(
        self,
        session: WorkerSession,
        *,
        tenant_id: str | None = None,
    ) -> None:
        self.session = session
        self.tenant_id = tenant_id or uuid.uuid4().hex
        self.namespace: dict[str, Any] = {}
        self.env: dict[str, Any] = {}
        self.inputs: dict[str, str] = {}
        self.structured_output = False
        self.proxied: dict[str, Callable[..., object]] = {}
        self.closed = False
        self._run_lock = threading.Lock()
        self.session.add_tenant(self)

    def seed(self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]) -> None:
        self.namespace = dict(tools)
        self.inputs = dict(inputs)
        for name, value in tools.items():
            if name not in {"INPUTS", "ENV", "AGENTS", AGENTS_BINDING}:
                self.inject(name, value)

    def inject(self, name: str, value: Any) -> None:
        self.namespace[name] = value
        metadata = get_tool_metadata(value) if callable(value) else None
        if name == "finish" or (metadata is not None and metadata.proxy):
            self.proxied[name] = value
            request: WireModel = InjectProxyRequest(
                id=self.session._next_id("inject_proxy"),
                tenant_id=self.tenant_id,
                name=name,
                is_async=bool(metadata and metadata.is_async),
            )
        else:
            self.proxied.pop(name, None)
            request = InjectRequest(
                id=self.session._next_id("inject"),
                tenant_id=self.tenant_id,
                name=name,
                value=encode_host_value(value),
            )
        response = self.session.request(request, timeout=self.session.timeout)
        if not response.ok:
            raise RuntimeError(response.error or f"failed to inject {name!r}")

    def get_var(self, name: str) -> Any:
        response = self.session.request(
            RetrieveRequest(
                id=self.session._next_id("retrieve"),
                tenant_id=self.tenant_id,
                name=name,
            ),
            timeout=self.session.timeout,
        )
        if not response.ok:
            raise KeyError(response.error or name)
        return response.value

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)
        self.proxied.pop(name, None)
        self.session.request(
            RemoveRequest(
                id=self.session._next_id("remove"),
                tenant_id=self.tenant_id,
                name=name,
            ),
            timeout=self.session.timeout,
        )

    def update_env(self, values: dict[str, Any]) -> None:
        self.env.update(values)

    async def run(self, code: str) -> ReplRun:
        if not code.strip():
            return ReplRun(output=MISSING_REPL_NOTE, status=ReplStatus.ERROR)
        state = _RunState(
            tenant=self,
            loop=asyncio.get_running_loop(),
            step=active_step(),
            proxied=dict(self.proxied),
        )
        try:
            return await asyncio.to_thread(self._run_blocking, code, state)
        except asyncio.CancelledError:
            await asyncio.to_thread(self.session.close, force=True)
            raise

    def _run_blocking(self, code: str, state: _RunState) -> ReplRun:
        with self._run_lock:
            return self.session.execute(self, code, state)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.session.release(self)


__all__ = [
    "ProtocolDesyncError",
    "ReplConnection",
    "WorkerRepl",
    "WorkerSession",
    "current_rpc_call_id",
]
