"""Host-side client for the minimal remote REPL protocol."""

from __future__ import annotations

import asyncio
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, Protocol

from rflow.runtime.protocol import (
    InjectImportRequest,
    InjectProxyRequest,
    InjectRequest,
    InjectSourceRequest,
    ProxyCall,
    ProxyResponse,
    RemoveRequest,
    ReplResponse,
    RunRequest,
    SetEnvRequest,
    WireModel,
)
from rflow.runtime.repl import DoneSignal, MissingReplError
from rflow.tools import get_tool_metadata


class ReplConnection(Protocol):
    """Minimal send/recv boundary used by ReplClient."""

    def send(self, msg: WireModel) -> None: ...

    def recv(self) -> ReplResponse | ProxyCall: ...

    def close(self) -> None: ...


class ReplClient:
    """ReplLike host client for a deployed minimal REPL server."""

    def __init__(self, connection: ReplConnection) -> None:
        self.connection = connection
        self.namespace: dict[str, Any] = {}
        self.done_result: str | None = None
        self.errored = False
        self.proxied: dict[str, Callable[..., object]] = {}
        self._done: Callable[[object], object] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._request_id = 0

    def _next_id(self, cmd: str) -> str:
        self._request_id += 1
        return f"{cmd}-{self._request_id}"

    def call(self, msg: WireModel) -> ReplResponse:
        self.connection.send(msg)
        while True:
            resp = self.connection.recv()
            if isinstance(resp, ReplResponse):
                if not resp.ok or resp.error is not None:
                    raise RuntimeError(resp.error or "remote REPL request failed")
                return resp
            self._handle_proxy(resp)

    def seed(
        self,
        tools: dict[str, Callable[..., object]],
        inputs: dict[str, str],
    ) -> None:
        self.namespace = dict(tools)
        self.namespace["INPUTS"] = dict(inputs)
        self.call(
            InjectRequest(id=self._next_id("inject"), name="INPUTS", value=dict(inputs))
        )
        for name, fn in tools.items():
            if name == "INPUTS":
                continue
            # ``done`` is a framework primitive: always a host proxy with special
            # completion semantics, even if seeded without tool metadata.
            if name == "done":
                self._done = fn
                self._inject_proxy("done", self._remote_done)
            else:
                # Everything else (launch_subagents, llm_query_batched, user
                # tools) is driven by its own metadata via inject_tool.
                self.inject_tool(name, fn)

    def inject_tool(self, name: str, fn: Any) -> None:
        """Ship one tool into the remote namespace (dynamic tools).

        Proxy tools run back on the host; other callables are imported or
        source-shipped into the sandbox; plain values are injected directly.
        """
        self.namespace[name] = fn
        if not callable(fn):
            self.call(InjectRequest(id=self._next_id("inject"), name=name, value=fn))
            return
        meta = get_tool_metadata(fn)
        if meta is not None and meta.proxy:
            self._inject_proxy(name, self._sync_callable(fn), is_async=meta.is_async)
        else:
            self._inject_local_tool(name, fn)

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)
        self.proxied.pop(name, None)
        self.call(RemoveRequest(id=self._next_id("remove"), name=name))

    def set_process_env(self, env: dict[str, str]) -> None:
        if not env:
            return
        self.call(
            SetEnvRequest(
                id=self._next_id("set_env"),
                values={str(k): str(v) for k, v in env.items()},
            )
        )

    async def run(self, code: str) -> str:
        if not code.strip():
            self.errored = True
            return f"{MissingReplError.__name__}: missing ```repl``` block"
        self._loop = asyncio.get_running_loop()
        resp = await asyncio.to_thread(
            self.call, RunRequest(id=self._next_id("run"), code=code)
        )
        self.errored = resp.errored
        return resp.output or ""

    def drain(self) -> str:
        return ""

    def close(self) -> None:
        self.connection.close()

    def _inject_proxy(
        self, name: str, fn: Callable[..., object], *, is_async: bool = False
    ) -> None:
        self.proxied[name] = fn
        self.call(
            InjectProxyRequest(
                id=self._next_id("inject_proxy"), name=name, is_async=is_async
            )
        )

    def _inject_local_tool(self, name: str, fn: Callable[..., object]) -> None:
        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", "") or ""
        if module and module != "__main__" and "<locals>" not in qualname:
            self.call(
                InjectImportRequest(
                    id=self._next_id("inject_import"),
                    name=name,
                    module=module,
                    qualname=qualname,
                )
            )
            return
        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                f"cannot ship local tool {name!r} into the sandbox: define it in "
                "an importable module or as a top-level function"
            ) from exc
        self.call(
            InjectSourceRequest(
                id=self._next_id("inject_source"),
                name=name,
                func_name=getattr(fn, "__name__", name),
                source=source,
            )
        )

    def _handle_proxy(self, resp: ProxyCall) -> None:
        fn = self.proxied[resp.proxy]
        try:
            result = fn(*resp.args, **resp.kwargs)
        except DoneSignal:
            self.connection.send(ProxyResponse(id=resp.id, done=True))
            return
        except Exception as exc:  # noqa: BLE001
            self.connection.send(
                ProxyResponse(
                    id=resp.id,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return
        self.connection.send(ProxyResponse(id=resp.id, value=result))

    def _remote_done(self, answer: object) -> None:
        if self._done is None:
            self.done_result = str(answer).strip()
        else:
            self._done(answer)
        raise DoneSignal()

    def _sync_callable(self, fn: Callable[..., object]) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            result = fn(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result
            return self._await_from_thread(result)

        return call

    def _await_from_thread(self, value: Any) -> object:
        async def wait() -> object:
            return await value

        if self._loop is None:
            return asyncio.run(wait())
        return asyncio.run_coroutine_threadsafe(wait(), self._loop).result()


__all__ = ["ReplClient", "ReplConnection"]
