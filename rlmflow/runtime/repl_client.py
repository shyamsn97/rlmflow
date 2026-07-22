"""Host-side client for the minimal remote REPL protocol."""

from __future__ import annotations

import asyncio
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, Protocol

from rlmflow.runtime.protocol import (
    CapabilitiesRequest,
    CapabilityMap,
    InjectImportRequest,
    InjectProxyRequest,
    InjectRequest,
    InjectSourceRequest,
    ProxyCall,
    ProxyResponse,
    RemoveRequest,
    ReplResponse,
    RetrieveRequest,
    RunRequest,
    SetEnvRequest,
    WireModel,
)
from rlmflow.runtime.repl import DoneSignal, MissingReplError, Repl
from rlmflow.runtime.serial import (
    CLOUDPICKLE,
    decode_object,
    encode_object,
    is_json_safe,
)
from rlmflow.tools import get_tool_metadata


class ReplConnection(Protocol):
    """Minimal send/recv boundary used by RemoteRepl."""

    def send(self, msg: WireModel) -> None: ...

    def recv(self) -> ReplResponse | ProxyCall: ...

    def close(self) -> None: ...


class RemoteRepl(Repl):
    """A :class:`~rlmflow.runtime.repl.Repl` that runs code in a sandbox.

    Not a REPL itself — a host-side stub that speaks the wire protocol over a
    :class:`ReplConnection` to a ``ReplServer`` (which wraps a ``LocalRepl`` in
    another process/container). Objects cross the boundary by value; see
    :meth:`inject`/:meth:`get_var`. Returned by the process-isolated runtimes
    (``SubprocessRuntime``/``DockerRuntime``/``ModalRuntime``).
    """

    def __init__(self, connection: ReplConnection) -> None:
        self.connection = connection
        self.namespace: dict[str, Any] = {}
        # Mirror of the sandbox's host-visible ``ENV`` state channel, refreshed
        # from each run response (see Repl.env).
        self.env: dict[str, Any] = {}
        self.done_result: str | None = None
        self.errored = False
        self.proxied: dict[str, Callable[..., object]] = {}
        self._done: Callable[[object], object] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._request_id = 0
        self._capabilities: CapabilityMap | None = None

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
                # tools) is driven by its own metadata via inject.
                self.inject(name, fn)

    def inject(self, name: str, fn: Any) -> None:
        """Ship any Python object into the remote namespace.

        Proxy tools run back on the host; other callables are imported or
        source-shipped into the sandbox. Plain (JSON) values are injected as-is;
        arbitrary live objects are shipped **by value** as a cloudpickle blob and
        rebuilt as an independent copy in the sandbox.
        """
        self.namespace[name] = fn
        if not callable(fn):
            self._inject_value(name, fn)
            return
        meta = get_tool_metadata(fn)
        if meta is not None and meta.proxy:
            self._inject_proxy(name, self._sync_callable(fn), is_async=meta.is_async)
        else:
            self._inject_local_tool(name, fn)

    def get_var(self, name: str) -> Any:
        """Read a variable back out of the remote Python namespace (by value).

        JSON-safe data comes back as-is; anything else is returned as a
        cloudpickle copy of the sandbox object. Raises if ``name`` is unbound.
        """
        resp = self.call(RetrieveRequest(id=self._next_id("retrieve"), name=name))
        if resp.value_encoding == CLOUDPICKLE:
            return decode_object(resp.value)
        return resp.value

    def get_env_var(self, name: str) -> Any:
        """Return a value from the mirrored REPL ``env`` metadata channel.

        Reflects the last sync from the sandbox (after ``run`` / ``update_env``).
        Raises ``KeyError`` if ``name`` is unset.
        """
        if name not in self.env:
            raise KeyError(name)
        return self.env[name]

    def capabilities(self) -> CapabilityMap:
        """Fetch (and cache) the sandbox's advertised capabilities."""
        if self._capabilities is None:
            resp = self.call(CapabilitiesRequest(id=self._next_id("capabilities")))
            self._capabilities = resp.capabilities or CapabilityMap()
        return self._capabilities

    def _inject_value(self, name: str, value: Any) -> None:
        # Plain JSON data goes over verbatim (readable, no dependency); a live
        # object is shipped by value via cloudpickle when the sandbox supports it.
        if is_json_safe(value):
            self.call(InjectRequest(id=self._next_id("inject"), name=name, value=value))
            return
        if not self.capabilities().cloudpickle:
            raise RuntimeError(
                f"cannot inject {name!r}: the value is not JSON-serializable and "
                "this sandbox does not support cloudpickle object injection "
                "(install cloudpickle in the sandbox, or inject JSON-safe data)."
            )
        self.call(
            InjectRequest(
                id=self._next_id("inject"),
                name=name,
                value=encode_object(value),
                encoding=CLOUDPICKLE,
            )
        )

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)
        self.proxied.pop(name, None)
        self.call(RemoveRequest(id=self._next_id("remove"), name=name))

    def update_env(self, values: dict[str, Any]) -> None:
        if not values:
            return
        self.env.update(values)
        self.call(SetEnvRequest(id=self._next_id("set_env"), values=dict(values)))

    async def run(self, code: str) -> str:
        if not code.strip():
            self.errored = True
            return f"{MissingReplError.__name__}: missing ```repl``` block"
        self._loop = asyncio.get_running_loop()
        resp = await asyncio.to_thread(
            self.call, RunRequest(id=self._next_id("run"), code=code)
        )
        self.errored = resp.errored
        if resp.env is not None:
            self.env = resp.env
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


__all__ = ["RemoteRepl", "ReplConnection"]
