"""Modal Sandbox runtime for the remote REPL protocol."""

from __future__ import annotations

import importlib
import threading
from collections import deque
from contextlib import suppress

from rlmflow.graph.nodes import Node
from rlmflow.runtime.protocol import (
    ProxyCall,
    ReplResponse,
    WireModel,
    dump_message,
    parse_client_message,
)
from rlmflow.runtime.repl import Repl
from rlmflow.runtime.repl_client import RemoteRepl, ReplConnection
from rlmflow.runtime.runtime import Runtime


class ModalConnection(ReplConnection):
    """Connect RemoteRepl to a repl_server process in a Modal Sandbox."""

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        remote_workdir: str = "/workspace",
        image: object = None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        **container_kwargs: object,
    ) -> None:
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.repl_timeout = repl_timeout
        self.container_kwargs = container_kwargs
        self.container = None
        self._stdout_iter = None
        self._stdout_pending = ""
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._closing = threading.Event()

    def _ensure_sandbox(self) -> None:
        if self.container is not None:
            return
        try:
            modal = importlib.import_module("modal")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "Modal remote REPL requires the optional `modal` dependency."
            ) from exc
        app = modal.App.lookup(self.app_name, create_if_missing=True)
        image = self.image or modal.Image.debian_slim().pip_install("rlmflow", "cloudpickle>=2.2")
        self.container = modal.Sandbox.create(
            "python",
            "-u",
            "-m",
            "rlmflow.runtime.repl_server",
            "--workdir",
            self.remote_workdir,
            app=app,
            image=image,
            timeout=self.timeout,
            **self.container_kwargs,
        )
        self._closing.clear()
        self._stdout_iter = iter(self.container.stdout)
        self._stdout_pending = ""

    def send(self, msg: WireModel) -> None:
        self._ensure_sandbox()
        assert self.container is not None
        self.container.stdin.write(dump_message(msg) + "\n")
        self.container.stdin.drain()

    def recv(self) -> ReplResponse | ProxyCall:
        self._ensure_sandbox()
        return parse_client_message(self._recv_line())

    def _recv_line(self) -> str:
        if self._stdout_iter is None:
            raise RuntimeError("Modal stdout stream is not available")
        try:
            while True:
                if "\n" in self._stdout_pending:
                    line, self._stdout_pending = self._stdout_pending.split("\n", 1)
                    if line:
                        return line
                self._stdout_pending += _to_text(next(self._stdout_iter))
        except StopIteration as exc:
            raise self._closed_error() from exc
        except Exception as exc:
            if self._is_expected_stream_close(exc):
                raise self._closed_error() from exc
            raise

    def _closed_error(self) -> RuntimeError:
        stderr = "".join(self._stderr_tail).strip()
        return RuntimeError(f"Modal REPL exited. stderr: {stderr or '<empty>'}")

    def _is_expected_stream_close(self, exc: Exception) -> bool:
        if self._closing.is_set():
            return True
        return exc.__class__.__name__ in {
            "ClientClosed",
            "StreamTerminatedError",
            "GRPCError",
        }

    def close(self) -> None:
        container, self.container = self.container, None
        self._closing.set()
        self._stdout_iter = None
        self._stdout_pending = ""
        self._stderr_tail.clear()
        if container is not None:
            with suppress(Exception):
                container.terminate()


def _to_text(data: object) -> str:
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return str(data)


class ModalRuntime(Runtime):
    """Run each agent in a Modal Sandbox using the remote protocol."""

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        remote_workdir: str = "/workspace",
        image: object = None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        **container_kwargs: object,
    ) -> None:
        super().__init__(working_directory=remote_workdir)
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.repl_timeout = repl_timeout
        self.container_kwargs = dict(container_kwargs)

    def deploy_repl_server(self, agent: Node) -> RemoteRepl:
        return RemoteRepl(
            ModalConnection(
                self.app_name,
                remote_workdir=self.remote_workdir,
                image=self.image,
                timeout=self.timeout,
                repl_timeout=self.repl_timeout,
                **self.container_kwargs,
            )
        )

    def open(self, agent: Node) -> Repl:
        return self.deploy_repl_server(agent)


__all__ = ["ModalConnection", "ModalRuntime"]
