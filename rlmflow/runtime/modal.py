"""Modal Sandbox provisioner for the lightweight Python worker."""

from __future__ import annotations

import importlib
import threading
from collections import deque
from contextlib import suppress
from typing import Any

from rlmflow.graph.nodes import AgentStart
from rlmflow.runtime.connections import STDERR_LINES
from rlmflow.runtime.protocol import (
    ProxyCall,
    ReplResponse,
    WireModel,
    dump_message,
    parse_client_message,
)
from rlmflow.runtime.repl import Repl
from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession
from rlmflow.runtime.runtime import Runtime


class _ModalConnection:
    def __init__(
        self,
        *,
        app_name: str,
        remote_workdir: str,
        image: object,
        timeout: int,
        container_kwargs: dict[str, object],
    ) -> None:
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.container_kwargs = container_kwargs
        self.container: Any = None
        self._stdout: Any = None
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_LINES)
        self._start_lock = threading.Lock()
        self._stderr_reader: threading.Thread | None = None

    def start(self) -> None:
        with self._start_lock:
            if self.container is not None:
                return
            try:
                modal = importlib.import_module("modal")
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ModuleNotFoundError(
                    "Modal execution requires the optional `modal` dependency."
                ) from exc
            app = modal.App.lookup(self.app_name, create_if_missing=True)
            image = self.image or modal.Image.debian_slim(python_version="3.12").pip_install(
                "rlmflow"
            )
            self.container = modal.Sandbox.create(
                "python",
                "-u",
                "-m",
                "rlmflow.runtime.repl_server",
                app=app,
                image=image,
                timeout=self.timeout,
                workdir=self.remote_workdir,
                **self.container_kwargs,
            )
            self._stdout = iter(self.container.stdout)
            self._drain_stderr()

    def _drain_stderr(self) -> None:
        def pump() -> None:
            with suppress(Exception):
                for chunk in self.container.stderr:
                    text = (
                        chunk.decode(errors="replace") if isinstance(chunk, bytes) else str(chunk)
                    )
                    self._stderr_tail.extend(text.rstrip("\n").splitlines())

        self._stderr_reader = threading.Thread(
            target=pump,
            daemon=True,
            name="Modal REPL worker stderr",
        )
        self._stderr_reader.start()

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def send(self, message: WireModel) -> None:
        self.start()
        self.container.stdin.write(dump_message(message) + "\n")
        self.container.stdin.drain()

    def recv(self) -> ReplResponse | ProxyCall:
        self.start()
        try:
            chunk = next(self._stdout)
        except StopIteration:
            raise RuntimeError(
                f"Modal REPL worker exited unexpectedly. stderr: {self.stderr_tail()}"
            ) from None
        text = chunk.decode(errors="replace") if isinstance(chunk, bytes) else str(chunk)
        return parse_client_message(text)

    def close(self, *, force: bool = False) -> None:
        del force
        with self._start_lock:
            container, self.container = self.container, None
        if container is None:
            return

        def terminate() -> None:
            with suppress(Exception):
                container.terminate()

        thread = threading.Thread(target=terminate, daemon=True)
        thread.start()
        thread.join(timeout=10)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=1)
            self._stderr_reader = None


class ModalRuntime(Runtime):
    """Run isolated or explicitly shared agents in Modal Sandbox workers."""

    def __init__(
        self,
        app_name: str = "rlmflow",
        *,
        remote_workdir: str = "/workspace",
        image: object = None,
        timeout: int = 3600,
        repl_timeout: float = 30,
        execution_timeout: float | None = None,
        **container_kwargs: object,
    ) -> None:
        super().__init__()
        self.app_name = app_name
        self.remote_workdir = remote_workdir
        self.image = image
        self.timeout = timeout
        self.repl_timeout = repl_timeout
        self.execution_timeout = execution_timeout
        self.container_kwargs = dict(container_kwargs)

    def open(self, agent: AgentStart) -> Repl:
        if agent.config.reuse_repl and agent.parent is not None:
            parent = agent.parent.parent_agent
            parent_repl = self.repls.get(parent.id)
            if not isinstance(parent_repl, WorkerRepl):
                raise RuntimeError("reuse_repl requires a live parent worker")
            return WorkerRepl(parent_repl.session, tenant_id=agent.id)
        connection = _ModalConnection(
            app_name=self.app_name,
            remote_workdir=self.remote_workdir,
            image=self.image,
            timeout=self.timeout,
            container_kwargs=self.container_kwargs,
        )
        session = WorkerSession(
            connection,
            timeout=self.repl_timeout,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)


__all__ = ["ModalRuntime"]
