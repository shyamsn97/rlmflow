"""Minimal runtime boundary for where REPL code executes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rflow.minimal.docker import build_docker_argv
from rflow.minimal.modal import ModalServerConnection
from rflow.minimal.remote import ReplClient
from rflow.minimal.repl import Repl
from rflow.minimal.stdio import PopenServerConnection
from rflow.minimal.tools import get_tool_metadata

if TYPE_CHECKING:
    from rflow.minimal.graph import Graph


@runtime_checkable
class ReplLike(Protocol):
    namespace: dict[str, Any]
    done_result: str | None
    errored: bool

    def seed(
        self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]
    ) -> None:
        ...

    async def run(self, code: str) -> str:
        ...

    def drain(self) -> str:
        ...

    def close(self) -> None:
        ...


class Runtime:
    """Factory for one REPL-like backend per graph agent."""

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.working_directory = (
            Path(working_directory) if working_directory is not None else None
        )
        self.tools: dict[str, Callable[..., object]] = {}

    def register_tool(
        self, fn: Callable[..., object], *, name: str | None = None
    ) -> None:
        meta = get_tool_metadata(fn)
        self.tools[name or (meta.name if meta is not None else fn.__name__)] = fn

    def register_tools(self, tools: list[Callable[..., object]]) -> None:
        for fn in tools:
            self.register_tool(fn)

    def open(self, graph: Graph) -> ReplLike:
        raise NotImplementedError

    def deploy_repl_server(self, graph: Graph) -> ReplClient:
        """Launch/connect to a remote REPL server and return its client."""
        raise NotImplementedError


class LocalRuntime(Runtime):
    """Run code in the current Python process."""

    def open(self, graph: Graph) -> ReplLike:
        return Repl(working_directory=self.working_directory)


class SubprocessRuntime(Runtime):
    """Run each agent in a local Python subprocess using the remote protocol."""

    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        python: str | Path | None = None,
        env: dict[str, str] | None = None,
        repl_timeout: float | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.python = str(python or sys.executable)
        self.env = env
        self.repl_timeout = repl_timeout

    def deploy_repl_server(self, graph: Graph) -> ReplClient:
        argv = [self.python, "-u", "-m", "rflow.minimal.remote_server"]
        if self.working_directory is not None:
            argv += ["--workdir", str(self.working_directory)]
        return ReplClient(
            PopenServerConnection(
                argv,
                env=self.env,
                label="minimal subprocess REPL",
                repl_timeout=self.repl_timeout,
            )
        )

    def open(self, graph: Graph) -> ReplLike:
        return self.deploy_repl_server(graph)


class DockerRuntime(Runtime):
    """Run each agent in a Docker container using the minimal remote protocol."""

    def __init__(
        self,
        image: str,
        *,
        working_directory: str | Path | None = None,
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        extra_args: list[str] | None = None,
        docker_bin: str = "docker",
        repl_timeout: float | None = None,
        **options: object,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.image = image
        if self.working_directory is not None:
            host = str(self.working_directory.resolve())
            if mounts is None:
                mounts = {host: "/workspace"}
            if workdir is None:
                workdir = "/workspace"
        self.options = dict(
            mounts=mounts,
            env=env,
            network=network,
            cpus=cpus,
            memory=memory,
            user=user,
            workdir=workdir,
            extra_args=extra_args,
            docker_bin=docker_bin,
            repl_timeout=repl_timeout,
            **options,
        )

    def deploy_repl_server(self, graph: Graph) -> ReplClient:
        cwd = str(self.working_directory) if self.working_directory else None
        options = dict(self.options)
        repl_timeout = options.pop("repl_timeout", None)
        argv = build_docker_argv(self.image, **options)
        return ReplClient(
            PopenServerConnection(
                argv,
                cwd=cwd,
                label="minimal Docker REPL",
                repl_timeout=repl_timeout,
            )
        )

    def open(self, graph: Graph) -> ReplLike:
        return self.deploy_repl_server(graph)


class ModalRuntime(Runtime):
    """Run each agent in a Modal Sandbox using the minimal remote protocol."""

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

    def deploy_repl_server(self, graph: Graph) -> ReplClient:
        return ReplClient(
            ModalServerConnection(
                self.app_name,
                remote_workdir=self.remote_workdir,
                image=self.image,
                timeout=self.timeout,
                repl_timeout=self.repl_timeout,
                **self.container_kwargs,
            )
        )

    def open(self, graph: Graph) -> ReplLike:
        return self.deploy_repl_server(graph)


__all__ = [
    "DockerRuntime",
    "LocalRuntime",
    "ModalRuntime",
    "ReplLike",
    "Runtime",
    "SubprocessRuntime",
]
