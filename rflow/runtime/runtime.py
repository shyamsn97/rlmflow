"""Runtime boundary: where REPL code executes (base + local/subprocess)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rflow.runtime.connections import PopenConnection
from rflow.runtime.repl import Repl, ReplLike
from rflow.runtime.repl_client import ReplClient
from rflow.tools import get_tool_metadata

if TYPE_CHECKING:
    from rflow.graph import Graph


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
        argv = [self.python, "-u", "-m", "rflow.runtime.repl_server"]
        if self.working_directory is not None:
            argv += ["--workdir", str(self.working_directory)]
        return ReplClient(
            PopenConnection(
                argv,
                env=self.env,
                label="minimal subprocess REPL",
                repl_timeout=self.repl_timeout,
            )
        )

    def open(self, graph: Graph) -> ReplLike:
        return self.deploy_repl_server(graph)


__all__ = ["LocalRuntime", "ReplLike", "Runtime", "SubprocessRuntime"]
