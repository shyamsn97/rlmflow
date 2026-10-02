"""Live Python-worker ownership for in-memory agents."""

from __future__ import annotations

import inspect
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import AgentStart, ExecAction, Node
from rlmflow.runtime.connections import DEFAULT_REPL_TIMEOUT, PopenConnection
from rlmflow.runtime.env import agent_process_env
from rlmflow.runtime.repl import Repl, ReplRun, ReplStatus

if TYPE_CHECKING:
    from rlmflow.runtime.repl_client import ReplConnection

ExecutionGuard = Callable[[ExecAction], str | None | Awaitable[str | None]]


def _agent(value: AgentStart | Node) -> AgentStart:
    if value.parent_agent is None:
        raise RuntimeError("node is detached")
    return value.parent_agent


class Runtime:
    """Own one logical REPL tenant per agent.

    Tenants own isolated workers by default. ``reuse_worker=True`` explicitly places
    a child tenant in its parent's worker with an isolated Python namespace. Worker runtimes
    implement ``connect()``; ``open`` wraps that connection in a session.
    """

    repl_timeout: float = DEFAULT_REPL_TIMEOUT
    execution_timeout: float | None = None

    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        preimports: Sequence[str] | None = None,
    ) -> None:
        self.working_directory = Path(working_directory) if working_directory is not None else None
        self.preimports = preimports
        self.repls: dict[str, Repl] = {}

    def worker_command(self, python: str | Path | None = None) -> list[str]:
        """Argv for one worker process, carrying this runtime's preimport choice.

        ``preimports=None`` leaves the flag off so the worker applies
        ``DEFAULT_PREIMPORTS``; an empty sequence sends an empty value, which binds
        nothing.
        """
        command = [str(python or sys.executable), "-u", "-m", "rlmflow.runtime.repl_server"]
        if self.preimports is not None:
            command += ["--preimport", ",".join(self.preimports)]
        return command

    def connect(self) -> ReplConnection:
        """A fresh, unstarted connection to one worker process."""
        raise NotImplementedError

    def open(self, agent: AgentStart) -> Repl:
        from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession

        reused = self.reused_worker_repl(agent)
        if reused is not None:
            return reused
        session = WorkerSession(
            self.connect(),
            timeout=self.repl_timeout,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)

    def get(self, value: AgentStart | Node) -> Repl | None:
        return self.repls.get(_agent(value).id)

    def repl_for(self, value: AgentStart | Node) -> Repl:
        agent = _agent(value)
        repl = self.repls.get(agent.id)
        if repl is None:
            repl = self.open(agent)
            self.repls[agent.id] = repl
            self.refresh_env(agent)
        return repl

    def reused_worker_repl(self, agent: AgentStart) -> Repl | None:
        """Return a child tenant in its parent's worker when explicitly requested."""
        if not agent.config.reuse_worker or agent.parent is None:
            return None
        from rlmflow.runtime.repl_client import WorkerRepl

        parent = agent.parent.parent_agent
        parent_repl = self.repls.get(parent.id)
        if not isinstance(parent_repl, WorkerRepl):
            raise RuntimeError("reuse_worker requires a live parent worker")
        return WorkerRepl(parent_repl.session, tenant_id=agent.id)

    def refresh_env(self, value: AgentStart | Node) -> None:
        """Refresh public metadata for an agent's live REPL."""
        agent = _agent(value)
        repl = self.repls.get(agent.id)
        if repl is None:
            return
        repl.update_env(
            agent_process_env(
                agent_id=agent.config.path,
                depth=agent.config.depth,
                parent_agent_id=agent.config.path.rpartition(".")[0] or None,
                max_depth=agent.config.max_depth,
            )
        )

    async def execute(self, value: AgentStart | Node, code: str) -> ReplRun:
        """Run code in this agent's REPL; a REPL that dies is an outcome, not a raise.

        Only a remote REPL can fail here — a local one captures whatever the code
        does — and the repair is this object's own bookkeeping: drop the dead entry
        so the next step opens a fresh one.
        """
        try:
            return await self.repl_for(value).run(code)
        except Exception as exc:  # noqa: BLE001 - a dead REPL is an observation
            self.close_repl(value)
            text = f"REPL execution failed: {type(exc).__name__}: {exc}"
            return ReplRun(output=text, status=ReplStatus.DEAD, error=exc)

    def close_repl(self, value: AgentStart | Node | str) -> None:
        key = value if isinstance(value, str) else _agent(value).id
        repl = self.repls.pop(key, None)
        if repl is not None:
            with suppress(Exception):
                repl.close()

    def close_repls(self) -> None:
        for key in tuple(self.repls):
            self.close_repl(key)

    def close(self) -> None:
        self.close_repls()

    def get_var(self, value: AgentStart | Node, name: str) -> Any:
        return self.repl_for(value).get_var(name)

    def get_env_var(self, value: AgentStart | Node, name: str) -> Any:
        return self.repl_for(value).get_env_var(name)

    def inject_live(self, name: str, obj: Any) -> None:
        for repl in self.repls.values():
            repl.inject(name, obj)

    def remove_live(self, name: str) -> None:
        for repl in self.repls.values():
            repl.remove_tool(name)


class WrappedRuntime:
    """Prepare actions before delegating execution to the original runtime.

    The original runtime remains available as ``runtime`` so custom steps can
    reach lower-level REPL operations or extend action preparation.
    """

    def __init__(
        self,
        runtime: Runtime,
        build_namespace: Callable[[Node], dict[str, Any]],
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        self.runtime = runtime
        self.build_namespace = build_namespace
        self.execution_guard = execution_guard

    async def execute(self, node: ExecAction) -> ReplRun:
        if self.execution_guard is not None:
            reason = self.execution_guard(node)
            if inspect.isawaitable(reason):
                reason = await reason
            if reason is not None:
                return ReplRun(output=reason, status=ReplStatus.ERROR)
        self.prepare(node)
        return await self.runtime.execute(node, node.code)

    def prepare(self, node: ExecAction) -> Repl:
        """Seed the live or replayed action through one preparation path."""
        agent = node.parent_agent
        repl = self.runtime.repl_for(agent)
        repl.seed(
            self.build_namespace(node),
            agent.invocation_arguments(),
        )
        return repl


class LocalRuntime(Runtime):
    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        repl_timeout: float = 120.0,
        execution_timeout: float | None = None,
        preimports: Sequence[str] | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory, preimports=preimports)
        self.repl_timeout = repl_timeout
        self.execution_timeout = execution_timeout

    def connect(self) -> PopenConnection:
        return PopenConnection(
            self.worker_command(),
            cwd=self.working_directory or Path.cwd(),
            label="Local REPL worker",
        )


class SubprocessRuntime(Runtime):
    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        python: str | Path | None = None,
        env: dict[str, str] | None = None,
        repl_timeout: float | None = DEFAULT_REPL_TIMEOUT,
        execution_timeout: float | None = None,
        preimports: Sequence[str] | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory, preimports=preimports)
        self.python = str(python or sys.executable)
        self.env = env
        self.repl_timeout = repl_timeout or DEFAULT_REPL_TIMEOUT
        self.execution_timeout = execution_timeout

    def connect(self) -> PopenConnection:
        return PopenConnection(
            self.worker_command(self.python),
            cwd=self.working_directory or Path.cwd(),
            env=None if self.env is None else {**os.environ, **self.env},
            label="Subprocess REPL worker",
        )


__all__ = [
    "DEFAULT_REPL_TIMEOUT",
    "LocalRuntime",
    "Repl",
    "Runtime",
    "SubprocessRuntime",
    "WrappedRuntime",
]
