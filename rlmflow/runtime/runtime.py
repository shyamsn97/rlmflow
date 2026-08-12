"""Live Python-worker ownership for in-memory agents."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.runtime.connections import DEFAULT_REPL_TIMEOUT
from rlmflow.runtime.env import agent_process_env
from rlmflow.runtime.repl import Repl, ReplRun, ReplStatus


def _agent(value: AgentStart | Node) -> AgentStart:
    if value.parent_agent is None:
        raise RuntimeError("node is detached")
    return value.parent_agent


class Runtime:
    """Own one logical REPL tenant per agent.

    Tenants own isolated workers by default. ``reuse_repl=True`` explicitly places
    a child tenant in its parent's worker and shared Python heap.
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.working_directory = Path(working_directory) if working_directory is not None else None
        self.repls: dict[str, Repl] = {}

    def open(self, agent: AgentStart) -> Repl:
        raise NotImplementedError

    def get(self, value: AgentStart | Node) -> Repl | None:
        return self.repls.get(_agent(value).id)

    def repl_for(self, value: AgentStart | Node) -> Repl:
        agent = _agent(value)
        repl = self.repls.get(agent.id)
        if repl is None:
            repl = self.open(agent)
            repl.update_env(
                agent_process_env(
                    agent_id=agent.config.path,
                    depth=agent.config.depth,
                    parent_agent_id=agent.config.path.rpartition(".")[0] or None,
                    max_depth=agent.config.max_depth,
                )
            )
            self.repls[agent.id] = repl
        return repl

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

    def namespace_for(self, value: AgentStart | Node) -> dict[str, Any] | None:
        repl = self.get(value)
        return repl.namespace if repl is not None else None

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


class LocalRuntime(Runtime):
    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        repl_timeout: float = 120.0,
        execution_timeout: float | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.repl_timeout = repl_timeout
        self.execution_timeout = execution_timeout

    def open(self, agent: AgentStart) -> Repl:
        from rlmflow.runtime.connections import PopenConnection
        from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession

        working_dir = self.working_directory or Path.cwd()
        if agent.config.reuse_repl and agent.parent is not None:
            parent = agent.parent.parent_agent
            parent_repl = self.repls.get(parent.id)
            if not isinstance(parent_repl, WorkerRepl):
                raise RuntimeError("reuse_repl requires a live parent worker")
            return WorkerRepl(parent_repl.session, tenant_id=agent.id)
        connection = PopenConnection(
            [sys.executable, "-u", "-m", "rlmflow.runtime.repl_server"],
            cwd=working_dir,
            label="Local REPL worker",
        )
        session = WorkerSession(
            connection,
            timeout=self.repl_timeout,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)


class SubprocessRuntime(Runtime):
    def __init__(
        self,
        *,
        working_directory: str | Path | None = None,
        python: str | Path | None = None,
        env: dict[str, str] | None = None,
        repl_timeout: float | None = DEFAULT_REPL_TIMEOUT,
        execution_timeout: float | None = None,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.python = str(python or sys.executable)
        self.env = env
        self.repl_timeout = repl_timeout
        self.execution_timeout = execution_timeout

    def open(self, agent: AgentStart) -> Repl:
        from rlmflow.runtime.connections import PopenConnection
        from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession

        working_dir = self.working_directory or Path.cwd()
        if agent.config.reuse_repl and agent.parent is not None:
            parent = agent.parent.parent_agent
            parent_repl = self.repls.get(parent.id)
            if not isinstance(parent_repl, WorkerRepl):
                raise RuntimeError("reuse_repl requires a live parent worker")
            return WorkerRepl(parent_repl.session, tenant_id=agent.id)
        env = None if self.env is None else {**os.environ, **self.env}
        connection = PopenConnection(
            [self.python, "-u", "-m", "rlmflow.runtime.repl_server"],
            cwd=working_dir,
            env=env,
            label="Subprocess REPL worker",
        )
        session = WorkerSession(
            connection,
            timeout=self.repl_timeout or DEFAULT_REPL_TIMEOUT,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)


__all__ = [
    "DEFAULT_REPL_TIMEOUT",
    "LocalRuntime",
    "Repl",
    "Runtime",
    "SubprocessRuntime",
]
