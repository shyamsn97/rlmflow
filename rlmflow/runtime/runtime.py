"""Live REPL ownership for in-memory agents."""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.runtime.connections import PopenConnection
from rlmflow.runtime.env import agent_process_env
from rlmflow.runtime.repl import LocalRepl, Repl, ReplRun, ReplStatus
from rlmflow.runtime.repl_client import RemoteRepl


def _agent(value: AgentStart | Node) -> AgentStart:
    if value.parent_agent is None:
        raise RuntimeError("node is detached")
    return value.parent_agent


class Runtime:
    """Own one live REPL per agent."""

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.working_directory = Path(working_directory) if working_directory is not None else None
        self.repls: dict[str, Repl] = {}

    def open(self, agent: AgentStart) -> Repl:
        raise NotImplementedError

    def deploy_repl_server(self, agent: AgentStart) -> RemoteRepl:
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

    def sync_inputs(self, value: AgentStart | Node) -> None:
        agent = _agent(value)
        repl = self.repls.get(agent.id)
        if repl is not None:
            repl.inject("INPUTS", dict(agent.config.inputs))

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
    def open(self, agent: AgentStart) -> Repl:
        return LocalRepl(working_directory=self.working_directory)


class SubprocessRuntime(Runtime):
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

    def deploy_repl_server(self, agent: AgentStart) -> RemoteRepl:
        argv = [self.python, "-u", "-m", "rlmflow.runtime.repl_server"]
        if self.working_directory is not None:
            argv += ["--workdir", str(self.working_directory)]
        return RemoteRepl(
            PopenConnection(
                argv,
                env=self.env,
                label="minimal subprocess REPL",
                repl_timeout=self.repl_timeout,
            )
        )

    def open(self, agent: AgentStart) -> Repl:
        return self.deploy_repl_server(agent)


__all__ = ["LocalRuntime", "Repl", "Runtime", "SubprocessRuntime"]
