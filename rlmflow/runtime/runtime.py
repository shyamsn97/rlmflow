"""Live Python-worker ownership for in-memory agents, and rebuilding a worker's
namespace by replaying the ``ExecAction``s in an agent's history."""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph import reads
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import AgentStart, ExecAction, Step, current_step, running_step
from rlmflow.runtime.repl import Repl, ReplRun, ReplStatus
from rlmflow.runtime.repl_server import (
    ProxyCall,
    ReplResponse,
    WireModel,
    dump_message,
    parse_client_message,
)

if TYPE_CHECKING:
    from rlmflow.graph.graph import Graph
    from rlmflow.runtime.repl_client import ReplConnection

# Public ``RLMFLOW_*`` metadata an agent can read from its per-REPL ``ENV``
# mapping (e.g. ``ENV["RLMFLOW_AGENT_ID"]``). Being per-REPL, it stays isolated
# across concurrent local agents. Private control state (output schemas, done
# results) never travels this way.

#: Current agent id (for example ``"root"`` or ``"root.worker"``).
RLMFLOW_AGENT_ID = "RLMFLOW_AGENT_ID"
#: Current agent depth in the spawn tree (``"0"`` for root).
RLMFLOW_DEPTH = "RLMFLOW_DEPTH"
#: Parent agent id, or ``""`` for root.
RLMFLOW_PARENT_AGENT_ID = "RLMFLOW_PARENT_AGENT_ID"
#: Recursion bound configured on the flow.
RLMFLOW_MAX_DEPTH = "RLMFLOW_MAX_DEPTH"
#: ``"1"`` for root, ``"0"`` otherwise.
RLMFLOW_IS_ROOT = "RLMFLOW_IS_ROOT"
#: ``"1"`` while recorded code is being re-run to rebuild a namespace, ``"0"`` during
#: a live turn. Code that reads it can skip work it should not repeat.
RLMFLOW_REPLAY = "RLMFLOW_REPLAY"


def agent_process_env(
    *,
    agent_id: str,
    depth: int,
    parent_agent_id: str | None,
    max_depth: int,
) -> dict[str, str]:
    """Return public ``RLMFLOW_*`` environment variables for one agent."""
    return {
        RLMFLOW_AGENT_ID: agent_id,
        RLMFLOW_DEPTH: str(depth),
        RLMFLOW_PARENT_AGENT_ID: parent_agent_id or "",
        RLMFLOW_MAX_DEPTH: str(max_depth),
        RLMFLOW_IS_ROOT: "1" if depth == 0 else "0",
        RLMFLOW_REPLAY: "0",
    }


STDERR_LINES = 200
DEFAULT_REPL_TIMEOUT = 300.0


class PopenConnection:
    """A lazy subprocess connected by protected JSON-line stdio."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        label: str = "REPL worker",
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.label = label
        self.proc: subprocess.Popen[bytes] | None = None
        self._start_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_LINES)
        self._stderr_reader: threading.Thread | None = None

    def start(self) -> None:
        self._process()

    def _process(self) -> subprocess.Popen[bytes]:
        with self._start_lock:
            if self.proc is None:
                self.proc = subprocess.Popen(
                    self.argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.cwd) if self.cwd is not None else None,
                    env=self.env,
                    bufsize=0,
                )
                self._drain_stderr(self.proc)
            return self.proc

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return

        def pump() -> None:
            with suppress(Exception):
                for line in iter(proc.stderr.readline, b""):
                    self._stderr_tail.append(line.decode(errors="replace").rstrip("\n"))

        self._stderr_reader = threading.Thread(
            target=pump,
            daemon=True,
            name=f"{self.label} stderr",
        )
        self._stderr_reader.start()

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def send(self, message: WireModel) -> None:
        proc = self._process()
        if proc.stdin is None:
            raise RuntimeError(f"{self.label} stdin is unavailable")
        proc.stdin.write((dump_message(message) + "\n").encode())
        proc.stdin.flush()

    def recv(self) -> ReplResponse | ProxyCall:
        proc = self._process()
        if proc.stdout is None:
            raise RuntimeError(f"{self.label} stdout is unavailable")
        line = proc.stdout.readline()
        if not line:
            raise self._exited_error()
        return parse_client_message(line.decode())

    def _exited_error(self) -> RuntimeError:
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=0.2)
        tail = self.stderr_tail()
        detail = f" stderr: {tail}" if tail else ""
        return RuntimeError(f"{self.label} exited unexpectedly.{detail}")

    def close(self, *, force: bool = False) -> None:
        with self._start_lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        if force and proc.poll() is None:
            with suppress(Exception):
                proc.terminate()
        with suppress(Exception):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=0.5 if force else 2)
        except subprocess.TimeoutExpired:
            with suppress(Exception):
                proc.kill()
                proc.wait(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            with suppress(Exception):
                if stream is not None and not stream.closed:
                    stream.close()


ExecutionGuard = Callable[[ExecAction], str | None | Awaitable[str | None]]


def _key(agent: AgentStart | str) -> str:
    return agent if isinstance(agent, str) else agent.id


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

    def open(self, agent: AgentStart, graph: Graph | None = None) -> Repl:
        from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession

        reused = self.reused_worker_repl(agent, graph)
        if reused is not None:
            return reused
        session = WorkerSession(
            self.connect(),
            timeout=self.repl_timeout,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)

    def get(self, agent: AgentStart | str) -> Repl | None:
        return self.repls.get(_key(agent))

    def repl_for(self, agent: AgentStart, graph: Graph | None = None) -> Repl:
        """The agent's live REPL, opened on first use; ``graph`` places it in the run."""
        repl = self.repls.get(agent.id)
        if repl is None:
            repl = self.open(agent, graph)
            self.repls[agent.id] = repl
            repl.update_env(agent_env(agent, graph))
        return repl

    def reused_worker_repl(self, agent: AgentStart, graph: Graph | None = None) -> Repl | None:
        """Return a child tenant in its launcher's worker when explicitly requested."""
        from rlmflow.graph import reads

        launcher = reads.launcher(graph, agent) if graph is not None else None
        if not agent.config.reuse_worker or launcher is None:
            return None
        from rlmflow.runtime.repl_client import WorkerRepl

        parent_repl = self.repls.get(launcher.id)
        if not isinstance(parent_repl, WorkerRepl):
            raise RuntimeError("reuse_worker requires a live parent worker")
        return WorkerRepl(parent_repl.session, tenant_id=agent.id)

    async def execute(self, agent: AgentStart, code: str, graph: Graph | None = None) -> ReplRun:
        """Run code in this agent's REPL; a REPL that dies is an outcome, not a raise.

        Only a remote REPL can fail here — a local one captures whatever the code
        does — and the repair is this object's own bookkeeping: drop the dead entry
        so the next step opens a fresh one.
        """
        try:
            return await self.repl_for(agent, graph).run(code)
        except asyncio.CancelledError:
            # A cancelled run force-closes its worker; never hand that worker out again.
            self.close_repl(agent)
            raise
        except Exception as exc:  # noqa: BLE001 - a dead REPL is an observation
            self.close_repl(agent)
            text = f"REPL execution failed: {type(exc).__name__}: {exc}"
            return ReplRun(output=text, status=ReplStatus.DEAD, error=exc)

    def close_repl(self, agent: AgentStart | str) -> None:
        repl = self.repls.pop(_key(agent), None)
        if repl is not None:
            with suppress(Exception):
                repl.close()

    def close_repls(self) -> None:
        for key in tuple(self.repls):
            self.close_repl(key)

    def close(self) -> None:
        self.close_repls()

    def get_var(self, agent: AgentStart, name: str) -> Any:
        return self.repl_for(agent).get_var(name)

    def get_env_var(self, agent: AgentStart, name: str) -> Any:
        return self.repl_for(agent).get_env_var(name)

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
        build_namespace: Callable[[ExecAction], dict[str, Any]],
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
        step = current_step()
        return await self.runtime.execute(step.agent, node.code, step.graph)

    def prepare(self, node: ExecAction) -> Repl:
        """Seed the live or replayed action of the current step through one path."""
        from rlmflow.graph import reads

        step = current_step()
        if step is None:
            raise RuntimeError("actions execute inside a step")
        g, agent = step.graph, step.agent
        repl = self.runtime.repl_for(agent, g)
        repl.update_env(agent_env(agent, g))
        repl.seed(
            self.build_namespace(node),
            {"task": agent.content, **agent.resolved_args(), **reads.inputs(g, agent)},
        )
        return repl


def agent_env(agent: AgentStart, graph: Graph | None = None) -> dict[str, str]:
    """The ``ENV`` metadata describing where ``agent`` sits in its run."""
    from rlmflow.graph import reads

    if graph is None:
        path, depth, parent = agent.config.name, 0, None
    else:
        path, depth = reads.path(graph, agent), reads.depth(graph, agent)
        launcher = reads.launcher(graph, agent)
        parent = reads.path(graph, launcher) if launcher is not None else None
    return agent_process_env(
        agent_id=path,
        depth=depth,
        parent_agent_id=parent,
        max_depth=agent.config.max_depth,
    )


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


async def rebuild_repl(flow: Any, g: Graph, agent: AgentStart) -> None:
    """Rebuild ``agent``'s namespace from the actions in its history.

    Appends nothing: launch calls resolve through their receipts to agents
    already in the graph. A block that failed the first time fails the same way
    here, which is why output and errors are discarded. An action at the
    frontier is skipped because the next step runs it live. A ``reuse_worker``
    agent first restores the launcher whose worker it shares.
    """
    flow._restored_agents.add(agent.id)
    if flow.runtime.get(agent) is not None:
        return
    launcher = reads.launcher(g, agent)
    if agent.config.reuse_worker and launcher is not None:
        await rebuild_repl(flow, g, launcher)
    flow.runtime.repl_for(agent, g)
    history = reads.history(g, agent)
    for node in history[:-1]:
        if not isinstance(node, ExecAction):
            continue
        with running_step(Step(g, agent, node)):
            repl = flow.wrapped_runtime.prepare(node)
            repl.update_env({RLMFLOW_REPLAY: "1"})
            try:
                await flow.runtime.execute(agent, node.code, g)
            finally:
                repl.update_env({RLMFLOW_REPLAY: "0"})


async def ensure_repl_rebuilt(flow: Any, g: Graph, agent: AgentStart) -> None:
    """Restore a namespace once per Flow, right before the agent's next action.

    An agent restored earlier whose REPL has since died stays cold.
    """
    if agent.id in flow._restored_agents:
        return
    async with flow._restore_lock:
        await rebuild_repl(flow, g, agent)


async def rebuild_unfinished_repls(flow: Any, g: Graph) -> None:
    """With eager restore, rebuild every unfinished agent's namespace up front."""
    if flow.restore != "replay":
        return
    for agent in g.nodes(AgentStart):
        if not reads.terminal(g, agent):
            await ensure_repl_rebuilt(flow, g, agent)


__all__ = [
    "DEFAULT_REPL_TIMEOUT",
    "LocalRuntime",
    "PopenConnection",
    "RLMFLOW_AGENT_ID",
    "RLMFLOW_DEPTH",
    "RLMFLOW_IS_ROOT",
    "RLMFLOW_MAX_DEPTH",
    "RLMFLOW_PARENT_AGENT_ID",
    "RLMFLOW_REPLAY",
    "Repl",
    "Runtime",
    "STDERR_LINES",
    "SubprocessRuntime",
    "WrappedRuntime",
    "agent_env",
    "agent_process_env",
    "ensure_repl_rebuilt",
    "rebuild_repl",
    "rebuild_unfinished_repls",
]
