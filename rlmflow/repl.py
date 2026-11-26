"""``Repl``: what an agent root can use — its REPL, files, tools and launched agents.

Flow builds one for a root when it steps it, and one for the node whose code is
running, which tools reach with ``current_repl()``. Host code gets one with
``flow.repl(node)``.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import IMAGES_KEY, AgentRoot, Node, strict_json

if TYPE_CHECKING:
    from rlmflow.agents.agent import Agent
    from rlmflow.flow import Flow
    from rlmflow.graph.calls import CallLog
    from rlmflow.graph.graph import Graph


class Repl:
    """One agent root's REPL handle.

    ``node`` is the node whose code is running, or the root's latest node when
    none is. ``calls`` is that node's ``CallLog``, which tools save their calls
    to, and ``replaying`` marks code run again to rebuild the REPL.
    """

    def __init__(
        self,
        flow: Flow,
        root: AgentRoot,
        node: Node | None = None,
        *,
        calls: CallLog | None = None,
        replaying: bool = False,
    ) -> None:
        self.flow = flow
        self.root = root
        self.graph: Graph = root._graph()
        self.node = node if node is not None else root.nodes()[-1]
        self.calls = calls
        self.replaying = replaying

    def __repr__(self) -> str:
        return f"Repl({self.root.name!r}, node={self.node.id!r})"

    @property
    def agent(self) -> Agent:
        """The configured Flow agent this root uses."""
        return self.flow.agent(self.root)

    @property
    def id(self) -> str:
        """The REPL, workspace and session key: the root's id unless a fork gave it a new one."""
        return self.root.repl_id or self.root.id

    @property
    def path(self) -> Path:
        """The root's files."""
        return self.flow.workspaces.path(self.root)

    @property
    def location(self) -> str | None:
        """Where code runs, when it isn't this machine."""
        return self.flow.runtime.location

    @property
    def environ(self) -> dict[str, str]:
        """``RLMFLOW_URL``, ``RLMFLOW_TOKEN`` and ``PATH`` for a CLI agent's process; starts the server."""
        return self.flow.agent_tools.environ(self.root)

    @property
    def env(self) -> dict[str, Any]:
        """The REPL's ``ENV`` mapping: what its code published for the host; what the host writes reaches its next code."""
        repl = self.flow.runtime.get(self.root)
        return repl.env if repl is not None else {}

    def _process(self) -> Any:
        return self.flow.runtime.repl_for(self.root, self.graph, max_depth=self.agent.max_depth)

    def get(self, name: str) -> Any:
        """A REPL variable's value."""
        return self._process().get_var(name)

    def set(self, **values: Any) -> None:
        """Bind REPL variables for later nodes; a function becomes a host tool."""
        from rlmflow.tools.declare import as_tool

        values = {name: as_tool(v) if callable(v) else v for name, v in values.items()}
        self.flow._bound.setdefault(self.id, {}).update(values)
        repl = self._process()
        for name, value in values.items():
            repl.inject(name, value)

    def tool_docs(self, *, own_only: bool = False) -> str:
        """The tools this root's code can call; ``own_only`` excludes built-ins."""
        from rlmflow.tools.declare import format_tool_docs, tool_allowed
        from rlmflow.tools.namespace import repl_tools

        if own_only:
            tools = self.flow.tools_for(self.graph)
            own = {n: fn for n, fn in tools.items() if tool_allowed(fn, self)}
            return format_tool_docs(own, header="Also in the REPL:")
        return format_tool_docs(repl_tools(self))

    def can_launch(self) -> bool:
        """Whether the run may launch runs: its agent's ``max_depth`` isn't reached."""
        depth = self.agent.max_depth
        return depth > 0 and len(self.root.lineage()) - 1 < depth

    def running(self) -> list[AgentRoot]:
        """Runs this one launched whose results haven't reached it yet."""
        from rlmflow.graph.queries import unseen_results

        unseen = {child.id for child, _final in unseen_results(self.graph, self.root)}
        return [c for c in self.root.children() if c.status() != "done" or c.id in unseen]

    def submitted(self, *, since: Node | None = None) -> Any:
        """What ``submit()`` got in code that ran after ``since``; ``None`` if it wasn't called."""
        for node in reversed(self.root.nodes()):
            if node is since:
                return None
            if node.runs_code and node.result is not None:
                return node.result
        return None

    def commit(self) -> str:
        """Commit the run's files as they are now, and return the commit."""
        return self.flow.workspaces.commit(self.root)

    async def run_command(self, argv: list[str]) -> str:
        """Run a CLI in the run's directory with ``environ``; returns stdout, and cancelling kills it."""
        env = self.environ
        if "/" not in argv[0] and shutil.which(argv[0], path=env.get("PATH")) is None:
            raise RuntimeError(f"{argv[0]!r} was not found; install it and sign in first")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.path,
            env={**os.environ, **env},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
        out, err = stdout.decode(errors="replace").strip(), stderr.decode(errors="replace").strip()
        if process.returncode:
            raise RuntimeError(
                f"{argv[0]} failed: {err or out or f'exit status {process.returncode}'}"
            )
        return out

    async def turn_ended(self) -> Node:
        """Wait for ``/stop`` from ``rlmflow serve``: the session's reply, session id and usage."""
        return await self.flow.agent_tools.turn_ended(self.root)

    async def catch_up(self) -> None:
        """Bring the REPL up to the run's last node, rebuilding it if it is behind, gone or forked.

        Nodes run again with their saved calls, so no tool runs twice.
        """
        from rlmflow.flow import execution

        await execution.prepare(self.flow, self.graph, self.root)

    async def exec(self, node: Node) -> Node:
        """Run ``node``'s code (one of this run's nodes), first catching the REPL up on the nodes before it."""
        from rlmflow.flow import execution

        await execution.run_node(self.flow, self.graph, self.root, node)
        return node


_current: contextvars.ContextVar[Repl | None] = contextvars.ContextVar(
    "rlmflow_current_repl", default=None
)


def current_repl() -> Repl | None:
    """The ``Repl`` of the node whose code is running, if any."""
    return _current.get()


@contextmanager
def running(repl: Repl | None) -> Iterator[None]:
    """Run the enclosed code as part of ``repl``'s node."""
    token = _current.set(repl)
    try:
        yield
    finally:
        _current.reset(token)


def attach_image(path: str | os.PathLike[str]) -> None:
    """Show the image file at ``path`` to the model along with the running node's output.

    The graph records the path, not the pixels, so the file must outlive the run's
    use of it. Harnesses that don't send images to their model ignore it.
    """
    repl = current_repl()
    if repl is None:
        raise RuntimeError("attach_image can only be called while a node runs")
    if not repl.replaying:
        repl.node.data.setdefault(IMAGES_KEY, []).append(os.path.abspath(os.fspath(path)))


def record_metadata(
    key: str, value: Any, *, append: bool = False, node: Node | None = None
) -> None:
    """Set ``metadata[key]`` on ``node``, or on the node whose code is running.

    ``append`` keeps a list at ``key``. Values must be JSON. Nothing is recorded
    while code is rerun to rebuild a REPL.
    """
    repl = current_repl()
    if repl is not None and repl.replaying:
        return
    target = node or (repl.node if repl is not None else None)
    if target is None:
        raise RuntimeError("metadata can only be recorded while a node runs")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("metadata key must be a non-empty string")
    stored = strict_json(value, f"metadata {key!r}")
    if not append:
        target.metadata[key] = stored
        return
    existing = target.metadata.setdefault(key, [])
    if not isinstance(existing, list):
        raise TypeError(f"metadata {key!r} is not a list")
    existing.append(stored)


__all__ = ["Repl", "attach_image", "current_repl", "record_metadata", "running"]
