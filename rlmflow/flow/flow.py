"""Run agents: step every run that has something to do, and stream every node as it finishes."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from inspect import isawaitable
from typing import Any, Literal, TypeAlias

from rlmflow.agent_tools.server import AgentToolsServer
from rlmflow.agents.agent import MAX_ITERS_EXCEEDED, Agent, steps
from rlmflow.flow import execution
from rlmflow.flow.stream import Stream
from rlmflow.graph.config import MAX_OUTPUT, validate_agent_name
from rlmflow.graph.edits import Invalidation
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    DONE,
    IDLE_KEY,
    RETURN,
    SEEN_KEY,
    SELF,
    STOPPED_KEY,
    TERMINAL_KEY,
    WORKSPACE_KEY,
    AgentRoot,
    Node,
    final_node,
)
from rlmflow.graph.queries import ready_runs, unseen_results
from rlmflow.hooks import Stop
from rlmflow.llm.base import Reply, client_for, complete
from rlmflow.llm.openai import OpenAIClient
from rlmflow.repl import Repl, running
from rlmflow.results import RESULT_MAX, json_safe, render_results, result_name
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.runtime import repl_key
from rlmflow.tools.builtins import query
from rlmflow.tools.declare import RESERVED_TOOLS, Toolkit, as_tool, as_tool_items, tool_name
from rlmflow.workspaces import LocalWorkspace, make_workspaces

#: Where a stream stops: after any node a run wrote itself (``"next"``), after one
#: that ran code or ended a step (``"idle"``), at the end (``"done"``), or when a
#: predicate on the streamed node holds.
Until: TypeAlias = Literal["next", "idle", "done"] | Callable[[Node], bool]


def _boundary(until: Until) -> Callable[[Node], bool] | None:
    if until == "done":
        return None
    if callable(until):
        return until
    if until == "next":
        return lambda node: node.producer == SELF
    if until == "idle":
        return lambda node: node.producer == SELF and (node.runs_code or node.ends_turn)
    raise ValueError(f"unknown until boundary: {until!r}")


class Flow:
    """Agents, model clients, the REPL runtime, tools and workspaces. Steps runs.

    ``agents`` are the named kinds of agent a run uses: ``flow.start(name, task)``
    and ``launch_agent(task, agent=name)`` pick one (a launch defaults to the
    launcher's own); without ``agents`` there is one, ``"default"``. The graph
    only names each run's agent. An agent's ``model`` is any model name, or a
    name in ``llm_clients`` for clients a name can't describe. ``workers`` caps
    how many model calls run at once.

    ``workspaces`` is where runs' files live: ``"copies"`` (top-level runs in
    the runtime's working directory, launched runs in copies of their
    launcher's files), ``"isolated"`` (every run in a copy), ``"shared"`` (one
    directory), or an object such as ``GitWorktrees``. ``None`` is
    ``"copies"`` when the REPL runs on this machine in a directory you gave the
    runtime, and ``"shared"`` otherwise.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, Agent] | None = None,
        tools: Any = (),
        runtime: Runtime | None = None,
        workspaces: Any = None,
        llm_clients: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> None:
        self.agents: dict[str, Agent] = {"default": Agent()} if agents is None else {}
        for name, agent in (agents or {}).items():
            validate_agent_name(name)
            if not isinstance(agent, Agent):
                raise TypeError(f"agent {name!r} must be an Agent, not {type(agent).__name__}")
            self.agents[name] = agent
        self.llms: dict[str, Any] = dict(llm_clients or {})
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self._model_slots = asyncio.Semaphore(workers) if workers else None
        self.runtime = runtime or LocalRuntime()
        self.workspaces = make_workspaces(workspaces, self.runtime)
        if not isinstance(self.workspaces, LocalWorkspace):
            self.runtime.directory_for = self.workspaces.path
        #: Tools and values every run's REPL binds, by name.
        self.tools: dict[str, Any] = {}
        #: Toolkits each stream opens a copy of; their tools exist only in that stream.
        self.toolkits: list[Toolkit] = []
        for item in as_tool_items(tools):
            self.add_tool(item)
        #: What ``Repl.set`` bound, by REPL id; each run of that REPL binds it again.
        self._bound: dict[str, dict[str, Any]] = {}
        #: The live stream of each graph this Flow is running, by graph id.
        self._streams: dict[str, Stream] = {}
        #: ``(graph id, run id)`` of runs whose agent is taking a step now.
        self._stepping: set[tuple[str, str]] = set()
        self._agent_tools: AgentToolsServer | None = None

    def agent(self, node: Node) -> Agent:
        """The agent ``node``'s run runs with."""
        name = node.root.agent
        try:
            return self.agents[name]
        except KeyError:
            known = ", ".join(sorted(self.agents))
            raise KeyError(f"no agent named {name!r} on this Flow (known: {known})") from None

    # -- models -------------------------------------------------------------

    def llm(self, model: str | None) -> Any:
        """The client for ``model``."""
        if model is None:
            raise ValueError("no model: set Agent(model=...) on the agent this run runs with")
        if model not in self.llms:
            self.llms[model] = client_for(model)
        return self.llms[model]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        *,
        reasoning_effort: str | None = None,
        **kwargs: Any,
    ) -> Reply:
        """One model call; a client's ``request_timeout`` caps it."""
        client = self.llm(model)
        if reasoning_effort is not None and isinstance(client, OpenAIClient):
            kwargs["reasoning_effort"] = reasoning_effort
        timeout = getattr(client, "request_timeout", None)
        if self._model_slots is None:
            return await complete(client, messages, timeout=timeout, **kwargs)
        async with self._model_slots:
            return await complete(client, messages, timeout=timeout, **kwargs)

    async def _model(
        self, model: str | None, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Reply:
        """The ``llm`` a step gets: one model call."""
        return await self.complete(messages, model, **kwargs)

    async def llm_query(self, prompt: str, *, model: str | None = None, schema: Any = None) -> Any:
        """One context-free model call from host code; ``model`` defaults to the ``default`` agent's."""
        default = self.agents.get("default")
        return await query(
            self, prompt, model=model or (default.model if default else None), schema=schema
        )

    # -- tools --------------------------------------------------------------

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        """Add a tool (a plain function works: its docstring describes it), a ``Toolkit``, or, with ``name``, any value to bind."""
        if isinstance(fn, Toolkit):
            self.toolkits.append(fn)
            return
        if callable(fn):
            fn = as_tool(fn)
        elif name is None:
            raise TypeError("add_tool(value) needs name= for a value that isn't a function")
        name = name or tool_name(fn)
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.tools[name] = fn

    def remove_tool(self, name: str) -> Any:
        """Stop binding ``name``; it is gone from each REPL at its next execution."""
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        return self.tools.pop(name, None)

    def tools_for(self, g: Graph | None) -> dict[str, Any]:
        """The Flow's tools plus those of the toolkits ``g``'s stream opened."""
        stream = self._stream(g)
        return {**self.tools, **stream.tools} if stream is not None else dict(self.tools)

    # -- graphs and runs -----------------------------------------------------------

    def start(self, agent: str, task: str = "", **args: Any) -> Graph:
        """A new graph with one top-level run of ``agent`` on ``task``.

        Keyword arguments become the run's REPL variables, as in ``Graph.start``.
        """
        if agent not in self.agents:
            raise KeyError(f"no agent named {agent!r} on this Flow")
        g = Graph()
        g.start(task, agent=agent, **args)
        return g

    def repl(self, node: Node) -> Repl:
        """The ``Repl`` of ``node``'s run, for host code."""
        return Repl(self, node.root)

    def launch(
        self,
        repl: Repl,
        task: str,
        *,
        agent: str,
        name: str | None = None,
        workspace: Any = None,
        **args: Any,
    ) -> AgentRoot:
        """Start a run launched by ``repl``'s running node; every harness's launches land here.

        ``workspace`` is where its files start: ``None`` for a copy of the
        launcher's, ``"shared"`` for the launcher's own directory, another run
        (a copy of its files) or a commit.
        """
        if agent not in self.agents:
            raise KeyError(f"no agent named {agent!r} on this Flow")
        g = repl.graph
        if (stream := self._stream(g)) is not None:
            if (refused := self._before_launch(stream, repl.node, task, agent)) is not None:
                raise RuntimeError(refused)
        child = g.start(task, agent=agent, name=name, parent=repl.node, **args)
        try:
            self.workspaces.branch(child, repl.root, workspace)
        except BaseException:
            g.remove(child)
            raise
        self.wake(g)
        return child

    @property
    def agent_tools(self) -> AgentToolsServer:
        """The agent-tools server CLI agents reach; started before their first step."""
        if self._agent_tools is None:
            self._agent_tools = AgentToolsServer(self)
        return self._agent_tools

    def stepping(self, run: Node) -> bool:
        """Whether ``run``'s agent is taking a step now."""
        return (run._graph().id, run.id) in self._stepping

    def append(self, g: Graph, at: Node, node: Node) -> Node:
        """Add ``node`` after the last node of ``at``'s run; a finished one streams now."""
        g.append(at, node)
        if node.state == DONE:
            self.land(node)
        return node

    def land(self, node: Node) -> None:
        """Stream a finished node written to a live graph, and wake its stream."""
        stream = self._stream(node.graph)
        if stream is not None:
            stream.land(node)

    async def _replay(self, g: Graph, run: Node | None = None) -> None:
        """Rebuild REPLs from the graph: one run's, or every unfinished run's."""
        for target in [run.root] if run is not None else g.agent_roots():
            if run is not None or target.status() != "done":
                execution.forget(self, target)
                await Repl(self, target).catch_up()

    # -- steps ----------------------------------------------------------------

    async def _advance(self, g: Graph, run: AgentRoot) -> Node:
        """Take one step for ``run``: run the node an edit left ready, or call its agent.

        It doesn't attach children's results; the stream does that before each
        step. A failure ends the run with a failure node; a top-level one
        raises. Returns the run's last node.
        """
        seen = len(run.nodes())
        started = time.time()
        try:
            agent = self.agent(run)
            ready = next((n for n in run.nodes() if n.state != DONE), None)
            last = run.nodes()[-1]
            if ready is not None:
                ready.data.setdefault(SEEN_KEY, len(ready.transcript()) - 1)
                await self.repl(run).exec(ready)
            elif agent.max_iters is not None and steps(last) >= agent.max_iters:
                self.append(g, run, final_node(MAX_ITERS_EXCEEDED, seen))
            else:
                await self._step(g, run, agent, last, seen, started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            scope = "run" if run.parent() is None else "child"
            failed = final_node(f"[{scope} failed: {text}]", seen, output=text, error=True)
            failed.started_at, failed.finished_at = started, time.time()
            if run.id in g:
                self.append(g, run, failed)
            if scope == "run":
                raise
        return run.nodes()[-1]

    async def _step(
        self, g: Graph, run: AgentRoot, agent: Agent, last: Node, seen: int, started: float
    ) -> None:
        """Call ``agent.step`` and append what it returns; run its code, or mark how the run is."""
        if agent.cli:
            await self.agent_tools.start()
        repl = Repl(self, run)
        key = (g.id, run.id)
        self._stepping.add(key)
        try:
            with running(repl):
                node = await agent.step(last, self._model, repl)
        finally:
            self._stepping.discard(key)
        if not isinstance(node, Node):
            raise TypeError(f"{type(agent).__name__}.step must return a Node")
        node.data[SEEN_KEY] = seen
        node.started_at = node.started_at or started
        if node.runs_code:
            self.append(g, run, node)
            await repl.exec(node)
            return
        # Code nodes finish in execution.run_node. Only a non-code reply can
        # directly end or idle an agent here.
        if node.result is not None:
            node.data[TERMINAL_KEY] = True
        elif not node.error:
            node.data[IDLE_KEY] = True
        if agent.cli and (workspace := await self.workspaces.snapshot(run)) is not None:
            node.data[WORKSPACE_KEY] = workspace
        node.finished_at = time.time()
        self.append(g, run, node)

    async def step(self, g: Graph, run: Node | None = None) -> list[Node]:
        """Take one step for every ready run of ``g``, or just for ``run``'s; returns their last nodes.

        Finished children's results are attached first. This is a manual,
        hookless step; use ``stream`` when hooks, budgets or streamed nodes
        must run.
        """
        targets = [run.root] if run is not None else ready_runs(g)
        for target in targets:
            self.deliver_results(g, target)
        return list(await asyncio.gather(*(self._advance(g, target) for target in targets)))

    async def resume(self, g: Graph, *, hooks: Iterable[Any] = ()) -> Graph:
        """Rebuild every unfinished run's REPL from the graph, then run ``g`` to the end."""
        await self._replay(g)
        async for _node in self.stream(g, hooks=hooks):
            pass
        return g

    def deliver_results(self, g: Graph, run: Node) -> Node | None:
        """Attach the results of ``run``'s finished children that nothing read yet.

        One user node binds each result as a variable named after its child.
        Never to a run that finished, and not while an edit has left a node to
        rerun, whose ``await`` takes the results itself. The stream calls this
        before scheduling a step and once more while draining; manual ``step``
        calls it directly.
        """
        run = run.root
        if run.status() == "done" or any(node.state != DONE for node in run.nodes()):
            return None
        finished = unseen_results(g, run)
        if not finished:
            return None
        limit = min(RESULT_MAX, MAX_OUTPUT)
        taken = self.tools_for(g).keys()
        inputs: dict[str, Any] = {}
        items = []
        for child, final in finished:
            name = result_name(child.name, inputs.keys() | taken, RESERVED_TOOLS)
            inputs[name] = json_safe(final.result)
            items.append((child.path(), child.agent, final.result))
        node = Node(
            "user",
            render_results(items, limit) + "\nSet: " + ", ".join(f"`{name}`" for name in inputs),
            input=inputs,
            producer="rlmflow",
        )
        self.append(g, run, node)
        for _child, final in finished:
            g.link(RETURN, final, node)
        return node

    # -- the live stream --------------------------------------------------------

    def _stream(self, g: Graph | None, *, required: bool = False) -> Stream | None:
        stream = self._streams.get(g.id) if g is not None else None
        if stream is None and required:
            raise RuntimeError(f"graph {g.id if g else None} is not streaming on this Flow")
        return stream

    def running(self, g: Graph | None = None) -> bool:
        """Whether ``g`` is streaming on this Flow; without a graph, whether any is."""
        return bool(self._streams) if g is None else g.id in self._streams

    def loop(self, g: Graph) -> asyncio.AbstractEventLoop | None:
        """The event loop streaming ``g``, if it is streaming."""
        stream = self._stream(g)
        return stream.loop if stream is not None else None

    def elapsed(self, g: Graph) -> float:
        """Seconds since ``g``'s stream started."""
        return self._stream(g, required=True).elapsed()

    def call_later(self, g: Graph, delay: float, fn: Callable[[], Any]) -> None:
        """Run ``fn`` on ``g``'s stream after ``delay`` seconds; it may return an awaitable."""
        self._stream(g, required=True).call_later(delay, fn)

    def wake(self, g: Graph | None = None) -> None:
        """Tell ``g``'s stream (every stream, without one) the graph changed; safe from any thread."""
        streams = list(self._streams.values()) if g is None else [self._stream(g)]
        for stream in streams:
            if stream is not None:
                stream.wake()

    async def wait_for(self, g: Graph, predicate: Callable[[], bool]) -> None:
        """Wait until ``predicate()`` holds; it is rechecked whenever ``g`` changes."""
        await self._stream(g, required=True).wait_for(predicate)

    async def _cancel(self, g: Graph, run_ids: Iterable[str]) -> None:
        stream = self._stream(g)
        if stream is not None:
            await stream.cancel(list(run_ids))

    async def edit(
        self, g: Graph, op: str, target: Node, *args: Any, **kwargs: Any
    ) -> Invalidation:
        """Apply ``g.<op>(target, ...)`` to a graph this Flow may be running.

        The runs the edit touches stop first, so no step writes over it.
        """
        if op not in ("edit", "delete", "remove", "replace"):
            raise ValueError(f"unknown edit {op!r}")
        preview = getattr(g, op)(target, *args, dry_run=True, **kwargs)
        await self._cancel(g, [*preview.rewound, *preview.removed])
        invalidation = getattr(g, op)(target, *args, **kwargs)
        self.wake(g)
        return invalidation

    async def stop(
        self, node: Node, *, reason: str = "stopped", subtree: bool = True
    ) -> list[Node]:
        """End ``node``'s run now (with every unfinished run it launched, unless ``subtree=False``).

        Its running step is cancelled; a node it was running is closed as
        stopped. Each stopped run gets a terminal node whose result is
        ``{"stopped": reason}``, which is what a parent awaiting it receives.
        A message reopens a stopped run like any finished one. Returns the
        terminal nodes.
        """
        g, run = node._graph(), node.root
        targets = [r for r in (run.subtree() if subtree else [run]) if r.status() != "done"]
        await self._cancel(g, [r.id for r in targets])
        note = f"[stopped: {reason}]"
        ended: list[Node] = []
        now = time.time()
        for target in targets:
            if target.id not in g:
                continue
            for earlier in target.nodes():
                if earlier.state != DONE:
                    earlier.state, earlier.error, earlier.finished_at = DONE, True, now
                    earlier.output = earlier.output or note
                    earlier.data[STOPPED_KEY] = reason
            final = final_node(
                {"stopped": reason}, len(target.nodes()), output=note, data={STOPPED_KEY: reason}
            )
            final.started_at = final.finished_at = now
            ended.append(self.append(g, target, final))
        self.wake(g)
        return ended

    def _check_can_stream(self, g: Graph) -> None:
        if g.id in self._streams:
            raise RuntimeError(f"graph {g.id} is already streaming on this Flow")
        held = {repl_key(r) for stream in self._streams.values() for r in stream.g.agent_roots()}
        shared = held & {repl_key(run) for run in g.agent_roots()}
        if shared:
            raise RuntimeError(
                "another graph streaming on this Flow uses the same REPLs "
                f"({', '.join(sorted(shared))}); fork with repl='copy' to run both"
            )

    async def stream(
        self,
        g: Graph,
        *,
        until: Until = "done",
        hooks: Iterable[Any] = (),
    ) -> AsyncIterator[Node]:
        """Step every run that has something to do, and yield each node as it finishes.

        One Flow can stream several graphs at once, but not the same graph
        twice, nor two graphs whose runs share a REPL (``g.fork(node,
        repl="same")``) while both are live. ``hooks`` see and steer this
        stream, a ``Budget`` among them (see ``rlmflow.hooks``).
        """
        self._check_can_stream(g)
        stop_at = _boundary(until)
        stream = self._streams[g.id] = Stream(g, list(hooks))
        try:
            async with self._live(stream):
                while True:
                    stream.mark_seen()
                    await self._run_due(stream)
                    stream.reap()
                    if stream.landed:
                        stop = False
                        for node in stream.drain():
                            await self._on_node(stream, node)
                            yield node
                            stop = stop or (stop_at is not None and stop_at(node))
                        if stop:
                            return
                        # The caller may have written to the graph between nodes.
                        continue
                    for run in ready_runs(g, stream.tasks):
                        self.agent(run)
                        self.deliver_results(g, run)
                        if await self._before_step(stream, run):
                            self._start(stream, run)
                    if stream.landed:
                        continue
                    if not stream.tasks:
                        return
                    await stream.wait_for_change()
        finally:
            del self._streams[g.id]

    @asynccontextmanager
    async def _live(self, stream: Stream) -> AsyncIterator[None]:
        """Open toolkit copies and start hooks; on the way out, however it ends, wind everything down.

        Winding down cancels the steps, attaches results that finished, hands
        the hooks every node nobody saw, ends the hooks that started, and
        closes the toolkits. A hook error there surfaces only if nothing else
        is already failing.
        """
        errors: list[BaseException] = []
        failing = True
        try:
            for toolkit in self.toolkits:
                # A copy per stream, so two live streams never share one's resources.
                opened = copy.copy(toolkit)
                fns = [as_tool(fn) for fn in await opened.open()]
                stream.toolkits.append(opened)
                stream.tools.update({tool_name(fn): fn for fn in fns})
            for hook in stream.hooks:
                stream.started_hooks.append(hook)
                if (start := getattr(hook, "start", None)) is not None:
                    start(self, stream.g)
            yield
            failing = False
        finally:
            try:
                for timer in stream.timers:
                    timer.cancel()
                await stream.cancel(list(stream.tasks))
                for run in stream.g.agent_roots():
                    self.deliver_results(stream.g, run)
                for node in stream.drain():
                    try:
                        await self._on_node(stream, node)
                    except Exception as error:  # noqa: BLE001 - surfaced below
                        errors.append(error)
                        break
            finally:
                for hook in reversed(stream.started_hooks):
                    if (end := getattr(hook, "end", None)) is not None:
                        try:
                            end(self, stream.g)
                        except Exception as error:  # noqa: BLE001 - surfaced below
                            errors.append(error)
                for toolkit in reversed(stream.toolkits):
                    await toolkit.close()
        if errors and not failing:
            raise errors[0]

    async def _run_due(self, stream: Stream) -> None:
        while stream.due:
            result = stream.due.pop(0)()
            if isawaitable(result):
                await result

    async def _on_node(self, stream: Stream, node: Node) -> None:
        for hook in stream.hooks:
            if (on_node := getattr(hook, "on_node", None)) is not None:
                result = on_node(self, node)
                if isawaitable(result):
                    await result

    def _before_launch(self, stream: Stream, node: Node, task: str, agent: str) -> str | None:
        """The first hook's reason to refuse this launch, or ``None``."""
        for hook in stream.hooks:
            ask = getattr(hook, "before_launch", None)
            if ask is not None and (refused := ask(self, node, task, agent)) is not None:
                return refused
        return None

    async def _before_step(self, stream: Stream, run: AgentRoot) -> bool:
        """Ask the hooks about ``run``'s step; whether it should start."""
        notes: list[Node] = []
        for hook in stream.hooks:
            ask = getattr(hook, "before_step", None)
            decision = ask(self, run) if ask is not None else None
            if isinstance(decision, Stop):
                await self.stop(run, reason=decision.reason, subtree=False)
                return False
            if decision is not None:
                notes.append(decision)
        for note in notes:
            self.append(stream.g, run, note)
        return True

    def _start(self, stream: Stream, run: AgentRoot) -> None:
        task = asyncio.ensure_future(self._advance(stream.g, run))
        task.add_done_callback(lambda _task: stream.wake())
        stream.tasks[run.id] = task

    # -- running to the end -------------------------------------------------------

    async def arun(
        self,
        target: Graph | str,
        task: str | None = None,
        *,
        hooks: Iterable[Any] = (),
        **args: Any,
    ) -> Any:
        """Run to the end: ``arun(agent, task, **args)`` returns the run's result; ``arun(g)`` returns ``g``."""
        if isinstance(target, str):
            if task is None:
                raise TypeError("run(agent, task): name the agent, then give the task")
            g = self.start(target, task, **args)
            root = g.root
        elif task is not None or args:
            raise TypeError("a task and arguments can only be passed with an agent name")
        else:
            g, root = target, None
        async for _node in self.stream(g, hooks=hooks):
            pass
        return g if root is None else root.final_result()

    def run(
        self,
        target: Graph | str,
        task: str | None = None,
        *,
        hooks: Iterable[Any] = (),
        **args: Any,
    ) -> Any:
        """``arun`` from synchronous code."""
        return asyncio.run(self.arun(target, task, hooks=hooks, **args))

    def close(self) -> None:
        """Close the REPLs and the agent-tools server."""
        if self._agent_tools is not None:
            self._agent_tools.close()
            self._agent_tools = None
        self.runtime.close()

    async def aclose(self) -> None:
        """Cancel live streams, and close the REPLs, the agent-tools server and model clients."""
        for stream in list(self._streams.values()):
            await stream.cancel(list(stream.tasks))
        owners = {id(client): client for client in self.llms.values()}
        closing = [
            close() for client in owners.values() if (close := getattr(client, "aclose", None))
        ]
        try:
            await asyncio.gather(*closing)
        finally:
            self.close()


__all__ = ["Flow", "Until"]
