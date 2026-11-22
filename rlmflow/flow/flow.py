"""Run agents: give each agent that has something to do a turn, and stream every node as it finishes."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import asynccontextmanager
from inspect import isawaitable
from typing import Any, Literal, TypeAlias

from rlmflow.agent_tools.server import AgentToolsServer
from rlmflow.flow import execution
from rlmflow.flow.stream import Stream
from rlmflow.graph.config import AgentConfig, can_spawn
from rlmflow.graph.edits import Invalidation
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    DONE,
    RETURN,
    SEEN_KEY,
    SELF,
    STOPPED_KEY,
    LLMUsage,
    Node,
    Step,
    final_node,
    running_step,
)
from rlmflow.graph.queries import ready_agents, unseen_results
from rlmflow.harnesses.prompts import STATE_PROMPT, VERIFY_PROMPT, skill
from rlmflow.hooks import Stop
from rlmflow.llm import client_for, complete
from rlmflow.results import RESULT_MAX, json_safe, render_results, result_name
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.repl import ReplRun
from rlmflow.runtime.runtime import repl_key
from rlmflow.tools import (
    RESERVED_TOOLS,
    Toolkit,
    as_tool,
    as_tool_items,
    query,
    tool_name,
)
from rlmflow.workspaces import LocalWorkspace, make_workspaces

#: Where a streamed run stops: after any node of an agent's own (``"next"``), after a
#: node that ran code or ended a turn (``"idle"``), at the end (``"done"``), or when
#: a predicate on the streamed ``Step`` holds.
StepUntil: TypeAlias = Literal["next", "idle", "done"] | Callable[[Step], bool]


def _boundary(until: StepUntil) -> Callable[[Step], bool] | None:
    if until == "done":
        return None
    if callable(until):
        return until
    if until == "next":
        return lambda step: step.node.producer == SELF
    if until == "idle":
        return lambda step: (
            step.node.producer == SELF and (step.node.runs_code or step.node.ends_turn)
        )
    raise ValueError(f"unknown until boundary: {until!r}")


class Flow:
    """Model clients, the REPL runtime, tools and workspaces. Runs agents' turns.

    The Flow decides how every agent runs. ``configs`` are the named kinds of
    agent that ``launch_agent(config=...)`` and ``flow.init(task, config=...)``
    pick from (``"default"`` is ``AgentConfig()`` unless given). The graph
    only names each agent's config and records the JSON of the one that ran
    it. A config's ``model`` is any model name, or a name in ``llm_clients``
    for clients a name can't describe. ``workers`` caps how many model calls
    run at once.

    ``workspaces`` is where agents' files live. ``"auto"``: when the runtime
    has a working directory on this machine, top-level agents work in it and
    each launched agent in a copy of its launcher's files; otherwise all share
    one directory. ``"shared"``, ``"isolated"`` (top-level agents get copies
    too), or an object such as ``GitWorktrees``.
    """

    def __init__(
        self,
        *,
        llm_clients: dict[str, Any] | None = None,
        configs: Iterable[AgentConfig] = (),
        runtime: Runtime | None = None,
        tools: Any = None,
        workspaces: Any = "auto",
        workers: int | None = None,
    ) -> None:
        self.llms: dict[str, Any] = dict(llm_clients or {})
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self._model_slots = asyncio.Semaphore(workers) if workers else None
        self.runtime = runtime or LocalRuntime()
        self.workspaces = make_workspaces(workspaces, self.runtime)
        if not isinstance(self.workspaces, LocalWorkspace):
            self.runtime.directory_for = self.workspaces.path
        #: Tools and values every agent's REPL binds, by name.
        self.tools: dict[str, Any] = {}
        #: Toolkits each stream opens a copy of; their tools exist only in that stream.
        self.toolkits: list[Toolkit] = []
        for item in as_tool_items(tools):
            self.add_tool(item)
        self.configs: dict[str, AgentConfig] = {"default": AgentConfig()}
        for config in configs:
            self.add_config(config)
        #: The live stream of each graph this Flow is running, by graph id.
        self._streams: dict[str, Stream] = {}
        self._agent_tools: AgentToolsServer | None = None

    # -- models -------------------------------------------------------------

    def llm(self, model: str | None) -> Any:
        """The client for ``model``."""
        if model is None:
            raise ValueError(
                "no model: set AgentConfig(model=...) on the config this agent runs with"
            )
        if model not in self.llms:
            self.llms[model] = client_for(model)
        return self.llms[model]

    async def complete(
        self, messages: list[dict[str, Any]], model: str | None, **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        """One model call; a client's ``request_timeout`` caps it."""
        client = self.llm(model)
        timeout = getattr(client, "request_timeout", None)
        if self._model_slots is None:
            return await complete(client, messages, timeout=timeout, **kwargs)
        async with self._model_slots:
            return await complete(client, messages, timeout=timeout, **kwargs)

    async def llm_query(self, prompt: str, *, model: str | None = None, schema: Any = None) -> Any:
        """One context-free model call from host code; ``model`` defaults to the ``default`` config's."""
        return await query(
            self, prompt, model=model or self.configs["default"].model, schema=schema
        )

    # -- configs ------------------------------------------------------------

    def add_config(self, config: AgentConfig) -> AgentConfig:
        """Name a kind of agent that ``launch_agent(config=config.name)`` starts."""
        if not isinstance(config, AgentConfig):
            raise TypeError(f"add_config takes an AgentConfig, not {type(config).__name__}")
        self.configs[config.name] = config
        return config

    def config(self, name: str) -> AgentConfig:
        if name not in self.configs:
            raise KeyError(
                f"no config named {name!r} on this Flow (known: {', '.join(sorted(self.configs))})"
            )
        return self.configs[name]

    def config_for(self, agent: Node, *, record: bool = True) -> AgentConfig:
        """Bind and return the config ``agent`` runs with: the one it names, else ``"default"``."""
        config = self.config(agent.config_name or "default")
        agent.bind_config(config, record=record)
        return config

    def _bind_configs(self, g: Graph) -> None:
        """Bind every agent's config; finished agents keep their record, and may name a gone one."""
        for agent in g.agents():
            if agent.status() != "done":
                self.config_for(agent)
            elif agent.config_name is None or agent.config_name in self.configs:
                self.config_for(agent, record=False)

    # -- tools --------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        """Bind ``value`` as ``name`` in every agent's REPL, from its next execution."""
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.tools[name] = value

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        """Add a tool (a plain function works: its docstring describes it) or a ``Toolkit``."""
        if isinstance(fn, Toolkit):
            self.toolkits.append(fn)
            return
        fn = as_tool(fn)
        self.inject(name or tool_name(fn), fn)

    def remove_tool(self, name: str) -> Any:
        """Stop binding ``name``; it is gone from each REPL at its next execution."""
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        return self.tools.pop(name, None)

    def tools_for(self, g: Graph | None) -> dict[str, Any]:
        """The Flow's tools plus those of the toolkits ``g``'s stream opened."""
        stream = self._stream(g)
        return {**self.tools, **stream.tools} if stream is not None else dict(self.tools)

    def namespace_for(self, step: Step) -> dict[str, Any]:
        """Extra bindings for this agent's REPL; none by default."""
        return {}

    # -- prompts ----------------------------------------------------------------

    def guidance(self, step: Step) -> dict[str, str | None]:
        """Prompt sections every harness includes; the orchestrate skill only for agents that can launch."""
        return {
            "verify": VERIFY_PROMPT,
            "state": STATE_PROMPT,
            "orchestration": skill("orchestrate") if can_spawn(step.graph, step.agent) else None,
        }

    def turn_prompt(self, step: Step) -> str:
        """What the agent is told at the start of each turn: the agents it launched and how they are."""
        children = step.agent.children()
        if not children:
            return ""
        lines = [
            f"- `{child.name}` ({child.path()}): "
            + ("result ready" if child.status() == "done" else child.status())
            for child in children
        ]
        return "\n".join(["Background agents:", *lines])

    def spawn(
        self,
        step: Step,
        task: str,
        *,
        config: str = "default",
        name: str | None = None,
        workspace: Any = None,
        **args: Any,
    ) -> Node:
        """Create an agent launched by the running node; every harness's launches land here.

        ``workspace`` is where its files start: ``None`` for a copy of the
        launcher's, ``"shared"`` for the launcher's own directory, another
        agent (a copy of its files) or a commit.
        """
        self.config(config)
        g = step.graph
        if (stream := self._stream(g)) is not None:
            if (refused := self._before_launch(stream, step, task, config)) is not None:
                raise RuntimeError(refused)
        child = g.start(task, parent=step.node, config=config, name=name, **args)
        try:
            self.config_for(child)
            self.workspaces.branch(child, step.agent, workspace)
        except BaseException:
            g.remove(child)
            raise
        self.wake(g)
        return child

    @property
    def agent_tools(self) -> AgentToolsServer:
        """The agent-tools server external harnesses reach; started on first use."""
        if self._agent_tools is None:
            self._agent_tools = AgentToolsServer(self)
        return self._agent_tools

    # -- nodes ----------------------------------------------------------------

    def append(self, g: Graph, agent: Node, node: Node) -> Node:
        """Add ``node`` after ``agent``'s last node; a finished one streams now."""
        g.append(agent, node)
        if node.state == DONE:
            self.land(node)
        return node

    def land(self, node: Node) -> None:
        """Stream a finished node written to a live graph, and wake its stream."""
        stream = self._stream(node.graph)
        if stream is not None:
            stream.land(node)

    async def run_node(self, g: Graph, agent: Node, node: Node) -> ReplRun | None:
        """Run one of ``agent``'s nodes in its REPL; see ``execution.run_node``."""
        return await execution.run_node(self, g, agent, node)

    async def prepare_repl(self, g: Graph, agent: Node) -> None:
        """Bring ``agent``'s REPL up to its last node."""
        await execution.prepare(self, g, agent)

    async def replay(self, g: Graph, agent: Node | None = None) -> None:
        """Rebuild REPLs from the graph: one agent's, or every unfinished agent's."""
        for target in [agent] if agent is not None else g.agents():
            if agent is not None or target.status() != "done":
                self.config_for(target, record=False)
                execution.forget(self, target)
                await self.prepare_repl(g, target)

    # -- turns ----------------------------------------------------------------

    async def step(self, g: Graph, agent: Node) -> Node:
        """Take one turn for ``agent``: run the node an edit left ready, or call its harness.

        It doesn't attach children's results; the stream does that before each
        turn, and host code calls ``deliver_results`` first. A failure ends
        the agent with a failure node; a top-level one raises. Returns the
        agent's last node.
        """
        self.config_for(agent)
        seen = len(agent.nodes())
        started = time.time()
        try:
            ready = next((n for n in agent.nodes() if n.state != DONE), None)
            if ready is not None:
                ready.data.setdefault(SEEN_KEY, len(ready.transcript()) - 1)
                await self.run_node(g, agent, ready)
            else:
                step = Step(g, agent, agent.nodes()[-1], flow=self)
                with running_step(step):
                    await agent.config.load_harness()(self, step)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            scope = "run" if agent.parent() is None else "child"
            failed = final_node(f"[{scope} failed: {text}]", seen, output=text, error=True)
            failed.started_at, failed.finished_at = started, time.time()
            if agent.id in g:
                self.append(g, agent, failed)
            if scope == "run":
                raise
        return agent.nodes()[-1]

    def deliver_results(self, g: Graph, agent: Node) -> Node | None:
        """Attach the results of ``agent``'s finished children that nothing read yet.

        One user node binds each result as a variable named after its child.
        Call it between turns. Never to an agent that finished, and not while
        an edit has left a node to rerun, whose ``await`` takes the results itself.
        """
        if agent.status() == "done" or any(node.state != DONE for node in agent.nodes()):
            return None
        finished = unseen_results(g, agent)
        if not finished:
            return None
        limit = min(RESULT_MAX, agent.config.max_output_length)
        taken = self.tools_for(g).keys()
        inputs: dict[str, Any] = {}
        items = []
        for child, final in finished:
            name = result_name(child.name, inputs.keys() | taken, RESERVED_TOOLS)
            inputs[name] = json_safe(final.result)
            items.append((child.path(), child.config_name or "root", final.result))
        node = Node(
            "user",
            render_results(items, limit) + "\nSet: " + ", ".join(f"`{name}`" for name in inputs),
            input=inputs,
            producer="rlmflow",
        )
        self.append(g, agent, node)
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

    async def _cancel(self, g: Graph, agent_ids: Iterable[str]) -> None:
        stream = self._stream(g)
        if stream is not None:
            await stream.cancel(list(agent_ids))

    async def edit(
        self, g: Graph, op: str, target: Node, *args: Any, **kwargs: Any
    ) -> Invalidation:
        """Apply ``g.<op>(target, ...)`` to a graph this Flow may be running.

        The agents the edit touches stop first, so no turn writes over it.
        """
        if op not in ("edit", "delete", "remove", "replace"):
            raise ValueError(f"unknown edit {op!r}")
        preview = getattr(g, op)(target, *args, dry_run=True, **kwargs)
        await self._cancel(g, [*preview.rewound, *preview.removed])
        invalidation = getattr(g, op)(target, *args, **kwargs)
        self.wake(g)
        return invalidation

    async def stop(self, g: Graph, agent: Node, *, reason: str, subtree: bool = True) -> list[Node]:
        """End ``agent`` now (with every unfinished agent it launched, unless ``subtree=False``).

        Its running turn is cancelled; a node it was running is closed as
        stopped. Each stopped agent gets a terminal node whose result is
        ``{"stopped": reason}``, which is what a parent awaiting it receives.
        A message reopens a stopped agent like any finished one. Returns the
        terminal nodes.
        """
        targets = [a for a in (agent.subtree() if subtree else [agent]) if a.status() != "done"]
        await self._cancel(g, [a.id for a in targets])
        note = f"[stopped: {reason}]"
        ended: list[Node] = []
        now = time.time()
        for target in targets:
            if target.id not in g:
                continue
            for node in target.nodes():
                if node.state != DONE:
                    node.state, node.error, node.finished_at = DONE, True, now
                    node.output = node.output or note
                    node.data[STOPPED_KEY] = reason
            node = final_node(
                {"stopped": reason}, len(target.nodes()), output=note, data={STOPPED_KEY: reason}
            )
            node.started_at = node.finished_at = now
            ended.append(self.append(g, target, node))
        self.wake(g)
        return ended

    def _check_can_stream(self, g: Graph) -> None:
        if g.id in self._streams:
            raise RuntimeError(f"graph {g.id} is already streaming on this Flow")
        held = {repl_key(a) for stream in self._streams.values() for a in stream.g.agents()}
        shared = held & {repl_key(agent) for agent in g.agents()}
        if shared:
            raise RuntimeError(
                "another graph streaming on this Flow uses the same REPLs "
                f"({', '.join(sorted(shared))}); fork with repl='copy' to run both"
            )

    async def run_streaming(
        self,
        g: Graph,
        *,
        until: StepUntil = "done",
        hooks: Iterable[Any] = (),
    ) -> AsyncIterator[Step]:
        """Run every agent that has something to do, and yield a ``Step`` for each node as it finishes.

        Every agent runs with the config it names, ``"default"`` when it names
        none. One Flow can stream several graphs at once, but not the same
        graph twice, nor two graphs whose agents share a REPL
        (``g.fork(node, repl="same")``) while both are live. ``hooks`` see and
        steer this stream, a ``Budget`` among them (see ``rlmflow.hooks``).
        """
        self._check_can_stream(g)
        stop_at = _boundary(until)
        stream = self._streams[g.id] = Stream(g, list(hooks))
        try:
            self._bind_configs(g)
            async with self._live(stream):
                while True:
                    stream.mark_seen()
                    await self._run_due(stream)
                    stream.reap()
                    if stream.landed:
                        stop = False
                        for step in self._drain(stream):
                            await self._on_step(stream, step)
                            yield step
                            stop = stop or (stop_at is not None and stop_at(step))
                        if stop:
                            return
                        # The caller may have written to the graph between steps.
                        continue
                    for agent in ready_agents(g, stream.tasks):
                        self.config_for(agent)
                        self.deliver_results(g, agent)
                        if await self._before_turn(stream, agent):
                            self._start_turn(stream, agent)
                    if stream.landed:
                        continue
                    if not stream.tasks:
                        return
                    await stream.wait_for_change()
        finally:
            del self._streams[g.id]

    # -- what the loop does with a stream -------------------------------------------

    @asynccontextmanager
    async def _live(self, stream: Stream) -> AsyncIterator[None]:
        """Open toolkit copies and start hooks; on the way out, however it ends, wind everything down.

        Winding down cancels the turns, attaches results that finished, hands
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
                for agent in stream.g.agents():
                    self.deliver_results(stream.g, agent)
                for step in self._drain(stream):
                    try:
                        await self._on_step(stream, step)
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

    def _drain(self, stream: Stream) -> Iterator[Step]:
        for node in stream.drain():
            yield Step(stream.g, node.agent, node, flow=self)

    async def _on_step(self, stream: Stream, step: Step) -> None:
        for hook in stream.hooks:
            if (on_step := getattr(hook, "on_step", None)) is not None:
                result = on_step(self, step)
                if isawaitable(result):
                    await result

    def _before_launch(self, stream: Stream, step: Step, task: str, config: str) -> str | None:
        """The first hook's reason to refuse this launch, or ``None``."""
        for hook in stream.hooks:
            ask = getattr(hook, "before_launch", None)
            if ask is not None and (refused := ask(self, step, task, config)) is not None:
                return refused
        return None

    async def _before_turn(self, stream: Stream, agent: Node) -> bool:
        """Ask the hooks about ``agent``'s turn; whether it should start."""
        notes: list[Node] = []
        for hook in stream.hooks:
            ask = getattr(hook, "before_turn", None)
            decision = ask(self, agent) if ask is not None else None
            if isinstance(decision, Stop):
                await self.stop(stream.g, agent, reason=decision.reason, subtree=False)
                return False
            if decision is not None:
                notes.append(decision)
        for note in notes:
            self.append(stream.g, agent, note)
        return True

    def _start_turn(self, stream: Stream, agent: Node) -> None:
        task = asyncio.ensure_future(self.step(stream.g, agent))
        task.add_done_callback(lambda _task: stream.wake())
        stream.tasks[agent.id] = task

    # -- running to the end -------------------------------------------------------

    def init(self, task: str = "", *, config: str | None = None, **args: Any) -> Graph:
        """A new graph with one top-level agent on ``task``, running config ``config``.

        Keyword arguments become the agent's REPL variables, as in ``Graph.start``.
        """
        if config is not None:
            self.config(config)
        g = Graph()
        g.start(task, config=config, **args)
        return g

    async def arun(
        self,
        target: Graph | str,
        *,
        config: str | None = None,
        hooks: Iterable[Any] = (),
        **args: Any,
    ) -> Any:
        """Run to the end. A graph returns itself; a task string returns its result."""
        if isinstance(target, str):
            g = self.init(target, config=config, **args)
            agent = g.root
        elif args or config is not None:
            raise TypeError("config and agent arguments can only be passed with a task string")
        else:
            g, agent = target, None
        async for _step in self.run_streaming(g, hooks=hooks):
            pass
        return g if agent is None else agent.final_result()

    def run(
        self,
        target: Graph | str,
        *,
        config: str | None = None,
        hooks: Iterable[Any] = (),
        **args: Any,
    ) -> Any:
        return asyncio.run(self.arun(target, config=config, hooks=hooks, **args))

    async def aclose(self) -> None:
        for stream in list(self._streams.values()):
            await stream.cancel(list(stream.tasks))
        if self._agent_tools is not None:
            self._agent_tools.close()
            self._agent_tools = None
        owners = {id(client): client for client in self.llms.values()}
        closing = [
            close() for client in owners.values() if (close := getattr(client, "aclose", None))
        ]
        try:
            await asyncio.gather(*closing)
        finally:
            self.runtime.close()


__all__ = [
    "Flow",
    "StepUntil",
]
