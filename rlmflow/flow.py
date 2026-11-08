"""Run agents: give each agent that has something to do a turn, and stream every node as it finishes."""

from __future__ import annotations

import asyncio
import json
import keyword
import re
import textwrap
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any, Literal, TypeAlias

from rlmflow.graph import calls
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.edits import GraphError, Invalidation
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    DONE,
    READY,
    RETURN,
    RUNNING,
    SELF,
    LLMUsage,
    Node,
    Step,
    running_step,
)
from rlmflow.graph.queries import unseen_results
from rlmflow.llm import client_for, complete
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.runtime.runtime import RLMFLOW_REPLAY
from rlmflow.tools import (
    RESERVED_TOOLS,
    Toolkit,
    as_tool,
    as_tool_items,
    build_namespace,
    tool_name,
)
from rlmflow.utils import truncate_output
from rlmflow.workspaces import LocalWorkspace

BUDGET_FINAL_TEXT = """\
This run has nearly used its token budget, so this is your last turn. Give the best \
supported final result now and don't start new work."""

DEAD_REPL_NOTE = """\
[this REPL died while running this code. It will be rebuilt from the recorded history \
before the next run, so variables set earlier come back; anything this code \
did is gone.]"""

#: Longest result text one agent contributes to a result node.
RESULT_MAX = 20_000

#: Where a streamed run stops: after any node of an agent's own (``"next"``), after a
#: node that ran code or ended a turn (``"idle"``), at the end (``"done"``), or when
#: a predicate on the streamed ``Step`` holds.
StepUntil: TypeAlias = Literal["next", "idle", "done", "finished"] | Callable[[Step], bool]


def _boundary(until: StepUntil) -> Callable[[Step], bool] | None:
    if until in ("done", "finished"):
        return None
    if callable(until):
        return until
    if until == "next":
        return lambda step: step.node.producer == SELF
    if until == "idle":
        return lambda step: (
            step.node.producer == SELF and (step.node.runs_code or "seen" in step.node.data)
        )
    raise ValueError(f"unknown until boundary: {until!r}")


class Flow:
    """Model clients, the REPL runtime, tools and workspaces. Runs agents' turns.

    The Flow decides how every agent runs. ``configs`` are the named kinds of
    agent that ``launch_agent(config=...)`` picks from (``"default"`` is
    ``AgentConfig()`` unless given); a root runs with the run's ``root_config``.
    The graph only names each agent's config and records the JSON of the one
    that ran it. A config's ``model`` is any model name, or a name in
    ``llm_clients`` for clients a name can't describe. ``workers`` caps how
    many model calls run at once.
    """

    def __init__(
        self,
        *,
        llm_clients: dict[str, Any] | None = None,
        configs: Iterable[AgentConfig] = (),
        runtime: Runtime | None = None,
        tools: Any = None,
        workspaces: Any = None,
        workers: int | None = None,
    ) -> None:
        self.llms: dict[str, Any] = dict(llm_clients or {})
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1")
        self._model_slots = asyncio.Semaphore(workers) if workers else None
        self.runtime = runtime or LocalRuntime()
        self.workspaces = workspaces or LocalWorkspace(self.runtime.working_directory)
        if workspaces is not None:
            self.runtime.directory_for = workspaces.path
        self.tools: dict[str, Any] = {}
        #: Toolkits opened when a run starts; their tools exist only during the run.
        self.toolkits: list[Toolkit] = []
        for item in as_tool_items(tools):
            self.add_tool(item)
        self.configs: dict[str, AgentConfig] = {"default": AgentConfig()}
        for config in configs:
            self.add_config(config)
        #: What roots run with; ``run_streaming(root_config=...)`` sets it.
        self.root_config: AgentConfig = self.configs["default"]
        #: The task running each agent's turn, by agent id.
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._graph: Graph | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._landed: list[Node] | None = None
        self._news = asyncio.Event()
        #: For each agent whose live REPL is up to date: the last node it ran, and its version.
        self._live: dict[str, tuple[str, int]] = {}
        self._repl_locks: dict[str, asyncio.Lock] = {}
        self._agent_tools: Any = None

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
        from rlmflow.tools import query

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

    def resolve_config(self, config: AgentConfig | str | None) -> AgentConfig:
        """An ``AgentConfig`` as given, a name on this Flow, or ``"default"``."""
        return config if isinstance(config, AgentConfig) else self.config(config or "default")

    def config_for(self, g: Graph, agent: Node, *, record: bool = True) -> AgentConfig:
        """Bind and return the config ``agent`` runs with: ``root_config`` for a root, else its named one."""
        name = agent.root_data.config_name
        config = self.root_config if name is None else self.config(name)
        agent.root_data.bind_config(config, record=record)
        return config

    # -- tools --------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        """Bind ``value`` as ``name`` in every agent's REPL namespace."""
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.tools[name] = value
        self.runtime.inject_live(name, value)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        """Add a tool (a plain function works: its docstring describes it) or a ``Toolkit``."""
        if isinstance(fn, Toolkit):
            self.toolkits.append(fn)
            return
        fn = as_tool(fn)
        self.inject(name or tool_name(fn), fn)

    async def _open_toolkits(self) -> list[str]:
        names: list[str] = []
        for toolkit in self.toolkits:
            for fn in await toolkit.open():
                fn = as_tool(fn)
                self.inject(tool_name(fn), fn)
                names.append(tool_name(fn))
        return names

    async def _close_toolkits(self, names: list[str]) -> None:
        for name in names:
            self.remove_tool(name)
        for toolkit in reversed(self.toolkits):
            await toolkit.close()

    def remove_tool(self, name: str) -> Any:
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.runtime.remove_live(name)
        return self.tools.pop(name, None)

    def namespace_for(self, step: Step) -> dict[str, Any]:
        """Extra bindings for this agent's REPL; none by default."""
        return {}

    def spawn(
        self,
        step: Step,
        task: str,
        *,
        config: str = "default",
        name: str | None = None,
        **args: Any,
    ) -> Node:
        """Create an agent launched by the running node; every harness's launches land here."""
        self.config(config)
        g = step.graph
        child = g.add_agent(task, parent=step.node, config=config, name=name, **args)
        self.config_for(g, child)
        self.workspaces.branch(child, step.agent)
        self.wake()
        return child

    @property
    def agent_tools(self) -> Any:
        """The agent-tools server external harnesses reach; started on first use."""
        if self._agent_tools is None:
            from rlmflow.agent_tools.server import AgentToolsServer

            self._agent_tools = AgentToolsServer(self)
        return self._agent_tools

    # -- nodes ----------------------------------------------------------------

    def append(self, g: Graph, agent: Node, node: Node) -> Node:
        """Add ``node`` to ``agent``'s history; a finished one streams now."""
        g.append(agent, node)
        if node.state == DONE:
            self._land(node)
        return node

    async def run_node(self, g: Graph, agent: Node, node: Node) -> ReplRun | None:
        """Run a node of ``agent``'s history in its REPL: bind its inputs, then run its code.

        The REPL first catches up on the nodes before it. A node that is edited
        or removed while it runs is left as the edit made it, and ``None`` is
        returned. Saved calls a rerun didn't make again are undone.
        """
        version = node.version
        rerun = bool(node.calls)
        log = calls.CallLog(node)
        node.state, node.started_at = RUNNING, time.time()
        try:
            async with self._repl_lock(agent):
                await self._catch_up(g, agent, g.history(node)[:-1])
                self._live.pop(agent.id, None)
                run = await self._execute(Step(g, agent, node, calls=log))
        except asyncio.CancelledError:
            if node.state == RUNNING:
                node.state = READY
            raise
        except BaseException as exc:
            node.output, node.error = f"{type(exc).__name__}: {exc}", True
            node.state, node.finished_at = DONE, time.time()
            raise
        if node.id not in g or node.version != version:
            self.runtime.close_repl(agent)
            return None
        if rerun and run.status is not ReplStatus.DEAD:
            self.undo_calls(g, log.unclaimed())
        config = agent.root_data.config
        node.output = truncate_output(run.output, config.max_output_length)
        node.error = run.failed
        if run.status is ReplStatus.DEAD:
            node.output = f"{node.output}\n{DEAD_REPL_NOTE}" if node.output else DEAD_REPL_NOTE
            node.data["dead"] = True
        if run.submitted:
            node.result = run.answer
            node.data["terminal"] = True
        if node.runs_code:
            workspace = await self.workspaces.snapshot(agent)
            if workspace is not None:
                node.data["workspace"] = workspace
        node.state, node.finished_at = DONE, time.time()
        if run.status is not ReplStatus.DEAD:
            self._live[agent.id] = (node.id, node.version)
        self._land(node)
        return run

    def _repl_lock(self, agent: Node) -> asyncio.Lock:
        return self._repl_locks.setdefault(agent.id, asyncio.Lock())

    async def _execute(self, step: Step) -> ReplRun:
        """Bind the step's node's inputs and run its code in the agent's REPL."""
        g, agent, node = step.graph, step.agent, step.node
        repl = self.runtime.repl_for(agent, g)
        clashes = sorted(node.input.keys() & (RESERVED_TOOLS | self.tools.keys()))
        if clashes:
            raise ValueError(f"inputs collide with REPL tools: {', '.join(clashes)}")
        with running_step(step):
            for name, value in node.input.items():
                repl.inject(name, value)
            if not node.runs_code:
                return ReplRun()
            repl.seed(build_namespace(self, step))
            return await self.runtime.execute(agent, node.code, g)

    async def _catch_up(self, g: Graph, agent: Node, history: list[Node]) -> None:
        """Bring the agent's REPL up to the end of ``history``, rebuilding it if needed.

        Nodes that ran before run again with their saved calls, so nothing runs
        twice; nodes attached since the REPL last ran just bind their inputs.
        """
        live = self._live.get(agent.id)
        start = None
        if live is not None and self.runtime.get(agent) is not None:
            start = next(
                (i + 1 for i in range(len(history) - 1, -1, -1) if _marker(history[i]) == live),
                None,
            )
        rebuilding = start is None
        if rebuilding:
            self.runtime.close_repl(agent)
            self._live.pop(agent.id, None)
            start = 0
        repl = self.runtime.repl_for(agent, g)
        last = None
        for earlier in history[start:]:
            if earlier.state != DONE or earlier.data.get("dead"):
                continue
            replaying = earlier.runs_code and earlier.started_at is not None
            if replaying:
                repl.update_env({RLMFLOW_REPLAY: "1"})
            step = Step(g, agent, earlier, replaying=replaying, calls=calls.CallLog(earlier))
            try:
                await self._execute(step)
            finally:
                if replaying:
                    repl.update_env({RLMFLOW_REPLAY: "0"})
            last = earlier
        if rebuilding:
            # Files last: rerun nodes may have written to them.
            snapshot = next(
                (n.data["workspace"] for n in reversed(history) if n.data.get("workspace")),
                None,
            )
            if snapshot is not None:
                await self.workspaces.checkout(agent, snapshot)
        if last is not None:
            self._live[agent.id] = _marker(last)

    def undo_calls(self, g: Graph, entries: list[dict[str, Any]]) -> None:
        """Undo saved calls a rerun didn't make again: remove the agents, take back the messages."""
        for entry in entries:
            target = entry.get("agent") if entry["tool"] == calls.LAUNCH else entry.get("message")
            if entry["tool"] not in (calls.LAUNCH, calls.SEND) or target not in g:
                continue
            try:
                if entry["tool"] == calls.LAUNCH:
                    g.remove(g.agent(target))
                else:
                    g.delete(g.node(target))
            except (GraphError, KeyError):
                continue

    async def prepare_repl(self, g: Graph, agent: Node) -> None:
        """Bring the agent's live REPL up to the end of its history."""
        async with self._repl_lock(agent):
            await self._catch_up(g, agent, g.history(agent))

    async def replay(self, g: Graph, agent: Node | None = None) -> None:
        """Rebuild REPLs from the graph: one agent's, or every unfinished agent's."""
        for target in [agent] if agent is not None else g.agents():
            if agent is not None or g.status(target) != "done":
                self.config_for(g, target, record=False)
                self._live.pop(target.id, None)
                await self.prepare_repl(g, target)

    # -- turns ----------------------------------------------------------------

    async def step(self, g: Graph, agent: Node) -> Node:
        """Take one turn for ``agent`` outside a run; returns its last node."""
        self.config_for(g, agent)
        self.deliver_results(g, agent)
        await self._turn(g, agent)
        return g.tail(agent)

    async def _turn(self, g: Graph, agent: Node) -> None:
        """One turn: run the node an edit left ready, or call the agent's harness."""
        seen = len(g.history(agent))
        started = time.time()
        try:
            config = self.config_for(g, agent)
            ready = next((n for n in agent.root_data.history if n.state != DONE), None)
            if ready is not None:
                ready.data.setdefault("seen", len(g.history(ready)) - 1)
                await self.run_node(g, agent, ready)
            else:
                await config.load_harness()(self, Step.latest(g, agent))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            scope = "run" if g.parent(agent) is None else "child"
            failed = Node(
                "assistant",
                result=f"[{scope} failed: {text}]",
                output=text,
                error=True,
                seen=seen,
                terminal=True,
            )
            failed.started_at, failed.finished_at = started, time.time()
            if agent.id in g:
                self.append(g, agent, failed)
            if scope == "run":
                raise

    def budget_exhausted(self, g: Graph, agent: Node) -> bool:
        """Whether the agent's run has spent its ``max_budget`` tokens."""
        limit = agent.root_data.config.max_budget
        if limit is None:
            return False
        return g.usage(g.subtree(g.lineage(agent)[0])).total >= limit

    def budget_nearly_spent(self, g: Graph, agent: Node) -> bool:
        """Whether one more average model call would exhaust the run's token budget."""
        limit = agent.root_data.config.max_budget
        if limit is None:
            return False
        run = g.subtree(g.lineage(agent)[0])
        turns = sum(1 for a in run for node in a.root_data.history if node.data.get("usage"))
        spent = g.usage(run).total
        return spent >= limit or (turns > 0 and spent + spent / turns >= limit)

    # -- results ----------------------------------------------------------------

    def deliver_results(self, g: Graph, agent: Node) -> Node | None:
        """Attach the results of ``agent``'s finished children that nothing read yet.

        One user node binds each result as a variable named after its child.
        Only between turns, never to an agent that finished, and not while an
        edit has left a node to rerun, whose ``await`` takes the results itself.
        """
        if agent.id in self.tasks or g.status(agent) == "done":
            return None
        if any(node.state != DONE for node in agent.root_data.history):
            return None
        finished = unseen_results(g, agent)
        if not finished:
            return None
        limit = min(RESULT_MAX, agent.root_data.config.max_output_length)
        inputs: dict[str, Any] = {}
        items = []
        for child, final in finished:
            name = result_name(child.root_data.name, inputs.keys() | self.tools.keys())
            inputs[name] = _json_safe(final.result)
            items.append((g.path(child), child.root_data.config_name or "root", final.result))
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

    # -- running --------------------------------------------------------------

    def _land(self, node: Node) -> None:
        if self._landed is not None:
            self._landed.append(node)
        self.wake()

    def wake(self) -> None:
        """Wake whatever waits for the graph to change; safe from any thread."""
        loop = self._loop
        if loop is not None:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not loop:
                loop.call_soon_threadsafe(self.wake)
                return
        news, self._news = self._news, asyncio.Event()
        news.set()

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        """Wait until ``predicate()`` holds; it is checked whenever the graph changes."""
        while True:
            news = self._news
            if predicate():
                return
            await news.wait()

    async def edit(
        self, g: Graph, op: str, target: Node, *args: Any, **kwargs: Any
    ) -> Invalidation:
        """Apply ``g.<op>(target, ...)`` to a graph this Flow may be running.

        The agents the edit touches stop first, so no turn writes over it.
        """
        if op not in ("edit", "delete", "remove", "replace"):
            raise ValueError(f"unknown edit {op!r}")
        preview = getattr(g, op)(target, *args, dry_run=True, **kwargs)
        await self._cancel([*preview.rewound, *preview.removed])
        invalidation = getattr(g, op)(target, *args, **kwargs)
        self.wake()
        return invalidation

    async def _cancel(self, agent_ids: Iterable[str]) -> None:
        tasks = [self.tasks.pop(i) for i in dict.fromkeys(agent_ids) if i in self.tasks]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_streaming(
        self,
        g: Graph,
        *,
        until: StepUntil = "done",
        root_config: AgentConfig | str | None = None,
    ) -> AsyncIterator[Step]:
        """Run every agent that has something to do, and yield a ``Step`` for each node as it finishes.

        Roots run with ``root_config`` (an ``AgentConfig``, a config name, or
        ``"default"``); every other agent with the config it names.
        """
        if self._graph is not None:
            raise RuntimeError("this Flow is already driving a run")
        self.root_config = self.resolve_config(root_config)
        for agent in g.agents():
            name = agent.root_data.config_name
            if g.status(agent) != "done":
                self.config_for(g, agent)
            elif name is None or name in self.configs:
                self.config_for(g, agent, record=False)
        stop_at = _boundary(until)
        self._graph, self._loop, self._landed = g, asyncio.get_running_loop(), []
        self._news = asyncio.Event()
        toolkit_tools: list[str] = []
        try:
            toolkit_tools = await self._open_toolkits()
            while True:
                news = self._news
                self._reap(g)
                self._schedule(g)
                if self._landed:
                    stop = False
                    while self._landed:
                        node = self._landed.pop(0)
                        if node.id in g:
                            step = Step(g, g.owner(node), node)
                            yield step
                            stop = stop or (stop_at is not None and stop_at(step))
                    if stop:
                        return
                    # The caller may have written to the graph between steps.
                    continue
                if not self.tasks:
                    return
                await news.wait()
        finally:
            await self._cancel(list(self.tasks))
            for agent in g.agents():
                self.deliver_results(g, agent)
            await self._close_toolkits(toolkit_tools)
            self._graph = self._loop = self._landed = None

    def _reap(self, g: Graph) -> None:
        """Forget finished turns; a top-level agent's failure ends the run."""
        for agent_id, task in list(self.tasks.items()):
            if not task.done():
                continue
            del self.tasks[agent_id]
            error = None if task.cancelled() else task.exception()
            if error is not None:
                raise error

    def _schedule(self, g: Graph) -> None:
        """Start a turn for every agent that has something to do and no turn in flight."""
        for agent in g.agents():
            if agent.id in self.tasks:
                continue
            self.deliver_results(g, agent)
            if g.status(agent) != "running":
                continue
            self.config_for(g, agent)
            if self.budget_nearly_spent(g, agent) and not _final_sent(g, agent):
                final = Node("user", BUDGET_FINAL_TEXT, producer="rlmflow", final=True)
                self.append(g, agent, final)
            task = asyncio.ensure_future(self._turn(g, agent))
            task.add_done_callback(lambda _task: self.wake())
            self.tasks[agent.id] = task

    async def arun(
        self, target: Graph | str, *, root_config: AgentConfig | str | None = None, **args: Any
    ) -> Any:
        """Run to the end. A graph returns itself; a task string returns its result."""
        if isinstance(target, str):
            g = Graph()
            agent = g.add_agent(target, **args)
        elif args:
            raise TypeError("agent arguments can only be passed with a task string")
        else:
            g, agent = target, None
        async for _step in self.run_streaming(g, root_config=root_config):
            pass
        return g if agent is None else g.result(agent)

    def run(
        self, target: Graph | str, *, root_config: AgentConfig | str | None = None, **args: Any
    ) -> Any:
        return asyncio.run(self.arun(target, root_config=root_config, **args))

    async def aclose(self) -> None:
        await self._cancel(list(self.tasks))
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
            self._live.clear()


def _marker(node: Node) -> tuple[str, int]:
    return node.id, node.version


def _final_sent(g: Graph, agent: Node) -> bool:
    return any(node.data.get("final") for node in g.history(agent))


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return str(value)


def result_name(agent_name: str, taken: Iterable[str] = ()) -> str:
    """The variable a result node binds a child's result to: its name as an identifier."""
    name = re.sub(r"\W", "_", agent_name)
    if not name or name[0].isdigit() or keyword.iskeyword(name) or name in RESERVED_TOOLS:
        name = f"_{name}"
    taken = set(taken)
    candidate, index = name, 1
    while candidate in taken:
        candidate, index = f"{name}_{index}", index + 1
    return candidate


def result_marker(path: str) -> str:
    return f"[result {path}]"


def render_results(items: list[tuple[str, str, Any]], limit: int) -> str:
    """``(path, config name, result)`` per finished agent, as one message."""
    parts = ["Agents you started have finished. Each block is that agent's result."]
    for path, config_name, result in items:
        body = _result_body(result)
        if len(body) > limit:
            body = body[: max(limit - 3, 0)] + "..."
        parts.append(f"{result_marker(path)}\nconfig: {config_name}\n{body}")
    return "\n\n".join(parts)


def _result_body(value: Any) -> str:
    """A mapping is its fields; anything else is JSON, labeled ``result``."""
    if isinstance(value, Mapping):
        if not value:
            return "result: {}"
        lines = []
        for key, item in value.items():
            rendered = _jsonish(item)
            lines.append(
                f"{key}:\n{textwrap.indent(rendered, '  ')}"
                if "\n" in rendered
                else f"{key}: {rendered}"
            )
        return "\n".join(lines)
    return f"result: {_jsonish(value)}"


def _jsonish(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False, indent=2)
    return json.dumps(value, default=str, ensure_ascii=False)


__all__ = [
    "BUDGET_FINAL_TEXT",
    "Flow",
    "StepUntil",
    "render_results",
    "result_name",
]
