"""Run agents: step each ready agent with its harness and stream every node as a ``Step``."""

from __future__ import annotations

import asyncio
import json
import textwrap
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

from rlmflow.engine.scheduler import (
    Pool,
    StepResult,
    StepUntil,
    TaskQueue,
    ThreadPool,
    resolve_boundary,
)
from rlmflow.engine.workspaces import LocalWorkspace
from rlmflow.graph import calls
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.graph import CELL_RESULTS, Graph, GraphError
from rlmflow.graph.nodes import (
    CODE,
    DONE,
    LLM,
    MESSAGE,
    PENDING,
    Agent,
    LLMUsage,
    Node,
    Step,
    current_code,
    replaying,
    running_cell,
    running_step,
)
from rlmflow.graph.queries import cells, unseen_results
from rlmflow.llm import PooledLLMClient, client_for
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

BUDGET_FINAL_TEXT = """\
This run has nearly used its token budget, so this is your last turn. Give the best \
supported final result now and don't start new work."""

DEAD_REPL_NOTE = """\
[this REPL died while running the cell. It will be rebuilt from the recorded cells \
before the next one, so variables set by earlier cells come back; anything this cell \
did is gone.]"""

#: Longest result text one agent contributes to a results message.
RESULT_MAX = 20_000


class Flow:
    """Model clients, the REPL runtime, tools and workspaces. Steps ready agents.

    The Flow decides how every agent runs. ``configs`` are the named kinds of
    agent that ``launch_agent(config=...)`` picks from (``"default"`` is
    ``AgentConfig()`` unless given); a root runs with the run's ``root_config``.
    The graph only names each agent's config and records the JSON of the one
    that ran it. A config's ``model`` is any model name, or a name in
    ``llm_clients`` for clients a name can't describe.
    ``workers`` sizes the thread pool blocking model calls run on, or is a
    ``Pool``.
    """

    def __init__(
        self,
        *,
        llm_clients: dict[str, Any] | None = None,
        configs: Iterable[AgentConfig] = (),
        runtime: Runtime | None = None,
        tools: Any = None,
        workspaces: Any = None,
        workers: int | Pool | None = None,
    ) -> None:
        self.llms: dict[str, Any] = dict(llm_clients or {})
        self.pool = workers if isinstance(workers, Pool) else ThreadPool(workers)
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
        self.queue: TaskQueue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        #: For each agent whose live REPL is up to date, the last cell it ran.
        self._live: dict[str, tuple[str, str] | None] = {}
        self._repl_locks: dict[str, asyncio.Lock] = {}
        self._agent_tools: Any = None

    # -- models -------------------------------------------------------------

    def llm(self, model: str | None) -> PooledLLMClient:
        """The client for ``model`` on this Flow's pool; a client's ``request_timeout`` caps each call."""
        if model is None:
            raise ValueError(
                "no model: set AgentConfig(model=...) on the config this agent runs with"
            )
        if model not in self.llms:
            self.llms[model] = client_for(model)
        client = self.llms[model]
        return PooledLLMClient(client, self.pool, timeout=getattr(client, "request_timeout", None))

    async def complete(
        self, messages: list[dict[str, Any]], model: str | None, **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        return await self.llm(model).completion(messages, **kwargs)

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

    def config_for(self, g: Graph, agent: Agent, *, record: bool = True) -> AgentConfig:
        """Bind and return the config ``agent`` runs with: ``root_config`` for a root, else its named one."""
        name = agent.config_name
        config = self.root_config if name is None else self.config(name)
        if record:
            agent.bind_config(config)
        else:
            agent._config = config
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
        """Extra bindings for this agent's REPL cells; none by default."""
        return {}

    def spawn(
        self,
        step: Step,
        task: str,
        *,
        config: str = "default",
        name: str | None = None,
        **args: Any,
    ) -> Agent:
        """Create an agent launched by the running cell; every harness's launches land here."""
        self.config(config)
        g = step.graph
        child = g.add_agent(
            task, parent=current_code() or step.agent, config=config, name=name, **args
        )
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

    # -- the REPL -----------------------------------------------------------

    async def _run_cell(self, step: Step, cell: Node, code: str) -> ReplRun:
        g, agent = step.graph, step.agent
        calls.begin(cell)
        with running_step(step), running_cell(cell):
            repl = self.runtime.repl_for(agent, g)
            repl.seed(build_namespace(self, step), {"task": agent.task, **agent.resolved_args()})
            return await self.runtime.execute(agent, code, g)

    def _repl_current(self, g: Graph, agent: Agent) -> bool:
        """Whether the agent's live REPL has run exactly its recorded cells."""
        return (
            agent.id in self._live
            and self._live[agent.id] == _last_cell(g, agent)
            and self.runtime.get(agent) is not None
        )

    async def prepare_repl(self, g: Graph, agent: Agent) -> None:
        """Make the agent's live REPL match its recorded cells, rebuilding it if needed.

        A rebuild reruns every recorded cell in order. Launches, awaits and other
        host tool calls return their saved results, so nothing runs twice.
        """
        if self._repl_current(g, agent):
            return
        lock = self._repl_locks.setdefault(agent.id, asyncio.Lock())
        async with lock:
            if self._repl_current(g, agent):
                return
            self.runtime.close_repl(agent)
            self.config_for(g, agent, record=g.status(agent) != "done")
            repl = self.runtime.repl_for(agent, g)
            history = g.history(agent)
            for cell in recorded_cells(g, agent):
                index = history.index(cell)
                step = Step(g, agent, history[index - 1] if index else agent)
                repl.update_env({RLMFLOW_REPLAY: "1"})
                try:
                    with replaying():
                        await self._run_cell(step, cell, cell.data["code"])
                finally:
                    calls.forget(cell)
                    repl.update_env({RLMFLOW_REPLAY: "0"})
            # Files last: rerun cells may have written to them.
            snapshot = next(
                (n.data["workspace"] for n in reversed(history) if n.data.get("workspace")),
                None,
            )
            if snapshot is not None:
                await self.workspaces.checkout(agent, snapshot)
            self._live[agent.id] = _last_cell(g, agent)

    async def execute(
        self, step: Step, code: str, cell: Node | None = None
    ) -> tuple[Node, ReplRun]:
        """Run one cell in ``step.agent``'s REPL and return its ``code`` node.

        The cell lands, marked pending, before its code runs, so whatever it does
        while running hangs off a node in the graph; it is finished in place. A
        cell that is stopped stays pending and runs again. ``cell`` re-runs a cell
        already in the graph; its saved calls return what they returned before
        (or what ``Graph.replace`` put there).
        """
        g, agent = step.graph, step.agent
        await self.prepare_repl(g, agent)
        rerun = cell is not None
        if cell is None:
            cell = Node(CODE, code=code, **{PENDING: True})
            cell.started_at = time.time()
            g.append(agent, cell)
        else:
            for key in CELL_RESULTS:
                cell.data.pop(key, None)
            cell.started_at = time.time()
        self._live.pop(agent.id, None)
        try:
            run = await self._run_cell(step, cell, code)
        except asyncio.CancelledError:
            calls.forget(cell)
            raise
        except BaseException as exc:
            calls.forget(cell)
            cell.data.pop(PENDING, None)
            cell.finished_at = time.time()
            g.update(cell, output=f"{type(exc).__name__}: {exc}", error=True)
            raise
        if rerun and run.status is not ReplStatus.DEAD:
            self.undo_calls(g, calls.finish(cell))
        else:
            calls.forget(cell)
        output = truncate_output(run.output, agent.config.max_output_length)
        finished: dict[str, Any] = {"output": output, "error": run.failed}
        if run.status is ReplStatus.DEAD:
            finished["output"] = f"{output}\n{DEAD_REPL_NOTE}" if output else DEAD_REPL_NOTE
            finished["dead"] = True
        workspace = await self.workspaces.snapshot(agent)
        if workspace is not None:
            finished["workspace"] = workspace
        cell.data.pop(PENDING, None)
        cell.finished_at = time.time()
        g.update(cell, **finished)
        if run.status is not ReplStatus.DEAD:
            self._live[agent.id] = _last_cell(g, agent)
        return cell, run

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
            except GraphError:
                continue

    async def rerun(self, g: Graph, agent: Agent, cell: Node) -> Node:
        """Run a pending cell again in place; a submit adds its ``done``."""
        history = g.history(agent)
        index = history.index(cell)
        step = Step(g, agent, history[index - 1] if index else agent)
        _, run = await self.execute(step, cell.data["code"], cell)
        return Node(DONE, result=run.answer) if run.submitted else cell

    async def replay(self, g: Graph, agent: Agent | None = None) -> None:
        """Rebuild REPLs from the graph: one agent's, or every unfinished agent's."""
        for target in [agent] if agent is not None else g.agents():
            if agent is not None or g.status(target) != "done":
                self._live.pop(target.id, None)
                await self.prepare_repl(g, target)

    # -- steps ----------------------------------------------------------------

    def wake(self) -> None:
        if self.queue is not None:
            self.queue.wake()

    async def step(self, g: Graph, agent: Agent) -> Node | None:
        """Take one step for ``agent`` outside a run."""
        g.deliver(agent)
        step = Step.latest(g, agent)
        with running_step(step):
            return (await self.run_step(step)).created

    async def run_step(self, step: Step) -> StepResult:
        """One step: call the agent's harness (or re-run an invalidated cell) and land its node."""
        g, agent = step.graph, step.agent
        started = time.time()
        error: BaseException | None = None
        landed: Node | None
        try:
            config = self.config_for(g, agent)
            if isinstance(step.node, Node) and step.node.pending:
                landed = await self.rerun(g, agent, step.node)
            else:
                landed = await config.load_harness()(self, step)
                if not isinstance(landed, Node):
                    raise TypeError(
                        f"harness {config.harness} returned {type(landed).__name__}, not a Node"
                    )
            # Cells land as they start; anything else the step returns lands now.
            if landed.id not in g:
                landed.started_at, landed.finished_at = started, time.time()
                g.append(agent, landed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed step finishes the agent
            error = exc
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            scope = "run" if g.parent(agent) is None else "child"
            landed = Node(DONE, result=f"[{scope} failed: {text}]", error=text)
            landed.started_at, landed.finished_at = started, time.time()
            g.append(agent, landed)
        return StepResult(step=step, created=landed, error=error)

    def budget_exhausted(self, g: Graph, agent: Agent) -> bool:
        """Whether the agent's run has spent its ``max_budget`` tokens."""
        limit = agent.config.max_budget
        if limit is None:
            return False
        return g.usage(g.subtree(g.lineage(agent)[0])).total >= limit

    def budget_nearly_spent(self, g: Graph, agent: Agent) -> bool:
        """Whether one more average model call would exhaust the run's token budget."""
        limit = agent.config.max_budget
        if limit is None:
            return False
        run = g.subtree(g.lineage(agent)[0])
        turns = sum(1 for a in run for node in a.nodes if node.type == LLM)
        spent = g.usage(run).total
        return spent >= limit or (turns > 0 and spent + spent / turns >= limit)

    # -- results ----------------------------------------------------------------

    def deliver_results(self, g: Graph, agent: Agent) -> Node | None:
        """Message ``agent`` the results of its finished children that no cell awaited.

        Only between turns: never while it steps, mid-action, or after it finished.
        """
        history = g.history(agent)
        last = history[-1] if history else None
        if agent.id in g.busy or last is None or last.type in (LLM, DONE) or last.pending:
            return None
        finished = unseen_results(g, agent)
        if not finished:
            return None
        limit = min(RESULT_MAX, agent.config.max_output_length)
        text = render_results(
            [(g.path(c), c.config_name or "root", g.result(c)) for c, _ in finished], limit
        )
        message = g.send(agent, text, sender="rlmflow")
        for _child, done in finished:
            g.link_return(done, message)
        return message

    # -- running --------------------------------------------------------------

    async def run_streaming(
        self,
        g: Graph,
        *,
        until: StepUntil = "done",
        root_config: AgentConfig | str | None = None,
    ) -> AsyncIterator[Step]:
        """Drive every ready agent and yield a ``Step`` for each node as it lands.

        Roots run with ``root_config`` (an ``AgentConfig``, a config name, or
        ``"default"``); every other agent with the config it names.
        """
        if self.queue is not None:
            raise RuntimeError("this Flow is already driving a run")
        self.root_config = self.resolve_config(root_config)
        for agent in g.agents():
            if g.status(agent) != "done":
                self.config_for(g, agent)
            elif agent.config_name is None or agent.config_name in self.configs:
                self.config_for(g, agent, record=False)
        boundary = resolve_boundary(until)
        queue = TaskQueue()
        self.queue = queue
        self._loop = asyncio.get_running_loop()
        landed: list[Node] = []

        def written(kind: str, item: Any) -> None:
            # A cell streams once, when it finishes.
            if kind == "node" and not item.pending:
                landed.append(item)
            queue.wake()

        unsubscribe = g.subscribe(written)
        toolkit_tools: list[str] = []
        try:
            toolkit_tools = await self._open_toolkits()
            for agent in g.agents():
                self.deliver_results(g, agent)
            self._schedule(g)
            while queue:
                result = await queue.next()
                if result is not None:
                    g.busy.discard(result.agent.id)
                    self.deliver_results(g, result.agent)
                    parent = g.parent(result.agent) if result.agent.id in g else None
                    if parent is not None:
                        self.deliver_results(g, parent)
                stop = False
                while landed:
                    node = landed.pop(0)
                    if node.id in g:
                        step = Step(g, g.owner(node), node)
                        yield step
                        stop = stop or (boundary is not None and boundary(step))
                if stop:
                    return
                await queue.notify()
                if (
                    result is not None
                    and result.error is not None
                    and g.parent(result.agent) is None
                ):
                    raise result.error
                self._schedule(g)
        finally:
            unsubscribe()
            interrupted = list(queue.running)
            await queue.cancel()
            g.busy.difference_update(interrupted)
            for agent_id in interrupted:
                self._live.pop(agent_id, None)
            for agent in g.agents():
                self.deliver_results(g, agent)
            await self._close_toolkits(toolkit_tools)
            if self.queue is queue:
                self.queue = None
                self._loop = None

    def _schedule(self, g: Graph) -> None:
        """Start one step for every ready agent that has none in flight."""
        queue = self.queue
        for agent in g.agents():
            if agent.id in g.busy or agent in queue or g.status(agent) != "running":
                continue
            self.config_for(g, agent)
            history = g.history(agent)
            last = history[-1] if history else None
            if (last is None or last.type != LLM) and self.budget_nearly_spent(g, agent):
                if not _final_sent(g, agent):
                    g.send(agent, BUDGET_FINAL_TEXT, sender="rlmflow", final=True)
            g.deliver(agent)
            g.busy.add(agent.id)
            queue.submit(Step.latest(g, agent), self.run_step)

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
        if self.queue is not None:
            await self.queue.cancel()
            self.queue = None
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
            self.pool.close()
            self.runtime.close()
            self._live.clear()


def recorded_cells(g: Graph, agent: Agent) -> list[Node]:
    """The cells a REPL rebuild re-runs: every finished one, except ones that killed the REPL."""
    return [node for node in cells(g, agent) if not node.data.get("dead") and not node.pending]


def _last_cell(g: Graph, agent: Agent) -> tuple[str, str] | None:
    """Which cell an up-to-date REPL ran last, and its code (an edit may change it)."""
    ran = recorded_cells(g, agent)
    return (ran[-1].id, ran[-1].data["code"]) if ran else None


def _final_sent(g: Graph, agent: Agent) -> bool:
    return any(
        node.type == MESSAGE and node.data.get("final")
        for node in [*g.history(agent), *agent.inbox]
    )


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
    "recorded_cells",
    "render_results",
]
