"""Run agents: step each ready agent with its harness and stream every node as a ``Step``."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from rlmflow.engine import handoff
from rlmflow.engine.scheduler import (
    Pool,
    StepResult,
    StepUntil,
    TaskQueue,
    ThreadPool,
    resolve_boundary,
)
from rlmflow.engine.workspaces import LocalWorkspace
from rlmflow.graph.config import AgentConfig
from rlmflow.graph.graph import FROM, Graph, StaleFrontier
from rlmflow.graph.nodes import (
    AGENT_REF_KEY,
    AGENT_START,
    CODE,
    DONE,
    LAUNCH_TOOLS,
    LLM_OUTPUT,
    MESSAGE,
    WAITING,
    LLMUsage,
    Node,
    Step,
    replaying,
    running_cell,
    running_step,
)
from rlmflow.harnesses.repl.runtime import LocalRuntime, Runtime
from rlmflow.harnesses.repl.runtime.repl import ReplRun, ReplStatus
from rlmflow.harnesses.repl.runtime.runtime import RLMFLOW_REPLAY
from rlmflow.harnesses.repl.tools import RESERVED_TOOLS, as_tool_items, build_namespace, tool_name
from rlmflow.llm import PooledLLMClient
from rlmflow.utils import truncate_output

BUDGET_FINAL_TEXT = """\
This run has nearly used its token budget, so this is your last turn. Give the best \
supported final result now and don't start new work."""

DEAD_REPL_NOTE = """\
[this REPL died while running the cell. It will be rebuilt from the recorded cells \
before the next one, so variables set by earlier cells come back; anything this cell \
did is gone.]"""


class Flow:
    """Model clients, the REPL runtime, tools and workspaces. Steps ready agents.

    Each agent runs with the harness named in its config. ``llm`` is the
    ``"default"`` model; ``llm_clients`` adds more by name. ``workers`` sizes the
    thread pool blocking model calls run on, or is a ``Pool``.
    """

    def __init__(
        self,
        llm: Any = None,
        *,
        llm_clients: dict[str, Any] | None = None,
        runtime: Runtime | None = None,
        tools: Any = None,
        workspaces: Any = None,
        workers: int | Pool | None = None,
    ) -> None:
        self.llms: dict[str, Any] = {**({"default": llm} if llm is not None else {}), **(llm_clients or {})}
        self.pool = workers if isinstance(workers, Pool) else ThreadPool(workers)
        self.runtime = runtime or LocalRuntime()
        self.workspaces = workspaces or LocalWorkspace(self.runtime.working_directory)
        if workspaces is not None:
            self.runtime.directory_for = workspaces.path
        self.tools: dict[str, Any] = {}
        for item in as_tool_items(tools):
            self.add_tool(item)
        self.queue: TaskQueue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        #: Each stepping agent's latest node; its own writes during a step move it.
        self._anchors: dict[str, Node] = {}
        #: Waiting agent id -> the agents its in-cell waits are on.
        self._waits: dict[str, list[str]] = {}
        #: Agents whose live REPL matches their recorded cells.
        self._live: set[str] = set()
        self._repl_locks: dict[str, asyncio.Lock] = {}
        self._agent_tools: Any = None

    # -- models -------------------------------------------------------------

    def llm(self, model: str = "default") -> PooledLLMClient:
        """The named client on this Flow's pool; a client's ``request_timeout`` caps each call."""
        if model not in self.llms:
            available = ", ".join(sorted(self.llms)) or "none"
            raise ValueError(f"unknown model {model!r}; this Flow has: {available}")
        client = self.llms[model]
        return PooledLLMClient(client, self.pool, timeout=getattr(client, "request_timeout", None))

    async def complete(
        self, messages: list[dict[str, str]], model: str = "default", **kwargs: Any
    ) -> tuple[str, LLMUsage]:
        return await self.llm(model).completion(messages, **kwargs)

    async def llm_query(self, prompt: str, *, model: str = "default", schema: Any = None) -> Any:
        """One context-free model call from host code."""
        from rlmflow.harnesses.repl.tools import query

        return await query(self, prompt, model=model, schema=schema)

    # -- tools --------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        """Bind ``value`` as ``name`` in every agent's REPL namespace."""
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.tools[name] = value
        self.runtime.inject_live(name, value)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        self.inject(name or tool_name(fn), fn)

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
        config: AgentConfig,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **args: Any,
    ) -> Node:
        """Create an agent launched by ``step.node``; every harness's launches land here."""
        child = step.graph.spawn(task, by=step.node, config=config, name=name, inputs=inputs, **args)
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

    def _repl_args(self, g: Graph, agent: Node) -> dict[str, Any]:
        return {"task": agent.task, **agent.resolved_args(), **g.inputs(agent)}

    async def _run_cell(self, step: Step, cell: Node, code: str) -> ReplRun:
        g, agent = step.graph, step.agent
        with running_step(step), running_cell(cell):
            repl = self.runtime.repl_for(agent, g)
            repl.seed(build_namespace(self, step), self._repl_args(g, agent))
            return await self.runtime.execute(agent, code, g)

    async def prepare_repl(self, g: Graph, agent: Node) -> None:
        """Make the agent's live REPL match its recorded cells, rebuilding it if needed.

        A rebuild reruns every recorded cell in order. Launches and ``llm_query``
        calls return their recorded results, so nothing is launched twice.
        """
        if agent.id in self._live and self.runtime.get(agent) is not None:
            return
        lock = self._repl_locks.setdefault(agent.id, asyncio.Lock())
        async with lock:
            if agent.id in self._live and self.runtime.get(agent) is not None:
                return
            self.runtime.close_repl(agent)
            repl = self.runtime.repl_for(agent, g)
            for cell in recorded_cells(g, agent):
                step = Step(g, agent, g.prev(cell) or agent)
                repl.update_env({RLMFLOW_REPLAY: "1"})
                try:
                    with replaying():
                        await self._run_cell(step, cell, cell.data["code"])
                finally:
                    repl.update_env({RLMFLOW_REPLAY: "0"})
            # Files last: rerun cells may have written to them.
            snapshot = next(
                (n.data["workspace"] for n in reversed(g.history(agent)) if n.data.get("workspace")),
                None,
            )
            if snapshot is not None:
                await self.workspaces.checkout(agent, snapshot)
            self._live.add(agent.id)

    async def execute(self, step: Step, code: str) -> tuple[Node, ReplRun]:
        """Run one cell in ``step.agent``'s REPL and return its unlanded ``code`` node."""
        g, agent = step.graph, step.agent
        await self.prepare_repl(g, agent)
        cell = Node(CODE, code=code)
        # A cell that ran from this model turn before but never landed (the run
        # halted mid-cell) already launched these agents; the rerun gets them back.
        if step.node.type == LLM_OUTPUT:
            for call_id, child in enumerate(launched_from(g, step.node)):
                cell.record_call(LAUNCH_TOOLS[0], call_id, {AGENT_REF_KEY: child.id})
        run = await self._run_cell(step, cell, code)
        output = truncate_output(run.output, agent.config.max_output_length)
        if run.status is ReplStatus.DEAD:
            self._live.discard(agent.id)
            output = f"{output}\n{DEAD_REPL_NOTE}" if output else DEAD_REPL_NOTE
            cell.data["dead"] = True
        cell.data.update(output=output, error=run.failed)
        if run.submitted:
            cell.data.update(submitted=True, result=run.answer)
        workspace = await self.workspaces.snapshot(agent)
        if workspace is not None:
            cell.data["workspace"] = workspace
        return cell, run

    async def replay(self, g: Graph, agent: Node | None = None) -> None:
        """Rebuild REPLs from the graph: one agent's, or every unfinished agent's."""
        for target in [agent] if agent is not None else g.agents():
            if agent is not None or not g.terminal(target):
                self._live.discard(target.id)
                await self.prepare_repl(g, target)

    # -- steps ----------------------------------------------------------------

    def anchor(self, g: Graph, agent: Node) -> Node:
        """The agent's latest node: its step's anchor while stepping, else its frontier."""
        return self._anchors.get(agent.id) or g.frontier(agent)

    def land(self, g: Graph, agent: Node, node: Node) -> Node:
        """Append a node the agent wrote during its own step, moving its anchor."""
        with g._lock:
            g.append(agent, node, after=self._anchors.get(agent.id))
            if agent.id in self._anchors:
                self._anchors[agent.id] = node
        self.wake()
        return node

    def wake(self) -> None:
        if self.queue is not None:
            self.queue.wake()

    async def step(self, g: Graph, agent: Node) -> Node | None:
        """Take one step for ``agent`` outside a run."""
        g.deliver(agent)
        step = Step(g, agent, g.frontier(agent))
        with running_step(step):
            return (await self.run_step(step)).created

    async def run_step(self, step: Step) -> StepResult:
        """One step: call the agent's harness and land the node it returns."""
        g, agent = step.graph, step.agent
        self._anchors[agent.id] = step.node
        started = time.time()
        error: BaseException | None = None
        landed: Node | None
        try:
            harness = agent.config.load_harness()
            landed = await harness(self, step)
            if not isinstance(landed, Node):
                raise TypeError(f"harness {agent.config.harness} returned {type(landed).__name__}, not a Node")
            _stamp(landed, started)
            g.append(agent, landed, after=self._anchors[agent.id])
        except asyncio.CancelledError:
            raise
        except StaleFrontier:
            landed = None  # a host write won; the agent is rescheduled from there
        except Exception as exc:  # noqa: BLE001 - a failed step finishes the agent
            error = exc
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            scope = "run" if g.launcher(agent) is None else "child"
            landed = Node(DONE, result=f"[{scope} failed: {text}]", error=text)
            _stamp(landed, started)
            g.append(agent, landed)
        finally:
            self._anchors.pop(agent.id, None)
        return StepResult(step=step, created=landed, error=error)

    def budget_exhausted(self, g: Graph, agent: Node) -> bool:
        """Whether the agent's run has spent its ``max_budget`` tokens."""
        limit = agent.config.max_budget
        if limit is None:
            return False
        run = g.tree(g.lineage(agent)[0])
        return g.usage(run).total >= limit

    def budget_nearly_spent(self, g: Graph, agent: Node) -> bool:
        """Whether one more average model call would exhaust the run's token budget."""
        limit = agent.config.max_budget
        if limit is None:
            return False
        run = g.tree(g.lineage(agent)[0])
        turns = sum(len(g.nodes(LLM_OUTPUT, agent=a.id)) for a in run)
        spent = g.usage(run).total
        return spent >= limit or (turns > 0 and spent + spent / turns >= limit)

    # -- running --------------------------------------------------------------

    async def run_streaming(self, g: Graph, *, until: StepUntil = "done") -> AsyncIterator[Step]:
        """Drive every ready agent and yield a ``Step`` for each node as it lands."""
        if self.queue is not None:
            raise RuntimeError("this Flow is already driving a run")
        boundary = resolve_boundary(until)
        cursor = len(g) - 1
        queue = TaskQueue()
        self.queue = queue
        self._loop = asyncio.get_running_loop()
        try:
            for agent in g.agents():
                handoff.react(g, agent)
            await self._schedule(g)
            while queue:
                result = await queue.next()
                if result is not None:
                    self._react(g, result)
                items = g.items(after=cursor)
                cursor += len(items)
                stop = False
                for item in items:
                    if isinstance(item, Node):
                        step = Step(g, g.node(item.agent_id), item)
                        yield step
                        stop = stop or (boundary is not None and boundary(step))
                if stop:
                    return
                await queue.notify()
                if result is not None and result.error is not None and g.launcher(result.agent) is None:
                    raise result.error
                await self._schedule(g)
        finally:
            interrupted = list(queue.running)
            await queue.cancel()
            self._live.difference_update(interrupted)
            if self.queue is queue:
                self.queue = None
                self._loop = None

    def _react(self, g: Graph, result: StepResult) -> None:
        """After a step lands: deliver results to it and from it."""
        agent = result.agent
        handoff.react(g, agent)
        if result.created is not None and result.created.type in (DONE, WAITING):
            for recipient in handoff.recipients(g, agent):
                if recipient not in self.queue:
                    handoff.react(g, recipient)

    async def _schedule(self, g: Graph) -> None:
        """Start one step for every ready agent that has none in flight."""
        queue = self.queue
        for agent in g.ready():
            if agent in queue:
                continue
            frontier = g.frontier(agent)
            if frontier.type != LLM_OUTPUT and self.budget_nearly_spent(g, agent):
                if not _final_sent(g, agent):
                    g.send(agent, BUDGET_FINAL_TEXT, sender="rlmflow", final=True)
            g.deliver(agent)
            if g.terminal(agent):
                continue
            queue.submit(Step(g, agent, g.frontier(agent)), self.run_step)

    async def arun(self, target: Graph | str, *, config: AgentConfig | str | None = None, **args: Any) -> Any:
        """Run to the end. A graph returns itself; a task string returns its result."""
        if isinstance(target, str):
            g = Graph()
            agent = g.create_agent(target, config=config, **args)
        elif args or config is not None:
            raise TypeError("config and agent arguments can only be passed with a task string")
        else:
            g, agent = target, None
        async for _step in self.run_streaming(g):
            pass
        return g if agent is None else g.result(agent)

    def run(self, target: Graph | str, *, config: AgentConfig | str | None = None, **args: Any) -> Any:
        return asyncio.run(self.arun(target, config=config, **args))

    async def aclose(self) -> None:
        if self.queue is not None:
            await self.queue.cancel()
            self.queue = None
        if self._agent_tools is not None:
            self._agent_tools.close()
            self._agent_tools = None
        owners = {id(client): client for client in self.llms.values()}
        closing = [close() for client in owners.values() if (close := getattr(client, "aclose", None))]
        try:
            await asyncio.gather(*closing)
        finally:
            self.pool.close()
            self.runtime.close()
            self._live.clear()


def recorded_cells(g: Graph, agent: Node) -> list[Node]:
    """Every cell in the agent's history that ran code, in order, except ones that killed the REPL."""
    return [
        node
        for node in g.history(agent)
        if node.type in (CODE, DONE) and node.data.get("code") and not node.data.get("dead")
    ]


def launched_from(g: Graph, node: Node) -> list[Node]:
    """Agents created by ``node``, in creation order."""
    found = (g.node(edge.source) for edge in g.edges(kind=FROM, target=node.id))
    return sorted((agent for agent in found if agent.type == AGENT_START), key=lambda a: a.seq)


def _final_sent(g: Graph, agent: Node) -> bool:
    return any(
        node.type == MESSAGE and node.data.get("final")
        for node in [*g.history(agent), *g.inbox(agent)]
    )


def _stamp(node: Node, started: float) -> None:
    node.started_at, node.finished_at = started, time.time()


def messages_to_query(messages: list[dict[str, Any]]) -> str:
    """Project chat messages into a single agent task."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        parts.append(f"{role}: {content if isinstance(content, str) else str(content)}")
    return "\n\n".join(parts)


class FlowLLM:
    """Expose a :class:`Flow` through a sync ``chat(messages)`` API."""

    def __init__(self, flow: Flow, config: AgentConfig | str | None = None) -> None:
        self.flow = flow
        self.config = config
        self.last_graph: Graph | None = None
        self.last_usage = LLMUsage()

    def chat(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        """The agent's result as text: strings as they are, other JSON values encoded."""
        graph = Graph()
        agent = graph.create_agent(messages_to_query(messages), config=self.config)
        self.flow.run(graph)
        self.last_graph = graph
        self.last_usage = graph.usage()
        result = graph.result(agent)
        return result if isinstance(result, str) else json.dumps(result)

    def completion(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], *args, **kwargs)

    def close(self) -> None:
        self.flow.runtime.close_repls()


__all__ = ["BUDGET_FINAL_TEXT", "Flow", "FlowLLM", "StepUntil", "messages_to_query", "recorded_cells"]
