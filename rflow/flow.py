"""Graph-first Flow harness with normal async child delegation.

Flow owns rflow semantics: graph mutations, prompts, LLM calls, REPL setup, and
tools. Scheduling is delegated to :class:`AsyncTaskQueue`, and event delivery is
delegated to :class:`EventStream`.

There is no local coroutine suspension protocol. A parent agent that runs
``await launch_subagents(...)`` stays in normal Python async code while Flow
parks that task, runs children, commits ``ResumeAction``, and resolves the
parent future.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

import rflow.prompts.projection as prompt_projection
from rflow.base import BaseFlow
from rflow.clients.llm import LLMClient, LLMUsage
from rflow.clients.llm_channel import LLMChannel
from rflow.event_stream import EventStream
from rflow.graph import (
    ChildHandle,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    GraphEvent,
    GraphNodeCommitted,
    LLMAction,
    LLMOutput,
    Node,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rflow.graph.actions import Action, ActionPlan, CallLLM, Exec, Recover
from rflow.integrations.structured import (
    Schema,
    StructuredOutputParser,
    json_schema_for,
)
from rflow.launches import LaunchWaiter, complete_launch, prepare_child_launch
from rflow.prompts import DEFAULT_BUILDER, SYSTEM_PROMPT, PromptBuilder, messages
from rflow.runtime.code import check_wait_syntax, find_code_blocks
from rflow.runtime.context import EngineContext
from rflow.runtime.env import agent_process_env
from rflow.runtime.runtime import LocalRuntime, RemoteRepl, ReplBackend, Runtime
from rflow.scheduler import AsyncTaskQueue
from rflow.tools import tool
from rflow.tools.builtins import (
    DEFAULT_MAX_QUERY_CHARS,
    make_done,
    make_get_subagent_result,
    make_launch_subagents,
    make_show_vars,
    make_spawn_child,
)

_step_context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "rflow_step_context", default=None
)


class Flow(BaseFlow):
    """Recursive agent harness.

    The public customization surface stays boring: override prompts, tool
    building, LLM calls, runtime creation, or graph commit behavior. The loop
    itself is a normal async work queue.
    """

    thread_safe: bool = False

    def __init__(
        self,
        llm: LLMClient,
        *,
        llm_clients: dict[str, LLMClient] | None = None,
        llm_max_concurrency: int | None = None,
        llm_request_timeout: float | None = 600,
        llm_thread_safe: dict[str, bool] | None = None,
        max_depth: int = 3,
        max_iters: int | None = 20,
        child_max_iters: int | None = 20,
        max_concurrency: int = 8,
        max_output_length: int = 4_000,
        max_budget: int | None = None,
        max_messages: int | None = None,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        eager_children: bool = False,
        pool: object | None = None,
        system_prompt: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        show_vars: bool = False,
        runtime: Runtime | None = None,
        enable_structured_output: bool = True,
        include_llm_query: bool = False,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth
        self.max_iters = max_iters
        self.child_max_iters = child_max_iters
        self.max_concurrency = max_concurrency
        self.max_output_length = max_output_length
        self.max_budget = max_budget
        self.max_messages = max_messages
        self.max_query_chars = max_query_chars
        self.eager_children = eager_children
        self.pool = pool
        self.enable_structured_output = enable_structured_output
        self.include_llm_query = include_llm_query
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.show_vars = show_vars
        self.runtime = runtime or LocalRuntime()

        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self._llm_channel = LLMChannel(
            self._llm_clients,
            max_concurrency=llm_max_concurrency or max_concurrency,
            request_timeout=llm_request_timeout,
            thread_safe=llm_thread_safe,
        )
        self.output_parser = StructuredOutputParser()
        self.last_usage: LLMUsage | None = None

        self.graph: Graph | None = None
        self.repls: dict[str, ReplBackend] = {}
        self._step = 0
        self._terminate_requested: set[str] = set()
        self._lock = threading.RLock()
        self._queue: AsyncTaskQueue | None = None
        self._launch_waiters: list[LaunchWaiter] = []
        self._events: EventStream[GraphEvent] = EventStream()
        self._capture_events: list[GraphEvent] | None = None
        self.last_step_events: list[GraphEvent] = []

    @property
    def llm_clients(self) -> dict[str, LLMClient]:
        return self._llm_clients

    def close(self) -> None:
        self._events.close()
        for repl in self.repls.values():
            try:
                repl.close()
            except Exception:  # noqa: BLE001
                pass
        self.repls = {}
        try:
            self.runtime.close()
        except Exception:  # noqa: BLE001
            pass
        self._llm_channel.shutdown()

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> Graph:
        inputs = self._validate_inputs(inputs)
        self.graph = Graph(
            agent_id="root",
            depth=0,
            query=query,
            inputs=inputs,
            output_schema=json_schema_for(output_schema) if output_schema else None,
        )
        self.repls = {}
        self.runtime.clear_graph_sync_cache()
        self._step = 0
        self._terminate_requested = set()
        self._queue = None
        self._launch_waiters = []
        self._ensure_system_prompt_current(self.graph)
        self.commit_node(
            self.graph, UserQuery(content=self.first_prompt(query, inputs, depth=0))
        )
        return self.graph

    def run(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> str:
        graph = self.start(query, inputs, output_schema=output_schema)
        self._drive()
        return graph.result()

    async def run_stream(
        self,
        query: str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
        until: str | Callable[[GraphEvent, Graph], bool] = "idle",
        n: int | None = None,
    ) -> AsyncIterator[GraphEvent]:
        self.start(query, inputs, output_schema=output_schema)
        drive = asyncio.create_task(self._drive_async(until=until, n=n))
        async def close_when_done() -> None:
            try:
                await drive
            finally:
                self._events.close()

        closer = asyncio.create_task(close_when_done())
        try:
            async for event in self._events.subscribe():
                yield event
            await drive
        finally:
            if not drive.done():
                drive.cancel()
            if not closer.done():
                closer.cancel()

    def step(
        self,
        override_graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
        until: str | Callable[[GraphEvent, Graph], bool] = "wave",
        n: int | None = None,
    ) -> Graph:
        self.setup_step(
            override_graph=override_graph,
            query=query,
            inputs=inputs,
            output_schema=output_schema,
        )
        self._drive(until=until, n=n)
        return self.graph

    def setup_step(
        self,
        *,
        override_graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
    ) -> Graph:
        if override_graph is not None:
            self.set_graph(override_graph)
        if self.graph is None:
            if query is None:
                raise RuntimeError("step() needs a query or an existing graph")
            self.start(query, inputs, output_schema=output_schema)
        else:
            if output_schema is not None:
                self.graph.output_schema = json_schema_for(output_schema)
            if query is not None:
                self._add_user_turn(query, inputs)
        return self.sync_graph_state()

    def set_graph(self, graph: Graph) -> Graph:
        same = self.graph is not None and graph.to_dict() == self.graph.to_dict()
        if not same:
            for aid in list(self.repls):
                self.runtime.discard_repl(self.repls, aid)
            self._terminate_requested = set()
            self._queue = None
            self._launch_waiters = []
        self.graph = graph.copy(deep=True)
        self.runtime.clear_graph_sync_cache()
        self._step = self.graph.max_global_step() or 0
        return self.graph

    def terminate(self, agent_ids: Iterable[str] | None = None) -> Graph:
        assert self.graph is not None, "call start() before terminate()"
        targets = agent_ids if agent_ids is not None else list(self.graph.agents)
        for aid in targets:
            agent = self.graph.agents.get(aid)
            if agent is not None and not agent.finished:
                self._terminate_requested.add(aid)
        return self.graph

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        query = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        result = self.run(query)
        inp, out = self.graph.tokens() if self.graph is not None else (0, 0)
        usage = LLMUsage(input_tokens=inp, output_tokens=out)
        self.last_usage = usage
        return result, usage

    def tui(self, *, max_steps_per_turn: int | None = None) -> Graph | None:
        from rflow.integrations.tui import run_tui

        return run_tui(self, max_steps_per_turn=max_steps_per_turn)

    # ── async scheduler ───────────────────────────────────────────────

    def _drive(
        self,
        *,
        until: str | Callable[[GraphEvent, Graph], bool] = "idle",
        n: int | None = None,
    ) -> None:
        asyncio.run(self._drive_async(until=until, n=n))

    async def _drive_async(
        self,
        *,
        until: str | Callable[[GraphEvent, Graph], bool] = "idle",
        n: int | None = None,
    ) -> None:
        if self.graph is None:
            raise RuntimeError("Flow has no graph")
        if n is not None and n < 1:
            raise ValueError("n must be >= 1")
        if not (until in {"event", "wave", "idle"} or callable(until)):
            raise ValueError(f"unknown step boundary: {until!r}")
        self.sync_graph_state()
        self._queue = AsyncTaskQueue(max_concurrency=self.max_concurrency)
        self._launch_waiters = []
        events: list[GraphEvent] = []
        previous_capture = self._capture_events
        self._capture_events = events
        try:
            while not self.graph.finished:
                before = len(events)
                self._resolve_waiters()
                self._enqueue_runnable()
                if self._queue.idle:
                    break
                await self._queue.run_until_idle()
                if self._boundary_reached(
                    until=until,
                    n=n,
                    events=events,
                    new_events=events[before:],
                ):
                    break
        finally:
            self.last_step_events = list(events)
            self._capture_events = previous_capture
            self._queue = None

    def _enqueue_runnable(self) -> None:
        if self.graph is None or self._queue is None:
            return
        waiting = {w.parent_agent_id for w in self._launch_waiters}
        already = self._queue.pending_keys | self._queue.active_keys
        runnable = [
            aid
            for aid in self.graph.get_runnable_nodes()
            if aid not in waiting and aid not in already
        ]
        plan = self.plan_for(self.graph, runnable)
        self._queue.submit_many(
            [
                (agent_id, lambda agent_id=agent_id: self._run_agent(agent_id))
                for agent_id in plan
            ]
        )

    async def _run_agent(self, agent_id: str) -> None:
        agent = self.graph.agents.get(agent_id) if self.graph is not None else None
        if agent is None or agent.finished:
            return
        action = self.plan_one(agent)
        if action is None:
            return
        token = _step_context.set(self._next_step())
        try:
            await self.act_async(action)
        finally:
            _step_context.reset(token)
            self._resolve_waiters()

    def _boundary_reached(
        self,
        *,
        until: str | Callable[[GraphEvent, Graph], bool],
        n: int | None,
        events: list[GraphEvent],
        new_events: list[GraphEvent],
    ) -> bool:
        if self.graph is None:
            return True
        if n is not None and len(events) >= n:
            return True
        if until == "idle":
            return False
        if until == "wave":
            return True
        if until == "event":
            return bool(new_events)
        return any(until(event, self.graph) for event in new_events)

    def _next_step(self) -> int:
        with self._lock:
            self._step += 1
            return self._step

    def _resolve_waiters(self) -> None:
        if self.graph is None:
            return
        agents = self.graph.agents
        pending: list[LaunchWaiter] = []
        for waiter in self._launch_waiters:
            ready = all(
                aid in agents and agents[aid].finished
                for aid in waiter.launch.agent_ids
            )
            if not ready:
                pending.append(waiter)
                continue
            parent = agents.get(waiter.parent_agent_id)
            if parent is not None and isinstance(parent.current(), SupervisingOutput):
                self.commit_node(
                    parent, ResumeAction(resumed_from=list(waiter.launch.agent_ids))
                )
            results = complete_launch(waiter, self._child_result)
            if not waiter.future.done():
                waiter.future.set_result(results)
        self._launch_waiters = pending
        if self._queue is not None:
            self._queue.wake()

    async def submit_children(
        self,
        parent_agent_id: str,
        specs: list[dict[str, Any]],
        launch_names: list[str],
    ) -> list[object]:
        if self.graph is None or self._queue is None:
            raise RuntimeError("launch_subagents(...) needs a running Flow scheduler")
        launch = prepare_child_launch(
            parent_agent_id=parent_agent_id,
            specs=specs,
            launch_names=launch_names,
            spawn_child=self.spawn_child,
        )
        if not launch.agent_ids:
            return launch.results

        parent = self.graph[parent_agent_id]
        pre = self._drain_repl_output(parent_agent_id)
        self.commit_node(
            parent,
            SupervisingOutput(
                output=pre,
                content=self.format_exec_output(pre) if pre.strip() else "",
                waiting_on=list(launch.agent_ids),
                launch_id=launch.launch_id,
                launch_specs=launch.launch_specs,
                launch_names=launch.launch_names,
            ),
        )
        future: asyncio.Future[list[object]] = (
            asyncio.get_running_loop().create_future()
        )
        self._launch_waiters.append(
            LaunchWaiter(parent_agent_id=parent_agent_id, launch=launch, future=future)
        )
        self._enqueue_runnable()
        return await self._queue.park(future)

    def launch_subagents_for(
        self, parent_agent_id: str
    ) -> Callable[[list[dict[str, Any]], list[str]], Any]:
        async def launch(
            specs: list[dict[str, Any]], launch_names: list[str]
        ) -> list[object]:
            return await self.submit_children(parent_agent_id, specs, launch_names)

        return launch

    # ── planning and actions ──────────────────────────────────────────

    def plan(self, graph: Graph) -> ActionPlan:
        return self.plan_for(graph, graph.get_runnable_nodes())

    def plan_for(self, graph: Graph, agent_ids: Iterable[str]) -> ActionPlan:
        agents = graph.agents
        plan: ActionPlan = {}
        for aid in agent_ids:
            agent = agents.get(aid)
            if agent is None:
                continue
            action = self.plan_one(agent)
            if action is not None:
                plan[aid] = action
        return plan

    def plan_one(self, agent: Graph) -> Action | None:
        cur = agent.current()
        if cur is None or cur.terminal:
            return None
        if isinstance(cur, SupervisingOutput):
            return Recover(agent.agent_id, cur.launch_id or cur.id)
        if isinstance(cur, LLMOutput):
            return Exec(agent.agent_id)
        if isinstance(cur, (UserQuery, ExecOutput, ErrorOutput)):
            last_terminal = next(
                (
                    i
                    for i in range(len(agent.nodes) - 1, -1, -1)
                    if agent.nodes[i].terminal
                ),
                -1,
            )
            iters = sum(
                isinstance(n, LLMAction) for n in agent.nodes[last_terminal + 1 :]
            )
            max_iter = (
                agent.max_iters if agent.max_iters is not None else self.max_iters
            )
            force_final = (max_iter is not None and iters >= max_iter) or (
                agent.agent_id in self._terminate_requested
            )
            return CallLLM(agent.agent_id, force_final=force_final)
        return None

    def act(self, action: Action) -> None:
        asyncio.run(self.act_async(action))

    async def act_async(self, action: Action) -> None:
        agent = self._find(action.agent_id)
        cur = agent.current()
        if cur is None or cur.terminal:
            return
        over = self._budget_exceeded(agent)
        if over is not None:
            self.commit_node(
                agent,
                DoneOutput(
                    result=f"[budget exceeded: {over} tokens]", output="", content=""
                ),
            )
            return
        if isinstance(action, CallLLM) and isinstance(
            cur, (UserQuery, ExecOutput, ErrorOutput)
        ):
            await self.step_llm_async(agent, force_final=action.force_final)
        elif isinstance(action, Exec) and isinstance(cur, LLMOutput):
            await self.step_exec_async(agent, cur)
        elif isinstance(action, Recover) and isinstance(cur, SupervisingOutput):
            self.step_recover_supervising(agent, cur)

    def step_llm(self, agent: Graph, *, force_final: bool) -> None:
        asyncio.run(self.step_llm_async(agent, force_final=force_final))

    async def step_llm_async(self, agent: Graph, *, force_final: bool) -> None:
        model_key = agent.model or "default"
        self.commit_node(agent, LLMAction(model=model_key))
        rendered = self.build_messages(agent, force_final=force_final)
        try:
            reply, usage = await asyncio.to_thread(
                self.call_llm, rendered, model=model_key
            )
        except Exception as exc:  # noqa: BLE001
            self.commit_node(
                agent,
                ErrorOutput(
                    error="llm_exception",
                    content=f"LLM call failed: {type(exc).__name__}: {exc}",
                ),
            )
            return
        self.record_usage(usage)
        blocks = find_code_blocks(reply)
        client = self.llm_client_for(agent, model=model_key)
        self.commit_node(
            agent,
            LLMOutput(
                reply=reply,
                code=blocks[0] if blocks else "",
                model=getattr(client, "model", model_key),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )

    def step_exec(self, agent: Graph, llm_output: LLMOutput) -> None:
        asyncio.run(self.step_exec_async(agent, llm_output))

    async def step_exec_async(self, agent: Graph, llm_output: LLMOutput) -> None:
        code = llm_output.code
        self.commit_node(agent, ExecAction(code=code))
        if not code:
            self.commit_node(
                agent,
                ErrorOutput(
                    error="no_code_block", content=self.no_code_block_message()
                ),
            )
            return
        await self.run_exec_async(agent, code)

    def run_exec(self, agent: Graph, code: str) -> None:
        asyncio.run(self.run_exec_async(agent, code))

    async def run_exec_async(self, agent: Graph, code: str) -> None:
        err = self.validate_code(code)
        if err is not None:
            self.commit_node(agent, ErrorOutput(error="invalid_wait", content=err))
            return
        try:
            repl = self.repl_for(agent)
            repl.engine_context.done_result = None
            if isinstance(repl, RemoteRepl):
                payload = await asyncio.to_thread(repl.start, code)
            else:
                start_async = getattr(repl, "start_async", None)
                payload = (
                    await start_async(code)
                    if callable(start_async)
                    else await asyncio.to_thread(repl.start, code)
                )
        except Exception as exc:  # noqa: BLE001
            if agent.agent_id in self.repls:
                self.runtime.discard_repl(self.repls, agent.agent_id)
            self.commit_node(
                agent,
                ErrorOutput(
                    error="exec_exception",
                    content=f"REPL execution failed: {type(exc).__name__}: {exc}",
                ),
            )
            return
        if isinstance(payload, tuple) and len(payload) == 2 and payload[0] is True:
            self.record_legacy_suspension(agent, payload[1])
            return
        if isinstance(payload, tuple) and len(payload) == 2 and payload[0] is False:
            payload = payload[1]
        self.record_observation(agent, repl, payload)

    # ── observations and recovery ─────────────────────────────────────

    def record_observation(
        self, agent: Graph, repl: ReplBackend, payload: object
    ) -> None:
        out = self.truncate_output(payload if isinstance(payload, str) else "")
        done = repl.engine_context.done_result
        if done is not None:
            self.commit_node(agent, DoneOutput(result=done, output=out, content=out))
            if agent.agent_id in self.repls:
                self.runtime.discard_repl(self.repls, agent.agent_id)
            return
        out = out if out.strip() else "(no output)"
        if repl.errored:
            self.commit_node(
                agent,
                ErrorOutput(
                    error="exec_exception",
                    output=out,
                    content=self.format_exec_output(out),
                ),
            )
            return
        self.commit_node(
            agent, ExecOutput(output=out, content=self.format_exec_output(out))
        )

    def record_legacy_suspension(self, agent: Graph, payload: object) -> None:
        request, pre = payload  # type: ignore[misc]
        pre = self.truncate_output(pre)
        self.commit_node(
            agent,
            SupervisingOutput(
                output=pre,
                content=self.format_exec_output(pre) if pre.strip() else "",
                waiting_on=list(request.agent_ids),
                launch_id=request.launch_id,
                launch_specs=list(request.launch_specs),
                launch_names=list(request.launch_names),
            ),
        )

    def step_recover_supervising(self, agent: Graph, sup: SupervisingOutput) -> None:
        if not self._can_resume(sup):
            return
        self.commit_node(
            agent,
            ExecOutput(
                output="",
                content=self.recovery_prompt(agent, sup),
                resumed_from=list(sup.waiting_on),
            ),
        )

    def recovery_prompt(self, agent: Graph, sup: SupervisingOutput) -> str:
        launch_id = sup.launch_id or sup.id
        launch_label = self._launch_label(agent, sup)
        children = "\n".join(f"- `{aid}`" for aid in sup.waiting_on)
        return "\n\n".join(
            [
                "You are recovering this agent after a delegated subagent call.",
                "The original live Python coroutine is unavailable.",
                (
                    f"Delegation `{launch_label}` is complete. Its immediate "
                    "child results are available in original launch order."
                ),
                f"Recovery id: `{launch_id}`",
                f"Immediate children:\n{children or '- <none>'}",
                (
                    "Call:\n"
                    f"```repl\nresults = get_subagent_result({launch_id!r})\n```"
                ),
                "Then continue this agent's task from those graph-backed results.",
            ]
        )

    def _launch_label(self, agent: Graph, sup: SupervisingOutput) -> str:
        if sup.launch_id:
            return sup.launch_id
        names = list(sup.launch_names or [])
        if not names:
            prefix = agent.agent_id + "."
            names = [
                aid.removeprefix(prefix).split(".", 1)[0] for aid in sup.waiting_on
            ]
        if not names:
            return sup.id
        label = "-".join(names[:3])
        if len(names) > 3:
            label += f"-plus-{len(names) - 3}"
        return f"legacy-launch-{label}"

    # ── child graphs and runtime setup ────────────────────────────────

    def spawn_child(
        self,
        parent_agent_id: str,
        name: str,
        query: str,
        inputs: dict[str, str] | None = None,
        model: str = "default",
        output_schema: Schema | None = None,
        *,
        strict_name: bool = False,
    ) -> ChildHandle | str:
        inputs = self._validate_inputs(inputs)
        if len(query) > self.max_query_chars:
            return (
                f"[refused: query too long ({len(query)} chars > "
                f"{self.max_query_chars})] keep query a short instruction and "
                "move bulk payloads into inputs"
            )
        with self._lock:
            parent = self.graph[parent_agent_id]
            if parent.depth >= self.max_depth:
                return f"[refused: max depth {self.max_depth}] do this inline"
            base_child_id = f"{parent_agent_id}.{name}"
            if strict_name and base_child_id in parent.children:
                raise ValueError(
                    f"child id {base_child_id!r} already exists; choose a unique "
                    "`name` for this launch_subagents(...) spec"
                )
            child_id = (
                base_child_id
                if strict_name
                else self._unique_id(parent_agent_id, name, parent)
            )
            parent_node = parent.current()
            child = Graph(
                agent_id=child_id,
                depth=parent.depth + 1,
                query=query,
                inputs=inputs,
                model=None if model == "default" else model,
                max_iters=self.child_max_iters,
                output_schema=(
                    json_schema_for(output_schema)
                    if output_schema is not None
                    else None
                ),
                parent_agent_id=parent_agent_id,
                parent_node_id=parent_node.id if parent_node else None,
            )
            self._ensure_system_prompt_current(child)
            parent.children[child_id] = child
            self.commit_node(
                child,
                UserQuery(
                    content=self.first_prompt(query, inputs, depth=parent.depth + 1)
                ),
            )
        return ChildHandle(child_id)

    @staticmethod
    def _unique_id(parent_id: str, name: str, parent: Graph) -> str:
        base = f"{parent_id}.{name}"
        if base not in parent.children:
            return base
        i = 1
        while f"{base}_{i}" in parent.children:
            i += 1
        return f"{base}_{i}"

    def repl_for(self, agent: Graph) -> ReplBackend:
        repl = self.repls.get(agent.agent_id)
        if repl is None:
            repl = self.make_repl(agent)
            self.seed_agent_context(repl, agent)
            inputs = agent.repl_inputs()
            tools = self.build_tools(repl.engine_context)
            if isinstance(repl, RemoteRepl):
                repl.seed(tools, inputs, max_query_chars=self.max_query_chars)
            else:
                repl.namespace.update(tools)
                if self.show_vars:
                    repl.namespace["SHOW_VARS"] = make_show_vars(repl.namespace)
                repl.namespace["INPUTS"] = inputs
            self.repls[agent.agent_id] = repl
            self.runtime.repl_env_cache[agent.agent_id] = dict(repl.process_env)
            self.runtime.repl_inputs_cache[agent.agent_id] = agent.repl_inputs()
        else:
            self._sync_repl_with_graph(repl, agent)
        return repl

    def make_repl(self, agent: Graph) -> ReplBackend:
        return self.runtime.open(agent)

    def seed_agent_context(self, repl: ReplBackend, agent: Graph) -> None:
        repl.engine_context = EngineContext(
            agent_id=agent.agent_id,
            output_schema=agent.output_schema,
            launch_children=self.launch_subagents_for(agent.agent_id),
        )
        repl.process_env.update(self._agent_process_env(agent))

    def _sync_repl_with_graph(self, repl: ReplBackend, agent: Graph) -> None:
        self.runtime.sync_repl(
            repl,
            agent,
            env=self._agent_process_env(agent),
            inputs=agent.repl_inputs(),
        )
        repl.engine_context.launch_children = self.launch_subagents_for(agent.agent_id)

    def sync_graph_state(self) -> Graph:
        if self.graph is None:
            raise RuntimeError("sync_graph_state() needs a graph")
        agents = self.graph.agents
        for agent in agents.values():
            self._ensure_system_prompt_current(agent)
        for aid in list(self.repls):
            agent = agents.get(aid)
            if agent is None:
                self.runtime.discard_repl(self.repls, aid)
            else:
                self._sync_repl_with_graph(self.repls[aid], agent)
        return self.graph

    def build_tools(self, engine_context: EngineContext | None = None) -> dict:
        engine_context = engine_context or EngineContext()
        spawn_child = make_spawn_child(self, engine_context)
        tools = {
            "done": make_done(self, engine_context),
            "launch_subagents": make_launch_subagents(
                engine_context.launch_children,
                max_query_chars=self.max_query_chars,
            ),
            "_rflow_spawn_child": spawn_child,
            "get_subagent_result": make_get_subagent_result(self, engine_context),
        }
        if self.include_llm_query:
            tools["llm_query_batched"] = self.llm_query_batched
        for name, fn in self.runtime.tools.items():
            tools.setdefault(name, fn)
        return tools

    def tool_namespace_for_prompt(self, graph: Graph) -> dict:
        repl = self.repls.get(graph.agent_id)
        if repl is not None:
            return repl.namespace
        return self.build_tools(
            EngineContext(agent_id=graph.agent_id, output_schema=graph.output_schema)
        )

    def _agent_process_env(self, agent: Graph) -> dict[str, str]:
        return agent_process_env(
            agent_id=agent.agent_id,
            depth=agent.depth,
            parent_agent_id=agent.parent_agent_id,
            max_depth=self.max_depth,
        )

    # ── LLM/tool/prompt helpers ───────────────────────────────────────

    def call_llm(
        self, messages: list[dict[str, str]], *, model: str | None = None, **llm_kwargs
    ) -> tuple[str, LLMUsage]:
        return self._llm_channel.call(model or "default", messages, **llm_kwargs)

    def llm_client_for(self, agent: Graph, *, model: str | None = None) -> LLMClient:
        return self._llm_clients.get(model or agent.model or "default", self.llm)

    def record_usage(self, usage: LLMUsage) -> None:
        self.last_usage = usage

    @tool(
        "Send a list of independent one-shot prompts to the model in parallel "
        "and get back a list of replies, in order. Use for simple fanout (no "
        "tools, no REPL). Pass output_schema (a JSON Schema dict) to validate "
        "each reply into a JSON-compatible value.",
        proxy=True,
    )
    def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: Schema | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list:
        if not isinstance(prompts, list) or not all(
            isinstance(p, str) for p in prompts
        ):
            raise TypeError("llm_query_batched(prompts) takes a list[str]")
        if model not in self._llm_clients:
            keys = ", ".join(sorted(self._llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}")
        schema = json_schema_for(output_schema) if output_schema is not None else None
        sent = prompts
        if schema is not None:
            hint = self._schema_instruction(schema)
            sent = [f"{p}\n\n{hint}" for p in prompts]
        llm_kwargs = {
            key: value
            for key, value in (
                ("temperature", temperature),
                ("top_p", top_p),
                ("max_tokens", max_tokens),
                ("stop", stop),
            )
            if value is not None
        }
        pairs = self._llm_channel.batch(model, sent, **llm_kwargs)
        self.record_usage(
            LLMUsage(
                input_tokens=sum(u.input_tokens for _, u in pairs),
                output_tokens=sum(u.output_tokens for _, u in pairs),
            )
        )
        texts = [text for text, _ in pairs]
        if schema is not None:
            return [self.output_parser(text, schema) for text in texts]
        return texts

    def build_messages(
        self, graph: Graph, *, force_final: bool = False
    ) -> list[dict[str, str]]:
        self._ensure_system_prompt_current(graph)
        system_prompt = graph.system_prompt or self.build_system_prompt(graph)
        return prompt_projection.build_messages(
            graph,
            system_prompt=system_prompt,
            max_messages=self.max_messages,
            continue_nudge=messages.create_nudge_message(),
            final_nudge=messages.create_final_action_message(),
            force_final=force_final,
        )

    def build_system_prompt(self, graph: Graph) -> str:
        if self.system_prompt is not None:
            schema = None
            if graph.output_schema is not None and self.enable_structured_output:
                schema = self._schema_instruction(graph.output_schema)
            return prompt_projection.build_system_prompt(
                self.system_prompt, schema_instruction=schema
            )
        return self.prompt_builder.build(self, graph)

    def _ensure_system_prompt_current(self, agent: Graph) -> None:
        fp = json.dumps(
            {
                "output_schema": agent.output_schema,
                "depth": agent.depth,
                "max_depth": self.max_depth,
                "enable_structured_output": self.enable_structured_output,
                "show_vars": self.show_vars,
                "runtime_tools": sorted(self.runtime.tools),
                "prompt_builder": id(self.prompt_builder),
                "system_prompt": self.system_prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.runtime.prompt_fingerprints.get(agent.agent_id) == fp:
            return
        agent.system_prompt = self.build_system_prompt(agent)
        self.runtime.prompt_fingerprints[agent.agent_id] = fp

    def first_prompt(
        self, query: str, inputs: dict[str, str] | None = None, *, depth: int = 0
    ) -> str:
        return prompt_projection.first_prompt(
            query, inputs or {}, depth=depth, max_depth=self.max_depth
        )

    def followup_prompt(self, query: str, *, depth: int = 0) -> str:
        return prompt_projection.followup_prompt(
            query, depth=depth, max_depth=self.max_depth
        )

    def format_exec_output(self, output: str) -> str:
        return prompt_projection.format_exec_output(output)

    def _schema_instruction(self, schema: Schema) -> str:
        return prompt_projection.schema_instruction(
            self.output_parser.system_prompt_hint(schema)
        )

    def no_code_block_message(self) -> str:
        return messages.NO_CODE_BLOCK

    def validate_code(self, code: str) -> str | None:
        return check_wait_syntax(code)

    def truncate_output(self, output: str) -> str:
        limit = self.max_output_length
        if limit and len(output) > limit:
            omitted = len(output) - limit
            return (
                output[:limit]
                + f"\n...<truncated {omitted} chars; keep full data in variables>"
            )
        return output

    def _budget_exceeded(self, agent: Graph) -> int | None:
        if self.max_budget is None:
            return None
        used = agent.total_tokens()
        return used if used >= self.max_budget else None

    def _can_resume(self, sup: SupervisingOutput) -> bool:
        agents = self.graph.agents
        return all(aid in agents and agents[aid].finished for aid in sup.waiting_on)

    def _child_result(self, agent_id: str) -> object:
        child = self.graph.agents.get(agent_id)
        if not (child and child.finished):
            return ""
        result = child.result()
        if child.output_schema is not None and result:
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return result
        return result

    def _drain_repl_output(self, agent_id: str) -> str:
        repl = self.repls.get(agent_id)
        if repl is None:
            return ""
        drain = getattr(repl, "drain_output", None)
        if callable(drain):
            return self.truncate_output(drain())
        return self.truncate_output(getattr(repl, "_output", ""))

    def _add_user_turn(self, query: str, inputs: dict[str, str] | None) -> None:
        root = self.graph
        extra = self._validate_inputs(inputs)
        root.query = query
        if extra:
            root.inputs = {**root.inputs, **extra}
        self.commit_node(
            root, UserQuery(content=self.followup_prompt(query, depth=root.depth))
        )

    _RESERVED = frozenset(
        {
            "done",
            "launch_subagents",
            "get_subagent_result",
            "_rflow_spawn_child",
            "llm_query_batched",
            "query",
            "HISTORY",
            "SHOW_VARS",
        }
    )

    @classmethod
    def _validate_inputs(cls, inputs: dict[str, str] | None) -> dict[str, str]:
        if inputs is None:
            return {}
        if not isinstance(inputs, dict):
            raise TypeError("inputs must be a dict of {str: str}")
        clean: dict[str, str] = {}
        for name, value in inputs.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"input name {name!r} must be a valid identifier")
            if name in cls._RESERVED:
                raise ValueError(f"input name {name!r} is reserved")
            if not isinstance(value, str):
                raise TypeError(
                    f"input {name!r} must be a str "
                    f"(JSON-encode structured values); got {type(value).__name__}"
                )
            clean[name] = value
        return clean

    # ── graph mutation ────────────────────────────────────────────────

    def commit_node(self, agent: Graph, node: Node) -> Node:
        with self._lock:
            prev = agent.nodes[-1] if agent.nodes else None
            seq = prev.seq + 1 if prev else 0
            stamped = node.update(
                agent_id=agent.agent_id,
                seq=seq,
                global_step=_step_context.get() or self._step,
            )
            agent.nodes.append(stamped)
        event = GraphNodeCommitted(
            type="node_committed",
            agent_id=stamped.agent_id,
            node_id=stamped.id,
            node_type=stamped.type,
            seq=stamped.seq,
            global_step=stamped.global_step,
        )
        if self._capture_events is not None:
            self._capture_events.append(event)
        self._events.publish(event)
        return stamped

    def _find(self, agent_id: str) -> Graph:
        with self._lock:
            return self.graph[agent_id]


def parallel_step(
    items: Sequence[
        Flow | tuple[Flow] | tuple[Flow, Graph | None] | tuple[Flow, dict[str, Any]]
    ],
    *,
    pool: object | None = None,
) -> list[Graph]:
    calls = [_parallel_step_call(item) for item in items]
    if not calls:
        return []

    async def run_all() -> list[Graph]:
        async def run_one(flow: Flow, kwargs: dict[str, Any]) -> Graph:
            flow.setup_step(**kwargs)
            await flow._drive_async()
            return flow.graph.snapshot()

        return await asyncio.gather(*(run_one(flow, kwargs) for flow, kwargs in calls))

    return asyncio.run(run_all())


def _parallel_step_call(
    item: Flow | tuple[Flow] | tuple[Flow, Graph | None] | tuple[Flow, dict[str, Any]],
) -> tuple[Flow, dict[str, Any]]:
    if isinstance(item, Flow):
        return item, {}
    if not isinstance(item, tuple) or not item:
        raise TypeError(
            "parallel_step(...) items must be Flow or tuple starting with Flow"
        )
    flow = item[0]
    if not isinstance(flow, Flow):
        raise TypeError("parallel_step(...) item[0] must be a Flow")
    kwargs: dict[str, Any] = {}
    if len(item) == 1:
        return flow, kwargs
    if len(item) != 2:
        raise TypeError(
            "parallel_step(...) tuples must be (flow,), (flow, override_graph), "
            "or (flow, step_kwargs)"
        )
    extra = item[1]
    if isinstance(extra, dict):
        kwargs.update(extra)
    else:
        kwargs["override_graph"] = extra
    return flow, kwargs


__all__ = ["Flow", "parallel_step", "find_code_blocks", "SYSTEM_PROMPT"]
