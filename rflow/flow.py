"""Minimal graph-first Flow with an async graph event stream."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from rflow.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    LLMUsage,
    Node,
    ResumeAction,
    SupervisingOutput,
    apply_graph_action,
)
from rflow.graph.events import (
    Event,
    GraphAction,
    GraphCreated,
    StepUntil,
    is_idle,
)
from rflow.pool import Pool, ThreadPool
from rflow.prompts import DEFAULT_BUILDER, PromptBuilder
from rflow.prompts.messages import build_messages, merge_summary
from rflow.runtime import LocalRuntime, Runtime
from rflow.runtime.env import agent_process_env
from rflow.runtime.repl import ReplLike
from rflow.structured import Schema, StructuredOutputParser, json_schema_for
from rflow.tasks import TaskQueue
from rflow.tools import tool
from rflow.tools.builtins import make_done, make_launch_subagents
from rflow.utils import (
    ReplKey,
    accepts_kwarg,
    budget_exceeded,
    code_block,
    common_prefix_len,
    graph_from_input,
    iter_budget,
    llm_output_metadata,
    repl_key,
    sampling_kwargs,
    tool_name,
    truncate_output,
    usage_from_client,
)

#: Default cap on a subagent ``query`` length: keep queries short instructions and
#: push bulk payloads into ``inputs`` instead.
DEFAULT_MAX_QUERY_CHARS = 2_000

#: Names owned by the harness/graph; ``add_tool``/``remove_tool`` won't touch them.
_RESERVED_TOOL_NAMES = frozenset({"done", "launch_subagents", "INPUTS"})

#: Sentinel results recorded when a run ends without an agent-produced answer.
TERMINATED = "[terminated]"
BUDGET_EXCEEDED = "[budget exceeded]"
MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


FULL_RUN = frozenset({"done", "finished"})


@dataclass
class Run:
    """Per-graph run state: one active streaming run per trajectory (graph_id).

    Engine config stays on ``Flow`` (shared, graph-agnostic). Each run owns its
    own scheduler and ``until`` boundary.
    """

    graph: Graph
    tasks: TaskQueue
    until: StepUntil = "done"
    #: True while a ``run_streaming`` consumer is actively driving this run.
    #: Guards against two consumers stealing events from one graph's queue.
    streaming: bool = False


class Flow:
    """Tiny recursive harness.

    Graph is the state. Flow is the environment/policy that applies graph
    actions to caller-owned graphs.
    """

    def __init__(
        self,
        llm,
        *,
        max_depth: int = 2,
        max_iters: int = 20,
        child_max_iters: int | None = None,
        max_messages: int | None = None,
        max_output_length: int = 4_000,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        max_budget: int | None = None,
        llm_request_timeout: float | None = None,
        system_prompt: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        tools: list[Any] | None = None,
        runtime: Runtime | None = None,
        llm_clients: dict[str, Any] | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        enable_structured_output: bool = True,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth
        self.max_iters = max_iters
        self.child_max_iters = child_max_iters
        self.max_messages = max_messages
        self.max_output_length = max_output_length
        self.max_query_chars = max_query_chars
        self.max_budget = max_budget
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.tools = {tool_name(fn): fn for fn in tools or []}
        self.runtime = runtime or LocalRuntime()
        # The one pool: caps how many *blocking* leaf calls (sync ``client.chat``)
        # run at once. Agent scheduling is unbounded async and never enters here.
        # ``workers=None`` leaves blocking calls on asyncio's default executor.
        self.pool = pool or ThreadPool(workers=workers)
        self.use_llm_query = use_llm_query
        self.enable_structured_output = enable_structured_output
        self.output_parser = StructuredOutputParser()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.repls: dict[ReplKey, ReplLike] = {}
        self._terminate_requested: set[str] = set()
        #: Active runs keyed by ``graph_id`` (one per trajectory). Created by
        #: ``run_for`` and cleared by ``finish_run``; mirrors how ``repls`` keys.
        self.runs: dict[str, Run] = {}

    def run(
        self,
        *,
        graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
        merge_inputs: bool = True,
    ) -> str:
        return asyncio.run(
            self.arun(
                graph=graph,
                query=query,
                inputs=inputs,
                output_schema=output_schema,
                merge_inputs=merge_inputs,
            )
        )

    async def arun(
        self,
        *,
        graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
        merge_inputs: bool = True,
    ) -> str:
        run = self.resolve_run(
            graph=graph,
            query=query,
            inputs=inputs,
            output_schema=output_schema,
            merge_inputs=merge_inputs,
        )
        async for _event in self._drive(run, until="done"):
            pass
        return run.graph.result()

    async def parallel_stream(
        self,
        *graphs: Graph | str,
        until: StepUntil = "done",
        n: int | None = None,
        close_repls: bool = False,
    ) -> AsyncIterator[Event]:
        """Drive several graphs on this one flow, merging their events into one
        stream. Each graph gets its own run (keyed by ``graph_id``) with its own
        scheduler and ``until`` boundary; every event carries ``graph_id`` so the
        merged stream is self-describing. Graphs must be distinct — two entries
        for the same ``graph_id`` would drive one run twice (``run_streaming``
        rejects the second).
        """
        streams = [
            aiter(
                self.run_streaming(
                    graph=graph_from_input(g),
                    until=until,
                    n=n,
                    close_repls=close_repls,
                )
            )
            for g in graphs
        ]
        # One in-flight "next event" task per live stream; whichever resolves
        # first is yielded, then re-armed. No sentinels, no queue, no bound.
        pending = {asyncio.ensure_future(anext(it)): it for it in streams}
        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    it = pending.pop(task)
                    try:
                        yield task.result()
                    except StopAsyncIteration:
                        continue
                    pending[asyncio.ensure_future(anext(it))] = it
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for it in streams:
                await it.aclose()

    async def parallel_run(
        self,
        *graphs: Graph | str,
        until: StepUntil = "done",
        n: int | None = None,
        close_repls: bool = False,
    ) -> list[Graph]:
        """Drive several graphs to completion, returning them in argument order.

        String queries are coerced to graphs first so the caller gets back the
        live graphs (and their ``result()``) without holding them beforehand.
        """
        coerced = [graph_from_input(g) for g in graphs]
        async for _event in self.parallel_stream(
            *coerced, until=until, n=n, close_repls=close_repls
        ):
            pass
        return coerced

    async def run_streaming(
        self,
        *,
        graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
        merge_inputs: bool = True,
        until: StepUntil = "done",
        n: int | None = None,
        close_repls: bool = False,
    ) -> AsyncIterator[Event]:
        """Stream events while scheduling follow-up work according to ``until``.

        Pass ``query`` alone to start a fresh graph, or ``graph`` alone to resume
        one. Passing both appends ``query`` as a new ``UserQuery`` turn on
        ``graph`` before driving — the long-running / multi-turn case, where one
        graph is re-driven across turns with its full history and warm REPL.
        ``inputs``/``output_schema`` are applied to the graph first, so they take
        effect even when resuming a finished graph; ``inputs`` merges into any
        existing values unless ``merge_inputs=False`` replaces them. See
        :meth:`resolve_run` for this phase-1 resolution.

        A run may only have one active streaming consumer at a time; concurrent
        streaming of the same graph raises. Set ``close_repls=True`` to also tear
        down this graph's REPLs when the run finishes (default keeps them for
        pause/resume, fork, and post-run inspection).
        """
        # Phase 1: resolve/register the run. Phase 2: drive it (see :meth:`_drive`).
        run = self.resolve_run(
            graph=graph,
            query=query,
            inputs=inputs,
            output_schema=output_schema,
            merge_inputs=merge_inputs,
        )
        async for event in self._drive(run, until=until, n=n, close_repls=close_repls):
            yield event

    async def _drive(
        self,
        run: Run,
        *,
        until: StepUntil = "done",
        n: int | None = None,
        close_repls: bool = False,
    ) -> AsyncIterator[Event]:
        """Phase 2: schedule agents and yield events until the ``until`` boundary.

        Assumes ``run`` is already resolved/registered. One active consumer per
        run; concurrent drives of the same graph raise.
        """
        if n is not None and n < 1:
            raise ValueError("n must be >= 1")
        if run.streaming:
            raise RuntimeError("this graph is already being streamed")
        graph = run.graph

        run.streaming = True
        run.until = until
        limit = n or 1
        steps = 0
        exc: BaseException | None = None
        try:
            while True:
                # No buffered events: seed one task per ready agent. Boundaries
                # work by choosing not to enqueue more.
                if not run.tasks.has_events():
                    for agent in graph.walk():
                        self.schedule_agent(run, agent)

                async for item in run.tasks.items():
                    event = item.item
                    yield event
                    agent = graph.agent_for(item.agent_id)
                    # "Stop" is just absence of follow-up work for this agent.
                    if self.should_continue(run, agent, event):
                        self.schedule_agent(run, agent)
                    else:
                        run.tasks.stop(agent.agent_id)

                exc = run.tasks.exception()
                if graph.finished or until in FULL_RUN:
                    break

                # A non-full boundary keeps the scheduler alive for the caller to
                # inspect or edit the graph before the next streaming call.
                steps += 1
                if steps >= limit:
                    return
        finally:
            # Full runs and finished bounded runs own teardown; paused bounded
            # runs keep their queue so a later call can resume from the graph.
            run.streaming = False
            if until in FULL_RUN or graph.finished:
                exc = run.tasks.exception()
                await self.finish_run(run, close_repls=close_repls)
        if exc is not None:
            raise exc

    def run_for(self, graph: Graph) -> Run:
        """Return the active run for ``graph``, creating one if needed.

        Pure run registry, keyed by ``graph.graph_id``: an existing (live or
        paused) run is resumed, otherwise a fresh run is registered. All state
        resolution (coercion, new turns, event emission) happens in
        :meth:`resolve_run` before this is called; this method never mutates the
        graph or emits.
        """
        run = self.runs.get(graph.graph_id)
        if run is None:
            run = self.runs[graph.graph_id] = Run(graph=graph, tasks=TaskQueue())
        return run

    def resolve_run(
        self,
        *,
        graph: Graph | None = None,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
        merge_inputs: bool = True,
    ) -> Run:
        """Resolve graph state and register the run, emitting structural events.

        Phase 1 of driving: turn the caller's request into a live, registered run
        before the phase-2 drive loop touches it. Exactly one of two shapes:

        - ``query`` alone builds a fresh graph, announced with ``GraphCreated``.
        - ``graph`` is resumed; ``inputs``/``output_schema`` are applied in place
          (so they take effect even on a finished graph), then any ``query`` is
          appended as a fresh ``UserQuery`` turn, emitted as an ``AppendNode`` so
          consumers/checkpointers observe it. ``inputs`` merges into existing
          values unless ``merge_inputs=False``; updated values are synced into a
          warm REPL if one already exists.

        Registration happens before emission because :meth:`apply_action` only
        streams events for graphs that already have a run.
        """
        if graph is None:
            if query is None:
                raise ValueError("resolve_run requires graph= or query=")
            graph = Graph(
                query=query, inputs=dict(inputs or {}), output_schema=output_schema
            )
            run = self.run_for(graph)
            self.apply_action(graph, GraphCreated(type="graph_created", graph=graph))
            return run

        if inputs:
            graph.inputs = {**graph.inputs, **inputs} if merge_inputs else dict(inputs)
        if output_schema is not None:
            graph.output_schema = json_schema_for(output_schema)
        run = self.run_for(graph)
        if inputs:
            self._sync_repl_inputs(graph)
        if query is not None:
            self.append_node(graph, query)
        return run

    def _sync_repl_inputs(self, graph: Graph) -> None:
        """Push the graph's current ``inputs`` into a warm REPL, if any.

        A resumed graph may already own a live REPL whose ``INPUTS`` namespace
        was seeded at creation; re-inject so a new turn sees updated values.
        """
        repl = self.repls.get(repl_key(graph))
        if repl is not None:
            repl.inject("INPUTS", dict(graph.inputs))

    async def finish_run(self, run: Run, *, close_repls: bool = False) -> None:
        await run.tasks.aclose()
        self.runs.pop(run.graph.graph_id, None)
        if close_repls:
            self.close_repls(run.graph.graph_id)

    def schedule_agent(self, run: Run, graph: Graph) -> None:
        if not graph.finished:
            run.tasks.add(graph.agent_id, lambda: self.run_agent_task(graph))

    def should_continue(self, run: Run, graph: Graph, event: Event) -> bool:
        if graph.finished:
            return False
        if isinstance(event.node, (SupervisingOutput, ResumeAction)):
            return False
        until = run.until
        if until in FULL_RUN:
            return True
        if until == "next":
            return False
        if until == "idle":
            return not is_idle(event)
        if until == "error":
            return not isinstance(event.node, ErrorOutput)
        if until == "supervising":
            return not isinstance(event.node, SupervisingOutput)
        if callable(until):
            return not bool(until(event, run.graph))
        raise ValueError(f"unknown until boundary: {until!r}")

    async def run_agent_task(self, graph: Graph) -> None:
        """Run exactly one graph-producing unit for an agent."""
        if graph.finished:
            return
        if graph.agent_id in self._terminate_requested:
            self.append_node(graph, DoneOutput(result=TERMINATED))
            return
        run = self.runs.get(graph.graph_id)
        root = run.graph if run is not None else graph
        if budget_exceeded(root, self.max_budget):
            self.append_node(graph, DoneOutput(result=BUDGET_EXCEEDED))
            return

        current = graph.current()
        if isinstance(current, LLMOutput):
            self.append_node(graph, ExecAction(code=current.code))
            return
        if isinstance(current, ExecAction):
            await self.exec_turn(graph, current.code)
            return

        llm_turns = sum(isinstance(node, LLMOutput) for node in graph.nodes)
        max_iters = iter_budget(graph.depth, self.max_iters, self.child_max_iters)
        if llm_turns >= max_iters:
            self.append_node(graph, DoneOutput(result=MAX_ITERS_EXCEEDED))
            return

        messages = self.messages(graph, force_final=llm_turns == max_iters - 1)
        reply, usage = await self._call_chat(messages, graph.model)
        code = code_block(reply)
        metadata = llm_output_metadata(graph.model, usage)
        self.append_node(graph, LLMOutput(content=reply, code=code, metadata=metadata))

    async def run_agent(self, graph: Graph) -> None:
        while not graph.finished:
            await self.run_agent_task(graph)

    def terminate(self, agent_ids: Iterable[str] | None = None) -> None:
        """Cooperatively stop agents (default: all running): each finishes its
        current turn, then records ``done("[terminated]")`` instead of iterating on.
        """
        if agent_ids is None:
            agent_ids = [
                agent_id for run in self.runs.values() for agent_id in run.graph.agents
            ]
        self._terminate_requested.update(agent_ids)

    async def exec_turn(self, graph: Graph, code: str, *, replay: bool = False) -> Node:
        """Observation turn: run the action against the repl -> one result node.

        With ``replay=True`` the node is returned but not appended (used to rebuild
        live repl state from a graph that already holds these result nodes).
        """
        repl = self.repl_for(graph)
        repl.done_result = None
        try:
            out = await repl.run(code)
        except Exception as exc:  # noqa: BLE001
            # Only a dead session reaches here (user-code errors set repl.errored):
            # drop the handle so repl_for respawns next turn.
            self.close_repl(graph)
            out = f"REPL execution failed: {type(exc).__name__}: {exc}"
            node = ErrorOutput(content=out, output=out, error="exec")
            if not replay:
                self.append_node(graph, node)
            return node

        # Cap the observation re-entering context (full data stays in repl vars;
        # done_result is never truncated).
        out = truncate_output(out, self.max_output_length)
        if repl.done_result is not None:
            node = DoneOutput(content=out, output=out, result=repl.done_result)
        elif repl.errored:
            node = ErrorOutput(content=out, output=out, error="exec")
        else:
            node = ExecOutput(content=out or "(no output)", output=out)
        if not replay:
            self.append_node(graph, node)
        return node

    def repl_for(self, graph: Graph) -> ReplLike:
        key = repl_key(graph)
        repl = self.repls.get(key)
        if repl is None:
            repl = self.runtime.open(graph)
            repl.seed(self.build_tools(graph, repl), graph.inputs)
            repl.update_env(
                agent_process_env(
                    agent_id=graph.agent_id,
                    depth=graph.depth,
                    parent_agent_id=graph.parent_agent_id,
                    max_depth=self.max_depth,
                )
            )
            self.repls[key] = repl
        return repl

    def close_repl(self, graph: Graph) -> None:
        repl = self.repls.pop(repl_key(graph), None)
        if repl is not None:
            with suppress(Exception):
                repl.close()

    def close_repls(self, graph_id: str | None = None) -> None:
        keys = [key for key in self.repls if graph_id is None or key[0] == graph_id]
        for key in keys:
            repl = self.repls.pop(key)
            with suppress(Exception):
                repl.close()

    async def rebuild_repl(
        self, graph: Graph, *, agent_id: str | None = None
    ) -> ReplLike:
        """Reconstruct an agent's REPL by replaying its ``ExecAction`` code blocks
        (no LLM calls, no appended nodes). The correctness floor for forks/reverts.
        """
        agent = graph.agent_for(agent_id)
        self.close_repl(agent)
        for node in agent.nodes:
            if isinstance(node, ExecAction) and node.code:
                await self.exec_turn(agent, node.code, replay=True)
        return self.repl_for(agent)

    async def fork(
        self,
        graph: Graph,
        *,
        from_node_id: str | None = None,
        agent_id: str | None = None,
        keep_anchor: bool = False,
    ) -> Graph:
        """Independent branch of ``graph`` (deepcopy, fresh ``graph_id``) with its
        REPL rebuilt to match the optionally-rewound trajectory. Never mutates parent.

        By default ``from_node_id`` is dropped (branch ends before it, to
        re-decide); pass ``keep_anchor=True`` to retain it and continue from
        after it.
        """
        child = graph.fork(
            from_node_id=from_node_id, agent_id=agent_id, keep_anchor=keep_anchor
        )
        await self.rebuild_repl(child, agent_id=agent_id)
        return child

    async def merge(
        self,
        parent: Graph,
        child: Graph,
        *,
        agent_id: str | None = None,
        summary: str | Callable[[Graph], str] | None = None,
    ) -> None:
        """Fold a child branch's post-fork DELTA back into the parent: keep its
        ``ExecAction``s (replayable, hidden from the prompt) plus one summary node,
        and bring its variables into the parent REPL via ``_merge_repl_state``.
        """
        p, c = parent.agent_for(agent_id), child.agent_for(agent_id)
        delta = c.nodes[common_prefix_len(p.nodes, c.nodes) :]

        for node in delta:
            if isinstance(node, ExecAction):
                self.append_node(p, deepcopy(node))

        if summary is None:
            text = merge_summary(child, delta)
        else:
            text = summary(child) if callable(summary) else summary
        self.append_node(p, ExecOutput(content=text, output=text))

        await self._merge_repl_state(parent, child, delta, agent_id=agent_id)

    async def _merge_repl_state(
        self,
        parent: Graph,
        child: Graph,
        delta: list[Node],
        *,
        agent_id: str | None = None,
    ) -> None:
        """Bring the child branch's variables into the parent's live REPL: adopt
        the child's running REPL when the parent has none (else rebuild from its
        graph); otherwise re-exec just the delta's ``ExecAction`` code.
        """
        p, c = parent.agent_for(agent_id), child.agent_for(agent_id)
        p_key, c_key = repl_key(p), repl_key(c)

        if p_key not in self.repls:
            child_repl = self.repls.pop(c_key, None)
            if child_repl is not None:
                self.repls[p_key] = child_repl
            else:
                await self.rebuild_repl(parent, agent_id=agent_id)
            return

        for node in delta:
            if isinstance(node, ExecAction) and node.code:
                await self.exec_turn(p, node.code, replay=True)

    def discard(self, *children: Graph) -> None:
        """Eagerly close rejected branches' REPLs. Optional (parent was never
        mutated, so GC suffices); only frees remote sandboxes. No-op on LocalRuntime.
        """
        for child in children:
            self.close_repls(child.graph_id)

    def launch_subagents(self, parent: Graph, repl: ReplLike):
        """Public factory for the recursive spawn tool (see ``tools.builtins``)."""
        return make_launch_subagents(self, parent, repl)

    def append_node(self, graph: Graph, node: Node | str) -> Event:
        """Append via :meth:`Graph.inject_action` and emit into the active run."""
        return self.apply_action(graph, graph.inject_action(node))

    def apply_action(self, graph: Graph, action: GraphAction) -> GraphAction:
        base = None if isinstance(action, GraphCreated) else graph
        apply_graph_action(base, action)
        run = self.runs.get(graph.graph_id)
        if run is not None:
            # Stamp originating graph_id so merged streams (parallel_stream) self-describe.
            event = (
                action
                if isinstance(action, GraphCreated)
                else replace(action, graph_id=graph.graph_id)
            )
            agent_id = (
                getattr(event, "agent_id", None)
                or getattr(event, "parent_agent_id", None)
                or graph.agent_id
            )
            run.tasks.emit(agent_id, event)
        return action

    def messages(
        self, graph: Graph, *, force_final: bool = False
    ) -> list[dict[str, str]]:
        return build_messages(
            graph,
            self.build_system_prompt(graph),
            max_messages=self.max_messages,
            force_final=force_final,
        )

    def build_system_prompt(self, graph: Graph) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        return self.prompt_builder.build(self, graph)

    async def _call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        **llm_kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        # The single leaf call: blocking clients run on the pool's threads
        # (capped by ``workers``), async clients stay on the loop.
        client = self.llm_client(model)
        # ``wait_for`` can't cancel a blocking thread mid-flight, so for blocking
        # clients we also push ``timeout`` into the request itself (the only
        # thing that bounds a hung call). Async clients are cancellable, so skip.
        if (
            self.llm_request_timeout is not None
            and "timeout" not in llm_kwargs
            and not inspect.iscoroutinefunction(client.chat)
            and accepts_kwarg(client.chat, "timeout")
        ):
            llm_kwargs["timeout"] = self.llm_request_timeout
        call = self.pool.call(client.chat, messages, **llm_kwargs)
        if self.llm_request_timeout is not None:
            call = asyncio.wait_for(call, self.llm_request_timeout)
        reply = await call
        return reply, usage_from_client(client)

    def llm_client(self, model: str = "default"):
        try:
            return self._llm_clients[model]
        except KeyError as exc:
            keys = ", ".join(sorted(self._llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}") from exc

    @tool(
        "Run a list of independent one-shot prompts as concurrent model calls; "
        "blocking calls are bounded by Flow's worker pool. Returns a list of "
        "results.",
        proxy=True,
    )
    async def llm_query_batched(
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
            isinstance(prompt, str) for prompt in prompts
        ):
            raise TypeError("llm_query_batched(prompts) takes a list[str]")

        schema = json_schema_for(output_schema) if output_schema is not None else None
        sent = prompts
        if schema is not None:
            hint = self.output_parser.system_prompt_hint(schema)
            sent = [
                f"{prompt}\n\nReturn JSON matching this schema:\n{hint}"
                for prompt in prompts
            ]

        llm_kwargs = sampling_kwargs(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop
        )

        async def call(prompt: str) -> str:
            text, _usage = await self._call_chat(
                [{"role": "user", "content": prompt}], model, **llm_kwargs
            )
            return text

        texts = await asyncio.gather(*(call(prompt) for prompt in sent))
        if schema is not None:
            return [self.output_parser(text, schema) for text in texts]
        return texts

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        """Register a tool after construction and push it into every live REPL.

        The tool is callable on the next turn of already-running agents and is
        included when new agents seed their REPLs. The per-turn system prompt
        reflects it automatically (``@tool``-decorated tools render a doc line).
        This is the hook that lets a tool create more tools at runtime.
        """
        key = name or tool_name(fn)
        if key in _RESERVED_TOOL_NAMES:
            raise ValueError(f"{key!r} is reserved and cannot be overridden")
        self.tools[key] = fn
        for repl in self.repls.values():
            repl.inject(key, fn)

    def remove_tool(self, name: str) -> Any:
        """Unregister a tool and drop it from every live REPL. Returns it (or None)."""
        if name in _RESERVED_TOOL_NAMES:
            raise ValueError(f"{name!r} is reserved and cannot be removed")
        fn = self.tools.pop(name, None)
        for repl in self.repls.values():
            repl.remove_tool(name)
        return fn

    def tool_namespace_for_prompt(self, graph: Graph) -> dict[str, Any]:
        repl = self.repls.get(repl_key(graph))
        return repl.namespace if repl is not None else self.build_tools(graph)

    def build_tools(self, graph: Graph, repl: ReplLike | None = None) -> dict[str, Any]:
        repl = repl or SimpleNamespace(done_result=None, drain=lambda: "")
        tools = {
            "done": make_done(self, graph, repl),
            "launch_subagents": self.launch_subagents(graph, repl),
            "INPUTS": dict(graph.inputs),
        }
        if self.use_llm_query:
            tools["llm_query_batched"] = self.llm_query_batched
        for namespace in (self.runtime.tools, self.tools):
            for name, fn in namespace.items():
                tools.setdefault(name, fn)
        return tools


__all__ = ["Flow", "LLMUsage", "Run", "code_block"]
