"""Minimal graph-first Flow with an async graph event stream."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Literal

from rlmflow.graph import (
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
    system_prompt_id,
)
from rlmflow.graph.events import (
    AddChild,
    Event,
    GraphAction,
    GraphCreated,
    StepUntil,
    is_idle,
)
from rlmflow.pool import Pool, ThreadPool
from rlmflow.prompts import (
    DEFAULT_BUILDER,
    PromptProfile,
    SystemPromptSource,
    as_prompt_profile,
    as_system_prompt_fn,
)
from rlmflow.prompts.messages import (
    CONTINUE_NUDGE,
    FINAL_ANSWER_ACTION,
    TRUNCATION_SUMMARY,
    UserPromptBuilder,
    UserPromptSource,
    as_user_prompt,
    coalesce_roles,
    merge_summary,
)
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.env import agent_process_env
from rlmflow.runtime.repl import Repl
from rlmflow.structured import Schema, StructuredOutputParser, json_schema_for
from rlmflow.tasks import TaskQueue
from rlmflow.tools import tool
from rlmflow.tools.builtins import make_done, make_launch_subagents
from rlmflow.utils import (
    ReplKey,
    accepts_kwarg,
    budget_exceeded,
    code_block,
    common_prefix_len,
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

#: Trajectory note appended to a ``fork(mode="lazy")`` branch: its REPL was not
#: rebuilt, so runtime-created state from earlier turns is unavailable.
LAZY_FORK_NOTE = (
    "This branch was forked from the parent trajectory WITHOUT restoring the "
    "parent's live REPL. Variables, functions, and tools you created at runtime "
    "in earlier turns are NOT available here (registered tools and INPUTS are "
    "still seeded). Re-create anything you need before relying on it."
)

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

    #: Chat-message wording owned by ``messages`` (override on a subclass to
    #: change it): the forced-final action, the continue nudge, and the marker
    #: inserted when the trajectory is truncated to fit ``max_messages``.
    final_action: str = FINAL_ANSWER_ACTION
    continue_nudge: str = CONTINUE_NUDGE
    truncation_summary: str = TRUNCATION_SUMMARY

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
        system_prompt: SystemPromptSource | None = None,
        user_prompt: UserPromptSource | None = None,
        prompts: dict[str, Any] | None = None,
        prompt_router: Literal["llm", "graph"] | Callable[[Flow, Graph], str] = "llm",
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
        # A system prompt is any source resolvable by ``as_system_prompt_fn`` — a
        # string, a ``(flow, graph) -> str`` function, or a ``SystemPromptBuilder``.
        # The user side is a ``UserPromptBuilder`` or a ``(flow, graph) -> turns``
        # function. Both are public/settable and resolved fresh each turn.
        self.system_prompt: SystemPromptSource = (
            system_prompt if system_prompt is not None else DEFAULT_BUILDER
        )
        self.user_prompt: UserPromptBuilder = (
            as_user_prompt(user_prompt)
            if user_prompt is not None
            else UserPromptBuilder()
        )
        # Named alternate prompt profiles for children, parallel to ``llm_clients``.
        # ``"default"`` is implicit (the flow's own ``system_prompt``/``user_prompt``)
        # and reserved; only alternates live here, normalized to ``PromptProfile``.
        if prompts and "default" in prompts:
            raise ValueError(
                "'default' is reserved; it maps to the flow's system_prompt/"
                "user_prompt. Register alternate profiles under other names."
            )
        self._prompts: dict[str, PromptProfile] = {
            name: as_prompt_profile(src) for name, src in (prompts or {}).items()
        }
        #: How profile names are chosen. ``"llm"`` (default): advertise registered
        #: profiles and honor ``graph.prompt_profile`` (set by a spawn spec or the
        #: host). ``"graph"``: honor ``graph.prompt_profile`` but do not advertise
        #: (host stamped the name). A ``(flow, graph) -> str`` callable is host
        #: policy that overrides ``graph.prompt_profile`` and suppresses advertising.
        if prompt_router not in ("llm", "graph") and not callable(prompt_router):
            raise TypeError(
                "prompt_router must be 'llm', 'graph', or a "
                f"(flow, graph) -> str callable; got {prompt_router!r}"
            )
        self.prompt_router = prompt_router
        # One flat map of REPL bindings (see ``inject``/``add_tool``). Whether an
        # entry is advertised in the system prompt is a property of the object —
        # only ``@tool``-decorated callables render a doc line — not of any
        # separate store, so tools and plain injected globals live together here.
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
        self.repls: dict[ReplKey, Repl] = {}
        self._terminate_requested: set[str] = set()
        #: Active runs keyed by ``graph_id`` (one per trajectory). Created by
        #: ``run_for`` and cleared by ``finish_run``; mirrors how ``repls`` keys.
        self.runs: dict[str, Run] = {}
        #: Generic extension seams other layers build on without subclassing.
        #: ``halts`` maps a name to a ``(event, graph) -> bool`` predicate that
        #: ``run_streaming(until=...)`` can resolve (see ``tools.graph_ops``).
        #: ``tool_factories`` are ``(flow, graph, repl) -> {name: obj}`` callables
        #: run per agent in ``build_tools`` so injected tools can close over the
        #: agent's own graph/REPL (the general case of a plain ``tools`` entry).
        self.halts: dict[str, Callable[..., bool]] = {}
        self.tool_factories: list[Callable[[Flow, Graph, Repl], dict[str, Any]]] = []

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
        async for _event in self.drive(run, until="done"):
            pass
        return run.graph.result()

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
        # Phase 1: resolve/register the run. Phase 2: drive it (see :meth:`drive`).
        run = self.resolve_run(
            graph=graph,
            query=query,
            inputs=inputs,
            output_schema=output_schema,
            merge_inputs=merge_inputs,
        )
        async for event in self.drive(run, until=until, n=n, close_repls=close_repls):
            yield event

    async def drive(
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
            self._sync_repl_inputs(graph)
        if output_schema is not None:
            graph.output_schema = json_schema_for(output_schema)
        run = self.run_for(graph)
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
        # A named halt registered on the flow (see ``tools.graph_ops.register_halt``)
        # resolves to its predicate here, so callers can pass the name as ``until``.
        predicate = self.halts.get(until)
        if predicate is not None:
            return not bool(predicate(event, run.graph))
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
        if llm_turns >= self.max_iters_for(graph):
            self.append_node(graph, DoneOutput(result=MAX_ITERS_EXCEEDED))
            return

        # ``messages`` runs the user builder, which prepares the graph for this
        # turn (commits any per-turn content and the trailing nudge as real
        # nodes) before projecting, so the transcript stays self-contained.
        messages = self.messages(graph)
        # The assembled list always leads with the system message; reuse it for
        # the metadata snapshot rather than rebuilding the prompt a second time.
        system_prompt = messages[0]["content"]
        reply, usage = await self._call_chat(messages, graph.model)
        code = code_block(reply)
        metadata = llm_output_metadata(graph.model, usage)
        # Record everything needed to reconstruct this exact call: a reference to
        # the system prompt used (stored once in the graph's content-addressed
        # table) and the context-window budget in force (so the truncated view the
        # model actually saw is reproducible).
        sid = system_prompt_id(system_prompt)
        graph.system_prompts.setdefault(sid, system_prompt)
        metadata["system_prompt"] = sid
        if self.max_messages is not None:
            metadata["max_messages"] = self.max_messages
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

    def repl_for(self, graph: Graph) -> Repl:
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

    def open_repls(self) -> Iterator[tuple[Graph, Repl]]:
        """Yield ``(graph, repl)`` for every live-run agent whose REPL is open.

        The pairing ``build_tools`` can't give you after the fact: it walks the
        active runs (root plus spawned children) and matches each agent to its
        materialized REPL. Used to backfill tools onto already-open REPLs when a
        factory is registered mid-flight (see ``tools.graph_ops.inject_tools``).
        """
        for run in self.runs.values():
            for agent in run.graph.walk():
                repl = self.repls.get(repl_key(agent))
                if repl is not None:
                    yield agent, repl

    def get_var(self, graph: Graph, name: str, *, agent_id: str | None = None) -> Any:
        """Read a variable out of an agent's REPL Python namespace.

        The inverse of injecting an object: pull a (possibly LLM-mutated) value
        back onto the host. In-process runtimes return the live object itself;
        process-isolated runtimes return a cloudpickle **copy** of the sandbox's
        object (host mutations and sandbox mutations do not alias). Raises if the
        name is unbound. For the per-REPL ``ENV`` metadata channel, use
        :meth:`get_env_var`.
        """
        agent = graph.agent_for(agent_id) if agent_id is not None else graph
        return self.repl_for(agent).get_var(name)

    def get_env_var(
        self, graph: Graph, name: str, *, agent_id: str | None = None
    ) -> Any:
        """Read a key from an agent's REPL ``env`` metadata channel (``ENV``).

        Distinct from :meth:`get_var`, which reads the Python namespace. The host
        seeds ``RLMFLOW_*`` keys here; agent code can also write via ``ENV[...]``.
        Raises ``KeyError`` if ``name`` is unset.
        """
        agent = graph.agent_for(agent_id) if agent_id is not None else graph
        return self.repl_for(agent).get_env_var(name)

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

    async def _replay_exec_actions(self, graph: Graph, nodes: list[Node]) -> None:
        """Re-run each ``ExecAction`` in ``nodes`` against ``graph``'s REPL as a
        replay (no appended nodes, no LLM). Shared by ``rebuild_repl`` (the whole
        trajectory) and ``_merge_repl_state`` (just the merged delta)."""
        for node in nodes:
            if isinstance(node, ExecAction) and node.code:
                await self.exec_turn(graph, node.code, replay=True)

    async def rebuild_repl(self, graph: Graph, *, agent_id: str | None = None) -> Repl:
        """Reconstruct an agent's REPL by replaying its ``ExecAction`` code blocks
        (no LLM calls, no appended nodes). The correctness floor for forks/reverts.
        """
        agent = graph.agent_for(agent_id)
        self.close_repl(agent)
        await self._replay_exec_actions(agent, agent.nodes)
        return self.repl_for(agent)

    async def fork(
        self,
        graph: Graph,
        *,
        from_node_id: str | None = None,
        agent_id: str | None = None,
        keep_anchor: bool = False,
        mode: Literal["replay", "lazy"] = "replay",
    ) -> Graph:
        """Independent branch of ``graph`` (deepcopy, fresh ``graph_id``). Never
        mutates the parent.

        By default ``from_node_id`` is dropped (branch ends before it, to
        re-decide); pass ``keep_anchor=True`` to retain it and continue from
        after it.

        ``mode`` controls how the branch's REPL is materialized:

        - ``"replay"`` (default) — rebuild the REPL by re-running the prefix's
          ``ExecAction`` code (deterministic, no LLM calls), so variables/tools
          created at runtime are present and the branch continues exactly where
          the trajectory left off. The correctness floor; works on any runtime.
        - ``"lazy"`` — skip the replay: copy the trajectory only and append a
          note that prior REPL state was NOT restored. The branch gets a fresh
          REPL on its next turn (registered tools + ``INPUTS`` are seeded, but
          runtime-created variables/tools are gone). Cheap — use when you only
          need the branch's history, or when replaying the prefix is expensive
          or not reproducible.
        """
        if mode not in ("replay", "lazy"):
            raise ValueError("mode must be 'replay' or 'lazy'")
        child = graph.fork(
            from_node_id=from_node_id, agent_id=agent_id, keep_anchor=keep_anchor
        )
        if mode == "replay":
            await self.rebuild_repl(child, agent_id=agent_id)
        else:
            # Lazy: no REPL reconstruction. Annotate the branch so its next turn
            # knows the prior warm state is gone (a fresh REPL is opened on first
            # use by ``repl_for``, seeded only with tools + INPUTS).
            child.inject(
                ExecOutput(content=LAZY_FORK_NOTE, output=LAZY_FORK_NOTE),
                agent_id=agent_id,
            )
        return child

    async def rewind(
        self,
        graph: Graph,
        *,
        n: int = 1,
        where: Callable[[LLMOutput], bool] | None = None,
        agent_id: str | None = None,
        mode: Literal["replay", "lazy"] = "replay",
    ) -> Graph:
        """Fork ``graph`` rewound by ``n`` decision turns, returning the new branch.

        Sugar over :meth:`fork` that counts decision turns (each ``LLMOutput`` is
        one) from the end instead of taking a ``from_node_id``: the branch drops
        the last ``n`` turns and re-decides from there. Non-destructive — the
        original is untouched and the branch REPL is rebuilt by replay.

        Pass ``where`` to count only the turns that matter — the branch anchors on
        the ``n``-th-from-last turn matching it, skipping setup/no-op turns that
        would otherwise be miscounted (e.g. a scripted construct turn or a final
        ``done`` turn that isn't a real move).
        """
        agent = graph.agent_for(agent_id)
        turns = [
            node
            for node in agent.nodes
            if isinstance(node, LLMOutput) and (where is None or where(node))
        ]
        if n < 1:
            raise ValueError("n must be >= 1 (how many decision turns to undo)")
        if n > len(turns):
            raise ValueError(
                f"cannot rewind {n} turns: graph has only {len(turns)} to undo"
            )
        return await self.fork(
            graph, from_node_id=turns[-n].id, agent_id=agent_id, mode=mode
        )

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

        await self._replay_exec_actions(p, delta)

    def discard(self, *children: Graph) -> None:
        """Eagerly close rejected branches' REPLs. Optional (parent was never
        mutated, so GC suffices); only frees remote sandboxes. No-op on LocalRuntime.
        """
        for child in children:
            self.close_repls(child.graph_id)

    def adopt(self, parent: Graph, child: Graph, *, name: str) -> Graph:
        """Reparent a prepared graph (a ``fork``/``rewind`` result) as a child agent
        of ``parent``, returning the attached child.

        Structural only: it remaps the child's ids, moves its already-built REPL to
        the new key, and emits ``AddChild``. It does **not** append a
        ``SupervisingOutput`` and does **not** run/await the child — the launcher
        (``launch_subagents``) owns the supervising node and the await loop. This is
        the warm counterpart to a cold ``Graph(query=...)`` child: the peer of
        ``fork``/``rewind`` that lets a launched child continue an existing
        trajectory instead of cold-starting.

        The child must be a single-agent graph (a plain fork/rewind). Reparenting a
        child that itself carries sub-children is out of scope.
        """
        if child.children:
            raise ValueError(
                "adopt only accepts a single-agent graph (a plain fork/rewind); "
                "reparenting a graph that carries sub-children is out of scope"
            )
        old_key = repl_key(child)  # capture before mutating identity
        new_id = f"{parent.agent_id}.{name}"
        child.set_graph_id(parent.graph_id)
        child.agent_id = new_id
        child.parent_agent_id = parent.agent_id
        child.depth = parent.depth + 1
        for node in child.nodes:
            node.agent_id = new_id
        # ``fork``/``rewind`` already replayed the REPL under the old key; move it to
        # the new key rather than replaying again (rebuild only if it's gone).
        repl = self.repls.pop(old_key, None)
        if repl is not None:
            self.repls[repl_key(child)] = repl
        self.apply_action(
            parent,
            AddChild(type="add_child", parent_agent_id=parent.agent_id, child=child),
        )
        return child

    def launch_subagents(self, parent: Graph, repl: Repl):
        """Public factory for the recursive spawn tool (see ``tools.builtins``)."""
        return make_launch_subagents(self, parent, repl)

    def launch_subgraphs(
        self,
        parent: Graph,
        children: list[Graph],
        *,
        queries: list[str] | None = None,
        names: list[str] | None = None,
    ):
        """Warm counterpart to ``launch_subagents``: run prepared ``Graph``s
        (``fork``/``rewind`` results) as children of ``parent``, in parallel, and
        await their results.

        A thin host-side wrapper over ``launch_subagents``' warm ``graph`` path:
        where ``launch_subagents`` builds children from ``query`` strings (cold),
        this attaches ready graphs (warm). Each child continues its own
        trajectory; pass ``queries`` (one per child) to append a kickoff user
        turn to each, and ``names`` to control the child agent ids (default
        ``c0``, ``c1``, ...). This exists only so host code has a named,
        graph-first entry point — the warm ``graph`` spec key stays an internal
        detail and never has to appear in model-facing tools or example code.
        Returns the awaitable ``launch_subagents`` result (one entry per child).
        """
        launch = self.launch_subagents(parent, self.repl_for(parent))
        specs: list[dict[str, Any]] = []
        for i, child in enumerate(children):
            spec: dict[str, Any] = {
                "graph": child,
                "name": names[i] if names is not None else f"c{i}",
            }
            if queries is not None:
                spec["query"] = queries[i]
            specs.append(spec)
        return launch(specs)

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

    def messages(self, graph: Graph) -> list[dict[str, str]]:
        """Assemble the full chat message list for one LLM call.

        The agent's prompt profile supplies the system + user builders. The user
        builder both *prepares* the graph for this turn (committing any per-turn
        content and the trailing nudge as real ``UserQuery`` nodes) and projects
        it to turns; Flow then prepends the system message, truncates the turns to
        ``max_messages``, and coalesces adjacent same-role turns (required last,
        since injected turns and the truncation marker can introduce consecutive
        user turns). Because it mutates, this must run exactly once per turn (it
        does: the sole call site is ``run_agent_task``). Subclass to own the whole
        list or change truncation.
        """
        profile = self._profile_for(graph)
        system = self.build_system_prompt(graph, profile)
        user = profile.user if profile.user is not None else self.user_prompt
        turns = self._truncate_turns(user(self, graph))
        return coalesce_roles([{"role": "system", "content": system}, *turns])

    def _truncate_turns(self, turns: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.max_messages is None or len(turns) + 1 <= self.max_messages:
            return turns
        keep = max(1, self.max_messages - 2)
        return [{"role": "user", "content": self.truncation_summary}, *turns[-keep:]]

    def max_iters_for(self, graph: Graph) -> int:
        """Effective iteration budget for ``graph`` at its recursion depth."""
        return iter_budget(graph.depth, self.max_iters, self.child_max_iters)

    def build_system_prompt(
        self, graph: Graph, profile: PromptProfile | None = None
    ) -> str:
        profile = profile if profile is not None else self._profile_for(graph)
        system = profile.system if profile.system is not None else self.system_prompt
        return as_system_prompt_fn(system)(self, graph)

    def prompt_name_for(self, graph: Graph) -> str:
        """Which profile name this agent uses.

        ``"llm"`` / ``"graph"`` both honor ``graph.prompt_profile`` (stamp from a
        spawn spec or host construction). A callable ``prompt_router`` is host
        policy and overrides that stamp. Override this method for custom routing.
        """
        router = self.prompt_router
        if callable(router):
            return router(self, graph)
        return graph.prompt_profile

    def prompt_profile(self, name: str = "default") -> PromptProfile:
        """Look up a registered profile by name. ``"default"`` is the implicit
        profile built from the flow's own ``system_prompt``/``user_prompt``."""
        if name == "default":
            return PromptProfile(self.system_prompt, self.user_prompt)
        try:
            return self._prompts[name]
        except KeyError as exc:
            keys = ", ".join(["default", *sorted(self._prompts)])
            raise ValueError(f"unknown prompt {name!r}. available: {keys}") from exc

    def _profile_for(self, graph: Graph) -> PromptProfile:
        # Resolve the profile for this agent with a self-contained-continuation
        # fallback: an unknown name (e.g. a graph loaded into a flow that lacks the
        # profile) yields the last system prompt actually recorded on the graph, so
        # resume keeps the prompt the agent ran under instead of drifting.
        name = self.prompt_name_for(graph)
        try:
            return self.prompt_profile(name)
        except ValueError:
            last = graph.latest_system_prompt()
            if last is not None:
                return PromptProfile(system=last)
            raise

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

    def inject(self, name: str, obj: Any) -> None:
        """Bind ``obj`` into every agent's REPL namespace under ``name``.

        The flow-wide, fork-durable version of ``repl.inject``: ``obj`` is pushed
        into every live REPL now and seeded into every REPL opened later —
        including each fork's rebuilt REPL — so replayed code that references it
        (e.g. ``game = Sokoban(...)``) works on every branch. This is the single
        entry point for everything the agent code touches: tools, classes, data,
        libraries. Whether the entry is *advertised* in the system prompt is a
        property of the object, not of this call — only ``@tool``-decorated
        callables render a doc line, so injecting a plain class or value is
        silently non-advertised. Use :meth:`add_tool` when you want the name
        derived from a function automatically.
        """
        if name in _RESERVED_TOOL_NAMES:
            raise ValueError(f"{name!r} is reserved and cannot be overridden")
        self.tools[name] = obj
        for repl in self.repls.values():
            repl.inject(name, obj)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        """Register a tool after construction, keyed by its own name.

        Thin convenience over :meth:`inject`: derives the binding name from ``fn``
        (``@tool`` name, or ``name=`` override) and injects it. The tool is
        callable on the next turn of already-running agents, seeded into new
        agents' REPLs, and — being ``@tool``-decorated — rendered in the per-turn
        system prompt automatically. This is the hook that lets a tool create
        more tools at runtime.
        """
        self.inject(name or tool_name(fn), fn)

    def remove_tool(self, name: str) -> Any:
        """Unregister a binding and drop it from every live REPL. Returns it (or None)."""
        if name in _RESERVED_TOOL_NAMES:
            raise ValueError(f"{name!r} is reserved and cannot be removed")
        fn = self.tools.pop(name, None)
        for repl in self.repls.values():
            repl.remove_tool(name)
        return fn

    def tool_namespace_for_prompt(self, graph: Graph) -> dict[str, Any]:
        repl = self.repls.get(repl_key(graph))
        return repl.namespace if repl is not None else self.build_tools(graph)

    def build_tools(self, graph: Graph, repl: Repl | None = None) -> dict[str, Any]:
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
        # Per-agent factories run last and via ``setdefault``, so reserved names
        # (``done``/``launch_subagents``/``INPUTS``) and static tools always win.
        for factory in self.tool_factories:
            for name, fn in factory(self, graph, repl).items():
                tools.setdefault(name, fn)
        return tools


__all__ = ["Flow", "LLMUsage", "Run", "code_block"]
