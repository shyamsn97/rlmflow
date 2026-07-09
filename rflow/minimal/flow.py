"""Minimal graph-first Flow with an async graph event stream."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from rflow.minimal.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    LLMUsage,
    Node,
    UserQuery,
    apply_graph_action,
)
from rflow.minimal.graph.events import (
    AppendNode,
    Event,
    EventStream,
    GraphAction,
    GraphCreated,
    StepUntil,
    reached,
)
from rflow.minimal.pool import AsyncPool, Pool
from rflow.minimal.prompts import DEFAULT_BUILDER, PromptBuilder
from rflow.minimal.prompts.messages import build_messages, merge_summary
from rflow.minimal.runtime import LocalRuntime, Runtime
from rflow.minimal.runtime.env import agent_process_env
from rflow.minimal.runtime.repl import ReplLike
from rflow.minimal.structured import Schema, StructuredOutputParser, json_schema_for
from rflow.minimal.tools import tool
from rflow.minimal.tools.builtins import make_done, make_launch_subagents
from rflow.minimal.utils import (
    ReplKey,
    budget_exceeded,
    call_sync_or_async,
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

GraphInput = Graph | str | None

#: Default cap on a subagent ``query`` length: keep queries short instructions and
#: push bulk payloads into ``inputs`` instead.
DEFAULT_MAX_QUERY_CHARS = 2_000

#: Names owned by the harness/graph; ``add_tool``/``remove_tool`` won't touch them.
_RESERVED_TOOL_NAMES = frozenset({"done", "launch_subagents", "INPUTS"})

#: Sentinel results recorded when a run ends without an agent-produced answer.
TERMINATED = "[terminated]"
BUDGET_EXCEEDED = "[budget exceeded]"
MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


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
        max_concurrency: int | None = None,
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
        self.pool = pool or AsyncPool(max_concurrency=max_concurrency)
        self.use_llm_query = use_llm_query
        self.enable_structured_output = enable_structured_output
        self.output_parser = StructuredOutputParser()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.repls: dict[ReplKey, ReplLike] = {}
        self._terminate_requested: set[str] = set()
        self._run_graph: Graph | None = None
        self._stream: EventStream | None = None

    def start(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> Graph:
        graph = graph_from_input(graph_or_query, inputs, output_schema)
        self._ensure_user_query(graph)
        return graph

    def run(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> str:
        return asyncio.run(
            self.arun(graph_or_query, inputs, output_schema=output_schema)
        )

    async def arun(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> str:
        graph = graph_from_input(graph_or_query, inputs, output_schema)
        async for _event in self.run_streaming(graph):
            pass
        return graph.result()

    async def step(
        self,
        graph_or_query: GraphInput = None,
        *,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
        until: StepUntil = "node",
        n: int | None = None,
    ) -> list[Event]:
        if n is not None and n < 1:
            raise ValueError("n must be >= 1")
        graph = await self._ensure_run(graph_or_query, inputs, output_schema)
        events: list[Event] = []
        while (event := await self._next_event()) is not None:
            events.append(event)
            if reached(until, n, event, events, graph) or self._stream is None:
                break
        return events

    async def run_streaming(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> AsyncIterator[Event]:
        await self._ensure_run(graph_or_query, inputs, output_schema)
        try:
            while True:
                event = await self._next_event()
                if event is None:
                    break
                yield event
        finally:
            if self._stream is not None:
                await self._stream.aclose()
            self._clear_run()

    #: Alias for :meth:`run_streaming` (same async generator, shorter name).
    run_stream = run_streaming

    async def _ensure_run(
        self,
        graph_or_query: GraphInput,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
    ) -> Graph:
        if self._stream is not None:
            if graph_or_query is not None:
                raise RuntimeError("a run is already active")
            if self._run_graph is None:
                raise RuntimeError("active run is missing its graph")
            return self._run_graph

        stream = EventStream()
        self._stream = stream
        if graph_or_query is None:
            if self._run_graph is None:
                self._stream = None
                raise ValueError("first run needs a graph or query")
            graph = self._run_graph
        else:
            graph = graph_from_input(graph_or_query, inputs, output_schema)
            if isinstance(graph_or_query, str):
                self.apply_action(
                    graph, GraphCreated(type="graph_created", graph=graph)
                )
            self._ensure_user_query(graph)
        if graph.finished:
            self._stream = None
            return graph
        self._run_graph = graph
        stream.start(self.run_agent(graph))
        return graph

    def _ensure_user_query(self, graph: Graph) -> None:
        if graph.nodes:
            return
        self.append_node(graph, UserQuery(content=graph.query))

    async def _next_event(self) -> Event | None:
        if self._stream is None:
            return None
        event = await self._stream.next()
        if event is None:
            self._clear_run()
        return event

    def _clear_run(self) -> None:
        self._stream = None
        self._run_graph = None

    async def run_agent(self, graph: Graph) -> None:
        max_iters = iter_budget(graph.depth, self.max_iters, self.child_max_iters)
        for i in range(max_iters):
            if graph.finished:
                return
            if graph.agent_id in self._terminate_requested:
                self.append_node(graph, DoneOutput(result=TERMINATED))
                return
            if budget_exceeded(self._run_graph or graph, self.max_budget):
                self.append_node(graph, DoneOutput(result=BUDGET_EXCEEDED))
                return
            # Last iteration: force a final done(...) instead of silently quitting.
            code = await self.llm_turn(graph, force_final=i == max_iters - 1)
            node = await self.exec_turn(graph, code)
            if isinstance(node, DoneOutput):
                return
        if not graph.finished:
            self.append_node(graph, DoneOutput(result=MAX_ITERS_EXCEEDED))

    def terminate(self, agent_ids: Iterable[str] | None = None) -> None:
        """Cooperatively stop agents (default: all running): each finishes its
        current turn, then records ``done("[terminated]")`` instead of iterating on.
        """
        if agent_ids is None:
            agent_ids = self._run_graph.agents if self._run_graph is not None else ()
        self._terminate_requested.update(agent_ids)

    async def llm_turn(self, graph: Graph, *, force_final: bool = False) -> str:
        """Provider + action turns: model output and the attempted action."""
        reply, usage = await self._pooled_chat(graph, force_final=force_final)
        code = code_block(reply)
        metadata = llm_output_metadata(graph.model, usage)
        self.append_node(graph, LLMOutput(content=reply, code=code, metadata=metadata))
        # Even invalid/missing code is recorded as the attempted action.
        self.append_node(graph, ExecAction(code=code))
        return code

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
            repl.set_process_env(
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

    async def rebuild_repl(self, graph: Graph, *, agent_id: str | None = None) -> ReplLike:
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
    ) -> Graph:
        """Independent branch of ``graph`` (deepcopy, fresh ``graph_id``) with its
        REPL rebuilt to match the optionally-rewound trajectory. Never mutates parent.
        """
        child = graph.fork(from_node_id=from_node_id, agent_id=agent_id)
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
        delta = c.nodes[common_prefix_len(p.nodes, c.nodes):]

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

    def append_node(self, graph: Graph, node) -> Event:
        action = AppendNode(
            type="append_node", agent_id=graph.agent_id, node_type=node.type, node=node
        )
        return self.apply_action(graph, action)

    def apply_action(self, graph: Graph, action: GraphAction) -> GraphAction:
        base = None if isinstance(action, GraphCreated) else graph
        apply_graph_action(base, action)
        if self._stream is not None:
            # Stamp originating graph_id so merged streams (FlowGroup) self-describe.
            event = (
                action
                if isinstance(action, GraphCreated)
                else replace(action, graph_id=graph.graph_id)
            )
            self._stream.emit(event)
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

    async def chat(self, graph: Graph) -> str:
        reply, _usage = await self._pooled_chat(graph)
        return reply

    async def _pooled_chat(
        self, graph: Graph, *, force_final: bool = False
    ) -> tuple[str, LLMUsage]:
        messages = self.messages(graph, force_final=force_final)
        return await self.pool.run(self._call_chat(messages, graph.model))

    async def _call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        **llm_kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        client = self.llm_client(model)
        call = call_sync_or_async(client.chat, messages, **llm_kwargs)
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
        "Run a list of independent one-shot prompts as concurrent model calls "
        "through Flow's shared pool; returns a list of results.",
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

        texts = await self.pool.gather(*(call(prompt) for prompt in sent))
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
            repl.inject_tool(key, fn)

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


__all__ = ["Flow", "LLMUsage", "code_block"]
