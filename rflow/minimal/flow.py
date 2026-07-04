"""Minimal graph-first Flow with an async graph event stream."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Literal

from rflow.minimal.code import find_code_blocks
from rflow.minimal.events import (
    AddChild,
    AppendNode,
    Event,
    GraphAction,
    GraphCreated,
)
from rflow.minimal.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    LLMUsage,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    apply_graph_action,
)
from rflow.minimal.pool import AsyncPool, SequentialPool
from rflow.minimal.prompts import DEFAULT_BUILDER, PromptBuilder
from rflow.minimal.repl import DoneSignal
from rflow.minimal.runtime import LocalRuntime, ReplLike, Runtime
from rflow.minimal.structured import Schema, StructuredOutputParser, json_schema_for
from rflow.minimal.tools import get_tool_metadata, tool

ReplKey = tuple[str, str]
Pool = AsyncPool | SequentialPool
StepUntil = (
    Literal["event", "node", "done", "finished", "idle", "supervising", "error"]
    | Callable[[Event, Graph], bool]
    | int
)
GraphInput = Graph | str | None


def code_block(text: str) -> str:
    blocks = find_code_blocks(text)
    return blocks[0] if blocks else ""


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
        system_prompt: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        show_vars: bool = False,
        tools: list[Any] | None = None,
        runtime: Runtime | None = None,
        llm_clients: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth
        self.max_iters = max_iters
        self.system_prompt = system_prompt
        self.prompt_builder = prompt_builder or DEFAULT_BUILDER
        self.show_vars = show_vars
        self.tools = {self._tool_name(fn): fn for fn in tools or []}
        self.runtime = runtime or LocalRuntime()
        self.pool = pool or AsyncPool(max_concurrency=max_concurrency)
        self.use_llm_query = use_llm_query
        self.enable_structured_output = True
        self.output_parser = StructuredOutputParser()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.repls: dict[ReplKey, ReplLike] = {}
        self._events: asyncio.Queue[Event] | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._run_graph: Graph | None = None

    def start(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> Graph:
        graph = self._graph_from_input(graph_or_query, inputs, output_schema)
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
        graph = self._graph_from_input(graph_or_query, inputs, output_schema)
        async for _event in self.run_stream(graph):
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

        def reached(event: Event, events: list[Event]) -> bool:
            if isinstance(until, int):
                return len(events) >= until
            if callable(until):
                return bool(until(event, graph)) or (
                    n is not None and len(events) >= n
                )
            if until == "event":
                return len(events) >= (n or 1)
            if until == "node":
                return sum(e.type == "append_node" for e in events) >= (n or 1)
            if until in {"done", "finished", "idle"}:
                return event.type == "append_node" and event.node_type == "done_output"
            if until == "supervising":
                return (
                    event.type == "append_node"
                    and event.node_type == "supervising_output"
                )
            if until == "error":
                return (
                    event.type == "append_node" and event.node_type == "error_output"
                )
            raise ValueError(f"unknown step boundary: {until!r}")

        events: list[Event] = []
        while True:
            event = await self._next_event()
            if event is None:
                break
            events.append(event)
            if reached(event, events):
                break
            if self._run_task is None:
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
            task = self._clear_run()
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    def run_stream(
        self,
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        *,
        output_schema: Schema | None = None,
    ) -> AsyncIterator[Event]:
        return self.run_streaming(graph_or_query, inputs, output_schema=output_schema)

    async def _ensure_run(
        self,
        graph_or_query: GraphInput,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
    ) -> Graph:
        if self._run_task is not None:
            if graph_or_query is not None:
                raise RuntimeError("a run is already active")
            if self._run_graph is None:
                raise RuntimeError("active run is missing its graph")
            return self._run_graph

        self._events = asyncio.Queue()
        if graph_or_query is None:
            if self._run_graph is None:
                self._events = None
                raise ValueError("first run needs a graph or query")
            graph = self._run_graph
        else:
            graph = self._graph_from_input(graph_or_query, inputs, output_schema)
            if isinstance(graph_or_query, str):
                self.apply_action(graph, GraphCreated(type="graph_created", graph=graph))
            self._ensure_user_query(graph)
        if graph.finished:
            self._events = None
            return graph
        self._run_graph = graph
        self._run_task = asyncio.create_task(self._run_agent(graph))
        return graph

    def _ensure_user_query(self, graph: Graph) -> None:
        if graph.nodes:
            return
        self.append_node(graph, UserQuery(content=graph.query))

    async def _next_event(self) -> Event | None:
        if self._run_task is None or self._events is None:
            return None
        if self._run_task.done() and self._events.empty():
            await self._run_task
            self._clear_run()
            return None
        return await self._events.get()

    def _clear_run(self) -> asyncio.Task[None] | None:
        task = self._run_task
        self._run_task = None
        self._events = None
        self._run_graph = None
        return task

    async def _run_agent(self, graph: Graph) -> None:
        for _ in range(self.max_iters):
            if graph.finished:
                return

            # Provider turn: model output becomes a graph node/event.
            reply, usage = await self.pool.run(
                self._call_chat(self.messages(graph), graph.model)
            )
            code = code_block(reply)
            self.append_node(
                graph,
                LLMOutput(
                    content=reply,
                    code=code,
                    metadata={
                        "model": graph.model,
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                        },
                    },
                ),
            )

            # Action turn: even invalid/missing code is recorded as the attempted action.
            self.append_node(graph, ExecAction(code=code))

            # Observation turn: running the action produces exactly one result node.
            repl = self.repl_for(graph)
            repl.done_result = None
            try:
                out = await repl.run(code)
            except Exception as exc:  # noqa: BLE001
                self.close_repl(graph)
                out = f"REPL execution failed: {type(exc).__name__}: {exc}"
                self.append_node(graph, ErrorOutput(content=out, output=out, error="exec"))
                continue

            if repl.done_result is not None:
                self.append_node(
                    graph,
                    DoneOutput(content=out, output=out, result=repl.done_result),
                )
                return
            node = (
                ErrorOutput(content=out, output=out, error="exec")
                if repl.errored
                else ExecOutput(content=out or "(no output)", output=out)
            )
            self.append_node(graph, node)
        self.append_node(graph, DoneOutput(result="[max_iters exceeded]"))

    def repl_for(self, graph: Graph) -> ReplLike:
        key = self._repl_key(graph)
        repl = self.repls.get(key)
        if repl is None:
            repl = self.runtime.open(graph)
            repl.seed(self.build_tools(graph, repl), graph.inputs)
            self.repls[key] = repl
        return repl

    def close_repl(self, graph: Graph) -> None:
        repl = self.repls.pop(self._repl_key(graph), None)
        if repl is not None:
            with suppress(Exception):
                repl.close()

    def close_repls(self, graph_id: str | None = None) -> None:
        keys = [key for key in self.repls if graph_id is None or key[0] == graph_id]
        for key in keys:
            repl = self.repls.pop(key)
            with suppress(Exception):
                repl.close()

    def _done(self, graph: Graph, repl: ReplLike):
        def done(answer: object) -> None:
            if graph.output_schema is not None:
                content = answer if isinstance(answer, str) else json.dumps(answer)
                self.output_parser(content, graph.output_schema)
                repl.done_result = content
            else:
                repl.done_result = str(answer)
            print(f"[done] {repl.done_result}")
            raise DoneSignal()

        return done

    def launch_subagents(self, parent: Graph, repl: ReplLike):
        async def launch_subagents(specs: list[dict[str, Any]]) -> list[Any]:
            results: list[Any] = []
            child_ids: list[str] = []
            for index, spec in enumerate(specs):
                name = spec.get("name", f"child{index}")
                if parent.depth >= self.max_depth:
                    results.append(f"[refused: max depth {self.max_depth}]")
                    continue
                child_id = f"{parent.agent_id}.{name}"
                child = Graph(
                    agent_id=child_id,
                    graph_id=parent.graph_id,
                    query=spec["query"],
                    inputs=dict(spec.get("inputs") or {}),
                    model=spec.get("model", "default"),
                    output_schema=(
                        json_schema_for(spec["output_schema"])
                        if spec.get("output_schema") is not None
                        else None
                    ),
                    depth=parent.depth + 1,
                    parent_agent_id=parent.agent_id,
                )
                self.apply_action(
                    parent,
                    AddChild(
                        type="add_child",
                        parent_agent_id=parent.agent_id,
                        child=child,
                    )
                )
                self.append_node(child, UserQuery(content=spec["query"]))
                child_ids.append(child_id)
                results.append("")

            self.append_node(
                parent,
                SupervisingOutput(output=repl.drain(), waiting_on=child_ids),
            )
            await self.pool.gather(*(self._run_agent(parent[cid]) for cid in child_ids))
            self.append_node(parent, ResumeAction(resumed_from=child_ids))
            for i, child_id in enumerate(child_ids):
                child = parent[child_id]
                result = child.result()
                results[i] = (
                    self.output_parser(result, child.output_schema)
                    if child.output_schema is not None
                    else result
                )
            return results

        return launch_subagents

    def append_node(self, graph: Graph, node) -> Event:
        return self.apply_action(
            graph,
            AppendNode(
                type="append_node",
                agent_id=graph.agent_id,
                node_type=node.type,
                node=node,
            )
        )

    def apply_action(self, graph: Graph, action: GraphAction) -> GraphAction:
        base = None if isinstance(action, GraphCreated) else graph
        apply_graph_action(base, action)
        if self._events is not None:
            self._events.put_nowait(action)
        return action

    def messages(self, graph: Graph) -> list[dict[str, str]]:
        msgs = [
            {
                "role": "system",
                "content": self.build_system_prompt(graph),
            }
        ]
        for node in graph.nodes:
            if isinstance(node, UserQuery):
                msgs.append({"role": "user", "content": node.content})
            elif isinstance(node, LLMOutput):
                msgs.append({"role": "assistant", "content": node.content})
            elif isinstance(node, (ExecOutput, ErrorOutput, SupervisingOutput)):
                msgs.append({"role": "user", "content": node.content or node.output})
        return msgs

    def build_system_prompt(self, graph: Graph) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        return self.prompt_builder.build(self, graph)

    async def chat(self, graph: Graph) -> str:
        reply, _usage = await self.pool.run(
            self._call_chat(self.messages(graph), graph.model)
        )
        return reply

    async def _call_chat(
        self, messages: list[dict[str, str]], model: str = "default"
    ) -> tuple[str, LLMUsage]:
        client = self.llm_client(model)
        chat = client.chat
        if inspect.iscoroutinefunction(chat):
            reply = await chat(messages)
            return reply, self._usage_from_client(client)
        reply = await asyncio.to_thread(chat, messages)
        if inspect.isawaitable(reply):
            reply = await reply
        return reply, self._usage_from_client(client)

    def llm_client(self, model: str = "default"):
        try:
            return self._llm_clients[model]
        except KeyError as exc:
            keys = ", ".join(sorted(self._llm_clients))
            raise ValueError(f"unknown model {model!r}. available: {keys}") from exc

    def _usage_from_client(self, client: Any) -> LLMUsage:
        usage = getattr(client, "last_usage", None)
        if usage is None:
            return LLMUsage()
        return LLMUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    @tool(
        "Await with a list of independent one-shot prompts to run model calls "
        "through Flow's shared pool.",
        proxy=True,
    )
    async def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: Schema | None = None,
    ) -> list:
        if not isinstance(prompts, list) or not all(
            isinstance(prompt, str) for prompt in prompts
        ):
            raise TypeError("llm_query_batched(prompts) takes a list[str]")

        schema = json_schema_for(output_schema) if output_schema is not None else None
        sent = prompts
        if schema is not None:
            hint = self.output_parser.system_prompt_hint(schema)
            sent = [f"{prompt}\n\nReturn JSON matching this schema:\n{hint}" for prompt in prompts]

        async def call(prompt: str) -> str:
            text, _usage = await self._call_chat(
                [{"role": "user", "content": prompt}], model
            )
            return text

        texts = await self.pool.gather(*(call(prompt) for prompt in sent))
        if schema is not None:
            return [self.output_parser(text, schema) for text in texts]
        return texts

    def tool_namespace_for_prompt(self, graph: Graph) -> dict[str, Any]:
        repl = self.repls.get(self._repl_key(graph))
        if repl is not None:
            return repl.namespace
        return self.build_tools(graph)

    def build_tools(self, graph: Graph, repl: ReplLike | None = None) -> dict[str, Any]:
        repl = repl or SimpleNamespace(done_result=None, drain=lambda: "")
        tools = {
            "done": self._done(graph, repl),
            "launch_subagents": self.launch_subagents(graph, repl),
            "INPUTS": dict(graph.inputs),
        }
        if self.use_llm_query:
            tools["llm_query_batched"] = self.llm_query_batched
        for namespace in (self.runtime.tools, self.tools):
            for name, fn in namespace.items():
                tools.setdefault(name, fn)
        return tools

    @staticmethod
    def _tool_name(fn: Any) -> str:
        meta = get_tool_metadata(fn)
        return meta.name if meta is not None else fn.__name__

    @staticmethod
    def _repl_key(graph: Graph) -> ReplKey:
        return (graph.graph_id, graph.agent_id)

    @staticmethod
    def _graph_from_input(
        graph_or_query: Graph | str,
        inputs: dict[str, str] | None = None,
        output_schema: Schema | None = None,
    ) -> Graph:
        if isinstance(graph_or_query, Graph):
            if inputs is not None:
                graph_or_query.inputs = dict(inputs)
            if output_schema is not None:
                graph_or_query.output_schema = json_schema_for(output_schema)
            return graph_or_query
        return Graph(
            query=graph_or_query,
            inputs=dict(inputs or {}),
            output_schema=json_schema_for(output_schema) if output_schema else None,
        )


__all__ = ["Flow", "LLMUsage", "code_block"]
