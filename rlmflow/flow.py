"""Run agents: prompt the model, execute its code, delegate to children."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from rlmflow import boundaries
from rlmflow.engine.boundaries import StepUntil
from rlmflow.engine.execution import Pool, TaskQueue, ThreadPool, Transition
from rlmflow.engine.steps import DEFAULT_STEPS, MessageBuilder, StepFunction
from rlmflow.graph.nodes import (
    DEFAULT_QUERY,
    AgentConfig,
    AgentStart,
    DoneOutput,
    ExecAction,
    LLMOutput,
    LLMUsage,
    Node,
    start,
    validate_agent_name,
)
from rlmflow.llm import LLMChunk, PooledLLMClient
from rlmflow.prompts import (
    DEFAULT_BUILDER,
    PromptProfile,
    RenderFn,
    SystemPromptSource,
    as_system_prompt_fn,
    default_render,
)
from rlmflow.runtime import ExecutionGuard, LocalRuntime, Runtime, WrappedRuntime
from rlmflow.runtime.env import RLMFLOW_REPLAY
from rlmflow.runtime.repl import DoneSignal, Repl, ReplRun
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.structured import json_schema_for, parse_structured_answer
from rlmflow.tools import RESERVED_TOOLS, tool
from rlmflow.tools.agents import (
    AGENT_OBSERVE_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    AgentHandle,
    build_agent_directory,
)
from rlmflow.tools.builtins import BuiltIns
from rlmflow.tools.llm_query import llm_query, llm_query_batched
from rlmflow.tools.tools import is_toolset, toolset_members
from rlmflow.utils.helpers import tool_name

BUDGET_EXCEEDED = "[budget exceeded]"


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a list of tools and toolsets.

    A toolset is one item, same as a function. ``tools=[FILE_TOOLS, grep_extra]``
    and ``tools=FILE_TOOLS`` both work; the list is not "one toolset or many
    functions."
    """
    if tools is None:
        return []
    if is_toolset(tools) or callable(tools):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        return list(tools)
    raise TypeError(
        f"tools must be a tool, a toolset, or a sequence of them, not {type(tools).__name__}"
    )


class FlowMessages(MessageBuilder):
    """``Flow.build_messages`` as the step-level message builder."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def build(self, node: Node) -> list[dict[str, str]]:
        return self.flow.build_messages(node)


@contextmanager
def timed(node: Node) -> Iterator[None]:
    """Stamp whatever a step lands after ``node`` with how long the step ran."""
    started = time.time()
    try:
        yield
    finally:
        landed = node.parent_agent.frontier
        if landed is not node:  # a crashed or cancelled step lands nothing
            landed.started_at, landed.finished_at = started, time.time()


class Flow:
    """Own the model, tools, prompts, and REPLs. The queue owns running agents."""

    def __init__(
        self,
        llm: Any,
        *,
        root_config: AgentConfig | None = None,
        restore: Literal["replay", "lazy"] = "replay",
        system_prompt: SystemPromptSource | None = None,
        render_fn: RenderFn | None = None,
        prompt_profiles: dict[str, PromptProfile] | None = None,
        prompt_router: Callable[[Flow, AgentStart], str] | None = None,
        tools: Any = None,
        runtime: Runtime | None = None,
        execution_guard: ExecutionGuard | None = None,
        llm_clients: dict[str, Any] | None = None,
        llm_request_timeout: float | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        use_agent_tree: bool = False,
        enable_structured_output: bool = False,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        if pool is not None and workers is not None:
            raise ValueError("pass workers or pool, not both")
        self.root_config = root_config or AgentConfig()
        self.restore = restore
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = system_prompt or DEFAULT_BUILDER
        self.render_fn = render_fn or default_render
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.use_agent_tree = use_agent_tree
        self.use_llm_query = use_llm_query
        self.enable_structured_output = enable_structured_output
        self.runtime = runtime or LocalRuntime()
        self.execution_guard = execution_guard
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
        self._restored_agents: set[int] = set()
        self._restore_lock = asyncio.Lock()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.tools: dict[str, Any] = {}
        self.toolsets: list[Any] = []
        self._bind_toolset(BuiltIns())
        for item in as_tool_items(tools):
            self.add_tool(item)
        if use_llm_query:
            self.add_tool(llm_query(self), name="llm_query")
            self.add_tool(llm_query_batched(self), name="llm_query_batched")
        self.steps = dict(DEFAULT_STEPS)
        self.messages = FlowMessages(self)
        self.wrapped_runtime = WrappedRuntime(
            self.runtime,
            self.build_tools,
            self.execution_guard,
        )

    @property
    def repls(self) -> dict[str, Repl]:
        return self.runtime.repls

    # -- Prompts ----------------------------------------------------------

    def profile(self, agent: AgentStart) -> PromptProfile:
        name = (
            self.prompt_router(self, agent)
            if self.prompt_router is not None
            else agent.config.prompt_profile
        )
        if name in self.prompt_profiles:
            return self.prompt_profiles[name]
        if name == "default":
            return PromptProfile(
                system=self.system_prompt,
                render_fn=self.render_fn,
            )
        raise ValueError(f"unknown prompt {name!r}")

    def build_messages(self, node: Node) -> list[dict[str, str]]:
        """The prompt as of ``node``: system message, then that agent's turns."""
        agent = node.parent_agent
        profile = self.profile(agent)
        render_fn = profile.render_fn or self.render_fn
        current_messages = render_fn(self.runtime, node)
        keep = agent.config.keep_n_messages
        if keep is not None:
            current_messages = current_messages[-keep:] if keep > 0 else []
        previous_keep = None if keep is None else max(keep - len(current_messages), 0)
        previous_messages = [] if node.prev is None else node.prev.project(keep=previous_keep)
        system = as_system_prompt_fn(profile.system or self.system_prompt)(self, node)
        return [
            {"role": "system", "content": system},
            *previous_messages,
            *current_messages,
        ]

    async def call_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        *,
        key: object | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMChunk]:
        client = self._create_pooled_llm(
            model,
            key=key,
            request_kwargs=kwargs,
        )
        async for chunk in client.stream(messages):
            yield chunk

    async def call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        *,
        key: object | None = None,
        **kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        client = self._create_pooled_llm(
            model,
            key=key,
            request_kwargs=kwargs,
        )
        return await client.completion(messages)

    def _create_pooled_llm(
        self,
        model: str,
        *,
        key: object | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> PooledLLMClient:
        if model not in self._llm_clients:
            raise ValueError(f"unknown model {model!r}")
        return PooledLLMClient(
            self._llm_clients[model],
            self.pool,
            timeout=self.llm_request_timeout,
            key=key,
            request_kwargs=request_kwargs,
        )

    def llm_for_step(self, node: Node) -> PooledLLMClient:
        agent = node.parent_agent
        return self._create_pooled_llm(
            agent.config.model,
            key=agent.id,
        )

    @property
    def llm_query(self):
        """The one-shot query tool bound to this flow."""
        return llm_query(self)

    @property
    def llm_query_batched(self):
        """The fan-out tool, bound to this flow — callable with or without opting in."""
        return llm_query_batched(self)

    # -- Steps ------------------------------------------------------------

    def update_step_fn(
        self,
        node_type: type[Node],
        step_type: type[StepFunction],
    ) -> None:
        self.steps[node_type] = step_type

    def get_step_fn(self, node: Node) -> type[StepFunction]:
        for node_type in type(node).__mro__:
            step_type = self.steps.get(node_type)
            if step_type is not None:
                return step_type
        raise TypeError(f"cannot step {type(node).__name__}")

    async def step(self, node: Node) -> Transition:
        """Take one complete graph step; only cancellation escapes as an exception."""
        error: BaseException | None = None

        with timed(node):
            try:
                if self.budget_exceeded(node):
                    landed = node.append(DoneOutput(result=BUDGET_EXCEEDED))
                else:
                    step_type = self.get_step_fn(node)
                    step = step_type(
                        llm=self.llm_for_step(node),
                        messages=self.messages,
                        runtime=self.wrapped_runtime,
                    )
                    landed = await step(node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed step is a transition
                error = exc
                detail = str(exc)
                text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
                scope = "run" if node.parent_agent is node.root else "child"
                landed = node.parent_agent.frontier.append(
                    DoneOutput(
                        content=text,
                        result=f"[{scope} failed: {text}]",
                    )
                )

        return Transition(submitted=node, created=landed, error=error)

    def budget_exceeded(self, node: Node) -> bool:
        limit = node.parent_agent.config.max_budget
        if limit is None:
            return False
        # Never discard a reply already paid for. The answer arrives in an LLMOutput
        # and only lands once its ExecAction runs finish(), so stopping at either one
        # spends the tokens and throws the result away. The gate belongs in front of
        # the next model call, which is where `budget_nearly_spent` asks for an answer.
        if isinstance(node, (LLMOutput, ExecAction)):
            return False
        root = node.root
        return root is not None and root.usage.total >= limit

    async def execute_action(self, node: ExecAction) -> ReplRun:
        return await self.wrapped_runtime.execute(node)

    # -- Tools ------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.tools[name] = value
        self.runtime.inject_live(name, value)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        if is_toolset(fn):
            instance = fn() if isinstance(fn, type) else fn
            self._bind_toolset(instance)
            return
        self.inject(name or tool_name(fn), fn)

    def _bind_toolset(self, instance: Any) -> None:
        for existing in self.toolsets:
            if type(existing) is type(instance):
                raise ValueError(f"toolset {type(instance).__name__!r} is already bound")
        pending: list[tuple[str, Any]] = []
        for tool_name_, method in toolset_members(instance):
            meta = getattr(getattr(method, "__func__", method), "_tool_meta", None)
            if meta is None or not meta.inject:
                continue
            if tool_name_ in RESERVED_TOOLS:
                raise ValueError(f"{tool_name_!r} is reserved")
            if tool_name_ in self.tools:
                raise ValueError(
                    f"tool {tool_name_!r} from {type(instance).__name__} collides "
                    f"with an already-bound name"
                )
            pending.append((tool_name_, method))
        self.toolsets.append(instance)
        for tool_name_, method in pending:
            self.inject(tool_name_, method)

    def remove_tool(self, name: str) -> Any:
        if name in RESERVED_TOOLS:
            raise ValueError(f"{name!r} is reserved")
        self.runtime.remove_live(name)
        return self.tools.pop(name, None)

    def tool_namespace_for_prompt(self, node: Node) -> dict[str, Any]:
        namespace = self.runtime.namespace_for(node)
        return namespace if namespace is not None else self.build_tools(node)

    def build_tools(self, node: Node) -> dict[str, Any]:
        finish = self.finish_tool(node)
        running = (
            tuple(current for current, _task in self.queue.running.values())
            if self.queue is not None
            else ()
        )
        agents = build_agent_directory(node.parent_agent, running_nodes=running)
        namespace = {
            **self.tools,
            "finish": finish,
            "launch_subagent": self.launch_tool(node),
            "asyncio": asyncio,
            "INPUTS": node.parent_agent.config.inputs,
            AGENTS_BINDING: agents,
            AGENT_OBSERVE_TOOL: self.observe_agent_tool(node),
            AGENT_WAIT_TOOL: self.wait_agent_tool(node),
        }
        if self.use_agent_tree:
            namespace["AGENTS"] = agents
        return namespace

    def wait_agent_tool(self, node: Node):
        @tool("Wait for an existing agent and return its result.", proxy=True)
        async def wait_agent(agent_id: str) -> Any:
            root = node.root
            if root is None:
                raise RuntimeError("node is detached")
            target = root.find_agent(agent_id)
            if target is None:
                raise KeyError(f"unknown agent {agent_id!r}")
            if not target.terminal:
                queue = self.queue
                if queue is None:
                    raise RuntimeError("waiting for an agent requires an active stream")
                await queue.join(target)
            if isinstance(node, ExecAction):
                node.mark_agent_retrieved(agent_id)
            return target.result()

        return wait_agent

    def observe_agent_tool(self, node: Node):
        @tool("Record access to one completed agent result.", proxy=True)
        def observe_agent(agent_id: str) -> None:
            root = node.root
            if root is None:
                raise RuntimeError("node is detached")
            target = root.find_agent(agent_id)
            if target is None:
                raise KeyError(f"unknown agent {agent_id!r}")
            if not target.terminal:
                raise asyncio.InvalidStateError(f"agent {target.config.path!r} is not completed")
            if isinstance(node, ExecAction):
                node.mark_agent_retrieved(agent_id)

        return observe_agent

    def finish_tool(self, node: Node):
        schema = node.parent_agent.config.output_schema

        @tool("Submit this agent's final answer and end its run.", proxy=True)
        def finish(answer: object) -> None:
            value = parse_structured_answer(answer, schema) if schema is not None else str(answer)
            raise DoneSignal(value)

        return finish

    def launch_tool(self, node: Node):
        mutation_lock = asyncio.Lock()

        @tool(
            "Spawn or resume one focused child agent with an explicitly chosen "
            "registered model; the caller remains responsible for final synthesis.",
            proxy=True,
        )
        async def launch_subagent(
            goal: str,
            *,
            model: str,
            name: str | None = None,
            inputs: dict[str, str] | None = None,
            output_schema: Any = None,
            prompt_profile: str | None = None,
            reuse_repl: bool = False,
        ) -> AgentHandle:
            if not isinstance(node, ExecAction):
                raise TypeError("launch_subagent requires an ExecAction")
            if not isinstance(goal, str):
                raise TypeError("launch_subagent goal must be a string")
            spec = {
                "query": goal,
                "name": name,
                "inputs": dict(inputs or {}),
                "model": model,
                "output_schema": output_schema,
                "prompt_profile": prompt_profile,
                "reuse_repl": reuse_repl,
            }
            async with mutation_lock:
                existing = {id(child) for child in node.children if isinstance(child, AgentStart)}
                resolved = self.resolve_child(node, spec, current_rpc_call_id())
                if id(resolved) not in existing:
                    self.submit_child(resolved)
            return AgentHandle(
                agent_id=resolved.id,
                name=resolved.config.name,
                path=resolved.config.path,
            )

        return launch_subagent

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        """Resolve one launch call to a direct child or refusal."""
        explicit_name = spec.get("name")
        existing = next(
            (
                child
                for child in action.children
                if isinstance(child, AgentStart)
                and (
                    child.config.launch_call_id == call_id
                    or (explicit_name is not None and child.config.name == explicit_name)
                )
            ),
            None,
        )
        if existing is not None:
            return existing

        parent = action.parent_agent
        query = spec.get("query", "")
        if explicit_name is None:
            used = {child.config.name for child in parent.sub_agents}
            index = call_id
            while f"child{index}" in used:
                index += 1
            name = f"child{index}"
        else:
            name = explicit_name
        validate_agent_name(name)
        if any(child.config.name == name for child in parent.sub_agents):
            raise ValueError(f"duplicate child name {name!r}")

        if parent.config.depth >= parent.config.max_depth:
            raise ValueError(f"cannot launch beyond max depth {parent.config.max_depth}")
        if len(query) > parent.config.max_query_chars:
            raise ValueError(f"subagent goal exceeds {parent.config.max_query_chars} characters")
        model = spec["model"]
        if model not in self._llm_clients:
            available = ", ".join(sorted(self._llm_clients))
            raise ValueError(f"unknown model {model!r}; available models: {available}")
        return self.new_child(action, name, spec, call_id=call_id)

    def submit_child(self, child: AgentStart) -> None:
        """Submit a newly attached child and all unfinished restored leaves."""
        queue = self.queue
        if queue is None:
            raise RuntimeError("launch_subagent requires an active stream")
        if child.terminal:
            return
        for leaf in child.leaves():
            owner = leaf.parent_agent
            if owner is not None and not owner.terminal:
                queue.submit(
                    leaf,
                    self.step,
                    publish=isinstance(leaf, AgentStart),
                )

    def new_child(
        self,
        node: Node,
        name: str,
        spec: dict[str, Any],
        *,
        call_id: int,
    ) -> AgentStart:
        """Open a child agent of ``node``'s agent, attached to ``node``."""
        schema = spec.get("output_schema")
        overrides = {
            key: value
            for key, value in {
                "inputs": dict(spec.get("inputs") or {}),
                "model": spec.get("model"),
                "prompt_profile": spec.get("prompt_profile"),
                "output_schema": (json_schema_for(schema) if schema is not None else None),
                "reuse_repl": spec.get("reuse_repl"),
                "launch_call_id": call_id,
            }.items()
            if value is not None
        }
        child = AgentStart(
            content=spec.get("query") or DEFAULT_QUERY,
            config=node.parent_agent.config.child(name, **overrides),
        )
        attached = node.append(child)
        node.children.sort(
            key=lambda value: (
                not isinstance(value, AgentStart),
                (
                    value.config.launch_call_id
                    if isinstance(value, AgentStart) and value.config.launch_call_id is not None
                    else 0
                ),
            )
        )
        parent = node.parent_agent
        parent.sub_agents.sort(
            key=lambda value: (
                value.parent.seq if value.parent is not None else 0,
                value.config.launch_call_id if value.config.launch_call_id is not None else 0,
            )
        )
        return attached

    # -- Running ----------------------------------------------------------

    async def replay(self, root: AgentStart) -> None:
        """Rebuild the namespaces of a graph we did not run, from its recorded code.

        Appends nothing: ``launch_subagent`` resolves children already attached
        to each action and reads finished results from their terminal nodes. A block
        that failed the first time fails the same way here, leaving the same partial
        bindings, which is why output and errors are discarded.
        """
        actions = [
            node
            for node in root.walk()
            if isinstance(node, ExecAction) and node is not node.parent_agent.frontier
        ]
        actions.sort(
            key=lambda node: (
                node.repl_execution_order is None,
                node.repl_execution_order or 0,
                node.created_at,
            )
        )
        for node in actions:
            agent = node.parent_agent
            if agent.terminal and not agent.config.reuse_repl:
                continue  # it answered; nothing will run in this namespace again
            repl = self.runtime.repl_for(agent)
            repl.structured_output = agent.config.output_schema is not None
            repl.seed(self.build_tools(node), agent.config.inputs)
            repl.update_env({RLMFLOW_REPLAY: "1"})
            try:
                await self.runtime.execute(node, node.code)
            finally:
                repl.update_env({RLMFLOW_REPLAY: "0"})

    async def ensure_replayed(self, agent: AgentStart) -> None:
        """Restore an unfinished namespace immediately before its first execution."""
        if id(agent) in self._restored_agents:
            return
        async with self._restore_lock:
            if id(agent) in self._restored_agents:
                return
            if self.runtime.get(agent) is not None:
                self._restored_agents.add(id(agent))
                return

            root = agent
            while root.config.reuse_repl and root.parent is not None:
                parent = root.parent.parent_agent
                if parent is None or self.runtime.get(parent) is not None:
                    break
                root = parent
            await self.replay(root)
            self._restored_agents.update(
                id(node) for node in root.walk() if isinstance(node, AgentStart)
            )

    async def run_streaming(
        self,
        root: AgentStart | str,
        *roots: AgentStart | str,
        until: StepUntil = "done",
        close_repls: bool = False,
    ) -> AsyncIterator[Node]:
        """Drive one or more roots through one graph-agnostic queue."""
        agents = [self.start(item) if isinstance(item, str) else item for item in (root, *roots)]
        if self.queue is not None:
            raise RuntimeError("this Flow is already driving a stream")
        if len({id(agent) for agent in agents}) != len(agents):
            raise RuntimeError("the same root cannot be driven twice")

        boundary = boundaries.resolve(until)
        for agent in agents:
            if self.runtime.get(agent) is None:  # a graph this Flow has not run
                if self.restore == "replay":
                    await self.replay(agent)
            else:
                self._restored_agents.add(id(agent))
                for node in agent.walk():
                    if (
                        isinstance(node, AgentStart)
                        and not node.terminal
                        and self.runtime.get(node) is None
                    ):
                        if self.restore == "replay":
                            await self.replay(node)

        queue = TaskQueue()
        self.queue = queue
        driving = {id(agent) for agent in agents}

        try:
            for agent in agents:
                for leaf in agent.leaves():
                    owner = leaf.parent_agent
                    if owner is not None and not owner.terminal:
                        if self.restore == "lazy" and isinstance(leaf, ExecAction):
                            await self.ensure_replayed(owner)
                        queue.submit(leaf, self.step)

            while driving and queue:
                transition = await queue.next()
                node = transition.created
                root = node.root
                if root is None or id(root) not in driving:
                    continue

                root_error = (
                    transition.error
                    if (
                        not transition.is_agent_start
                        and transition.error is not None
                        and transition.submitted.parent_agent is root
                    )
                    else None
                )

                yield node
                stop = boundary is not None and boundary(node, root)
                if stop:
                    driving.discard(id(root))
                    await queue.cancel(root.walk())
                elif not transition.is_agent_start and not node.parent_agent.terminal:
                    if self.restore == "lazy" and isinstance(node, ExecAction):
                        await self.ensure_replayed(node.parent_agent)
                    queue.submit(node, self.step)

                if root_error is not None:
                    raise root_error
        finally:
            await queue.cancel()
            if self.queue is queue:
                self.queue = None
            if close_repls:
                for agent in agents:
                    for node in agent.walk():
                        if isinstance(node, AgentStart):
                            self.runtime.close_repl(node)

    async def arun(self, root: AgentStart | str, *, close_repls: bool = False) -> Any:
        agent = self.start(root) if isinstance(root, str) else root
        async for _node in self.run_streaming(agent, close_repls=close_repls):
            pass
        return agent.result()

    def run(self, root: AgentStart | str, *, close_repls: bool = False) -> Any:
        return asyncio.run(self.arun(root, close_repls=close_repls))

    def start(self, query: str = "", **overrides: Any) -> AgentStart:
        """A root agent carrying this flow's defaults, which keyword overrides win against.

        The module-level ``start`` does the building; this only supplies the defaults
        the flow was constructed with, so ``Flow(root_config=...)`` reaches the roots you
        run on it.
        """
        return start(query, config=self.root_config, **overrides)

    async def aclose(self) -> None:
        if self.queue is not None:
            await self.queue.cancel()
            self.queue = None
        closed: set[int] = set()
        clients = []
        for client in self._llm_clients.values():
            if id(client) in closed:
                continue
            closed.add(id(client))
            close = getattr(client, "aclose", None)
            if close is not None:
                clients.append(close())
        try:
            await asyncio.gather(*clients)
        finally:
            self.pool.close()
            self.runtime.close()


__all__ = ["Flow", "StepUntil", "start"]
