"""Run agents: prompt the model, execute its code, delegate to children."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

from rlmflow import boundaries
from rlmflow.engine import delegation, handoff, restore
from rlmflow.engine.boundaries import StepUntil
from rlmflow.engine.execution import Pool, TaskQueue, ThreadPool, Transition
from rlmflow.engine.transitions import Transitions
from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    DoneOutput,
    ExecAction,
    FinalQuery,
    LLMOutput,
    LLMUsage,
    Node,
    start,
)
from rlmflow.llm import LLMChunk, PooledLLMClient
from rlmflow.prompts import (
    PromptBuilder,
    PromptProfile,
    RenderFn,
    SystemPromptSource,
    as_system_prompt_fn,
    default_render,
    messages,
)
from rlmflow.prompts.prompts import render_inputs_text
from rlmflow.runtime import ExecutionGuard, LocalRuntime, Runtime, WrappedRuntime
from rlmflow.runtime.repl import Repl, ReplRun
from rlmflow.tools import namespace
from rlmflow.tools.builtins import BuiltIns
from rlmflow.tools.llm_query import llm_query, llm_query_batched


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

    transitions: ClassVar[Transitions] = Transitions()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.transitions = cls.transitions.derive()

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
        delegation_model: str | None = None,
        llm_request_timeout: float | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        use_llm_query_batched: bool = True,
        use_agent_tree: bool = False,
        enable_structured_output: bool = True,
        transitions: Transitions | None = None,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        if pool is not None and workers is not None:
            raise ValueError("pass workers or pool, not both")
        self.root_config = root_config or AgentConfig()
        self.restore = restore
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = PromptBuilder() if system_prompt is None else system_prompt
        self.render_fn = render_fn or default_render
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.use_agent_tree = use_agent_tree
        self.use_llm_query = use_llm_query
        self.use_llm_query_batched = use_llm_query and use_llm_query_batched
        self.enable_structured_output = enable_structured_output
        self.transitions = (transitions or type(self).transitions).derive()
        self.runtime = runtime or LocalRuntime()
        self.execution_guard = execution_guard
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
        self._restored_agents: set[int] = set()
        self._restore_lock = asyncio.Lock()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        if delegation_model is not None and delegation_model not in self._llm_clients:
            available = ", ".join(sorted(self._llm_clients))
            raise ValueError(
                f"unknown delegation model {delegation_model!r}; available models: {available}"
            )
        self.delegation_model = delegation_model
        self.tools: dict[str, Any] = {}
        self.toolsets: list[Any] = []
        self._bind_toolset(BuiltIns())
        for item in namespace.as_tool_items(tools):
            self.add_tool(item)
        if use_llm_query:
            self.add_tool(llm_query(self), name="llm_query")
            if use_llm_query_batched:
                self.add_tool(llm_query_batched(self), name="llm_query_batched")
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
        return messages.build_messages(self, node)

    def build_system_prompt(self, node: Node) -> str:
        """The inherited protocol string for ``node``'s agent.

        Resolves ``profile.system`` / ``flow.system_prompt`` / ``PromptBuilder()``.
        Query nodes wrap this in ``UserQuery.build_system_prompt``.
        """
        profile = self.profile(node.parent_agent)
        source = profile.system or self.system_prompt or PromptBuilder()
        return as_system_prompt_fn(source)(self, node)

    def render_tools(self, node: Node) -> str:
        """Live ``Available in the REPL`` list: gated builtins plus host tools."""
        return messages.render_tools(self, node)

    def render_inputs(self, node: Node) -> str:
        """Per-agent INPUTS sizes, output schema, and depth."""
        return render_inputs_text(self, node)

    def transition_footer(self, node: Node) -> str:
        return messages.transition_footer(self, node)

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

    async def step(self, node: Node) -> Node:
        """Take one graph step and return the created node."""
        return (await self._drive(node)).created

    async def _drive(self, node: Node) -> Transition:
        """Queue entry: one step, with infrastructure failure as a terminal node."""
        error: BaseException | None = None
        finalizing = self.budget_exceeded(node)
        if finalizing:
            await self._wait_for_budget_descendants(node.parent_agent)

        with timed(node):
            try:
                if finalizing:
                    landed = FinalQuery()
                else:
                    produced = self.transitions.resolve(node)(self, node)
                    landed = await produced if inspect.isawaitable(produced) else produced
                node.append(landed)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed step is a transition
                error = exc
                detail = str(exc)
                text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
                scope = "run" if node.parent_agent is node.root else "child"
                landed = DoneOutput(
                    content=text,
                    result=f"[{scope} failed: {text}]",
                )
                node.parent_agent.frontier.append(landed)

        return Transition(submitted=node, created=landed, error=error)

    def budget_exhausted(self, node: Node) -> bool:
        """Whether this run has reached its configured soft token threshold."""
        agent = node.parent_agent
        limit = agent.config.max_budget
        if limit is None:
            return False
        root = node.root
        return root is not None and root.usage.total >= limit

    def budget_exceeded(self, node: Node) -> bool:
        """Whether this node must enter its agent's one final turn."""
        # Never discard a reply already paid for. The answer arrives in an LLMOutput
        # and only lands once its ExecAction runs finish(), so stopping at either one
        # spends the tokens and throws the result away. FinalQuery itself must also
        # run even though the threshold has already been crossed.
        if isinstance(node, (FinalQuery, LLMOutput, ExecAction)):
            return False
        return self.budget_exhausted(node)

    async def _wait_for_budget_descendants(self, agent: AgentStart) -> None:
        """Let unfinished descendants finalize before their parent synthesizes."""
        queue = self.queue
        if queue is None:
            return
        descendants: list[AgentStart] = []
        stack = list(agent.sub_agents)
        while stack:
            child = stack.pop()
            if not child.terminal:
                descendants.append(child)
            stack.extend(child.sub_agents)
        if descendants:
            await asyncio.gather(*(queue.join(child) for child in descendants))

    async def execute_action(self, node: ExecAction) -> ReplRun:
        return await self.wrapped_runtime.execute(node)

    # -- Tools ------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        namespace.inject(self, name, value)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        namespace.add_tool(self, fn, name=name)

    def _bind_toolset(self, instance: Any) -> None:
        namespace.bind_toolset(self, instance)

    def remove_tool(self, name: str) -> Any:
        return namespace.remove_tool(self, name)

    def tool_namespace_for_prompt(self, node: Node) -> dict[str, Any]:
        bound = self.runtime.namespace_for(node)
        return bound if bound is not None else self.build_tools(node)

    def build_tools(self, node: Node) -> dict[str, Any]:
        return namespace.build_namespace(self, node)

    def transition_tool(self, node: Node):
        return namespace.transition_tool(self, node)

    def wait_agent_tool(self, node: Node):
        return namespace.wait_agent_tool(self, node)

    def observe_agent_tool(self, node: Node):
        return namespace.observe_agent_tool(self, node)

    def finish_tool(self, node: Node):
        return namespace.finish_tool(self, node)

    def launch_tool(self, node: Node):
        return delegation.launch_tool(self, node)

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        """Resolve one previously unseen launch call to a child or refusal."""
        return delegation.resolve_child(self, action, spec, call_id)

    def submit_child(self, child: AgentStart) -> None:
        """Submit a newly attached child and all unfinished restored leaves."""
        delegation.submit_child(self, child)

    def new_child(
        self,
        node: Node,
        name: str,
        spec: dict[str, Any],
    ) -> AgentStart:
        """Open a child agent of ``node``'s agent, attached to ``node``."""
        return delegation.new_child(self, node, name, spec)

    # -- Running ----------------------------------------------------------

    async def replay(self, root: AgentStart) -> None:
        """Rebuild the namespaces of a graph we did not run, from its recorded code."""
        await restore.replay(self, root)

    async def ensure_replayed(self, agent: AgentStart) -> None:
        """Restore an unfinished namespace immediately before its first execution."""
        await restore.ensure_replayed(self, agent)

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

        for agent in agents:
            for node in agent.walk():
                if isinstance(node, AgentStart):
                    node.dependency_scheduled = False
            handoff.restore(agent)

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
                        delegation.submit_leaf(self, leaf)

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
                else:
                    extra = handoff.apply(node)
                    for dependent in delegation.ready_dependents(node):
                        self.submit_child(dependent)
                    if not transition.is_agent_start and not node.parent_agent.terminal:
                        if node.parent_agent.frontier is node:
                            if self.restore == "lazy" and isinstance(node, ExecAction):
                                await self.ensure_replayed(node.parent_agent)
                            queue.submit(node, self._drive)
                    for leaf in extra:
                        owner = leaf.parent_agent
                        if owner is not None and not owner.terminal:
                            delegation.submit_leaf(self, leaf)

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


import rlmflow.engine.steps  # noqa: F401, E402

__all__ = ["Flow", "StepUntil", "start"]
