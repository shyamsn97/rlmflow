"""Run agents: find the ready ones in a graph, step each, append what it produced."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any, Literal

from rlmflow.engine import handoff
from rlmflow.engine.scheduler import (
    Pool,
    StepResult,
    StepUntil,
    TaskQueue,
    ThreadPool,
    resolve_boundary,
)
from rlmflow.engine.steps import (
    DEFAULT_TRANSITIONS,
    MAX_BUDGET_EXCEEDED,
    AgentBackend,
    TransitionFunction,
    check_backends,
    choose_transition,
    final_turn_spent,
    resolve_transition,
)
from rlmflow.graph import ops, reads
from rlmflow.graph.config import ContextPolicy, can_spawn
from rlmflow.graph.graph import Edge, Graph, StaleFrontier
from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    DoneOutput,
    ExecAction,
    FinalQuery,
    LLMOutput,
    LLMUsage,
    Node,
    Step,
    TurnMode,
    current_step,
    normalize_schemas,
    running_step,
)
from rlmflow.llm import PooledLLMClient
from rlmflow.prompts import (
    AGENTS_TEXT,
    CORE_PROTOCOL_TEXT,
    EVIDENCE_POLICY,
    EXECUTION_POLICY,
    LOCAL_WORK_EXAMPLE,
    WORKER_ADDENDUM,
    PinnedPromptSectionError,
    Prompt,
    PromptModifierSource,
    PromptProfile,
    SystemPromptSource,
    add_argument_sections,
    apply_prompt_modifiers,
    build_messages,
    prompt_for,
    render_tools,
)
from rlmflow.runtime import ExecutionGuard, LocalRuntime, Runtime, WrappedRuntime
from rlmflow.runtime.repl import Repl
from rlmflow.runtime.runtime import ensure_repl_rebuilt, rebuild_repl, rebuild_unfinished_repls
from rlmflow.structured import (
    DEFAULT_OUTPUT_CODEC,
    ChatStructuredEngine,
    OutputCodec,
    Schema,
    StructuredEngine,
    StructuredRequest,
    StructuredResult,
    UnsupportedStructuredOutputError,
    as_finish_value,
)
from rlmflow.tools import launch, namespace
from rlmflow.tools.builtins import builtin_tools, query


def _stepping_agent() -> AgentStart:
    step = current_step()
    if step is None:
        raise RuntimeError("this runs inside a step; wrap direct calls in running_step(...)")
    return step.agent


class Flow:
    """Own the model, tools, prompts, and REPLs. The queue owns running agents."""

    def __init__(
        self,
        llm: Any,
        *,
        root_config: AgentConfig | None = None,
        restore: Literal["replay", "lazy"] = "replay",
        system_prompt: SystemPromptSource | None = None,
        system_prompt_modifiers: PromptModifierSource | None = None,
        user_prompt_modifiers: PromptModifierSource | None = None,
        prompt_profiles: dict[str, PromptProfile] | None = None,
        prompt_router: Callable[[Flow, AgentStart], str] | None = None,
        tools: Any = None,
        runtime: Runtime | None = None,
        execution_guard: ExecutionGuard | None = None,
        llm_clients: dict[str, Any] | None = None,
        backends: dict[str, AgentBackend] | None = None,
        llm_request_timeout: float | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        llm_query_cache: bool = False,
        use_agent_tree: bool = False,
        output_codec: OutputCodec | None = None,
        structured_engine: StructuredEngine | None = None,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        if pool is not None and workers is not None:
            raise ValueError("pass workers or pool, not both")
        self.backends = check_backends(backends)
        self.output_codec = output_codec or DEFAULT_OUTPUT_CODEC
        self.root_config = normalize_schemas(root_config or AgentConfig(), self.output_codec)
        self.restore = restore
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = system_prompt
        self.system_prompt_modifiers = system_prompt_modifiers
        self.user_prompt_modifiers = user_prompt_modifiers
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.use_agent_tree = use_agent_tree
        self.use_llm_query = use_llm_query
        self.llm_query_cache = llm_query_cache
        self._llm_query_cache: dict[str, Any] = {}
        self._transitions: list[TransitionFunction] = []
        self.runtime = runtime or LocalRuntime()
        self.execution_guard = execution_guard
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._verifiers: dict[str, Callable[[object], Any]] = {}
        self._restored_agents: set[str] = set()
        #: Waiting agent id -> the agents its open waits and asks are on.
        self._waits: dict[str, list[str]] = {}
        self._restore_lock = asyncio.Lock()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        for backend in self.backends.values():
            if (bind := getattr(backend, "bind", None)) is not None:
                bind(self)
        self.structured_engine = (
            structured_engine if structured_engine is not None else ChatStructuredEngine()
        )
        bind_engine = getattr(self.structured_engine, "bind", None)
        if bind_engine is not None:
            bind_engine(self)
        self.tools: dict[str, Any] = {}
        for item in namespace.as_tool_items(tools):
            self.add_tool(item)
        self.wrapped_runtime = WrappedRuntime(
            self.runtime,
            self.build_namespace,
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
                system_modifiers=self.system_prompt_modifiers,
                user_modifiers=self.user_prompt_modifiers,
            )
        raise ValueError(f"unknown prompt {name!r}")

    def build_messages(self, node: Node) -> list[dict[str, str]]:
        """The prompt as of ``node``: system message, then that agent's turns."""
        return build_messages(self, node)

    def default_system_prompt(self, node: Node) -> Prompt:
        """Build a fresh shared system prompt before node customization."""
        agent = _stepping_agent()
        profile = self.profile(agent)
        source = profile.system if profile.system is not None else self.system_prompt
        if source is None:
            prompt = Prompt()
            prompt.add_section(
                "protocol.core",
                CORE_PROTOCOL_TEXT,
                order=100,
                source="Flow.default_system_prompt",
                pinned=True,
            )
            prompt.add_section(
                "policy.evidence",
                EVIDENCE_POLICY,
                order=150,
                source="Flow.default_system_prompt",
            )
            prompt.add_section(
                "policy.execution",
                EXECUTION_POLICY if can_spawn(current_step().graph, agent) else WORKER_ADDENDUM,
                order=200,
                source="Flow.default_system_prompt",
            )
            tools = self.render_tools(node)
            if tools:
                prompt.add_section(
                    "capabilities.tools",
                    tools,
                    order=500,
                    source="Flow.render_tools",
                    pinned=True,
                )
            if node.turn_mode is not TurnMode.FINAL:
                prompt.add_section(
                    "guidance.local",
                    LOCAL_WORK_EXAMPLE,
                    order=510,
                    source="Flow.default_system_prompt",
                )
                if self.use_agent_tree:
                    prompt.add_section(
                        "capabilities.agents",
                        AGENTS_TEXT,
                        order=520,
                        source="Flow.default_system_prompt",
                    )
            add_argument_sections(prompt, self, node)
        else:
            prompt = prompt_for(source, self, node)
        self._modify(prompt, node, self.system_prompt_modifiers, profile.system_modifiers)
        return prompt

    def build_system_prompt(self, node: Node) -> Prompt:
        """Build the final sectioned system prompt visible to ``node``."""
        default = self.default_system_prompt(node)
        prompt = node.build_system_prompt(default, self)
        if not isinstance(prompt, Prompt):
            raise TypeError(
                f"{type(node).__name__}.build_system_prompt() must return Prompt, "
                f"not {type(prompt).__name__}"
            )
        for section in default:
            if section.pinned and prompt.get(section.key) != section:
                raise PinnedPromptSectionError(
                    f"node {type(node).__name__} dropped or replaced pinned "
                    f"prompt section {section.key!r}"
                )
        return prompt

    def default_user_prompt(self, node: Node) -> Prompt:
        """Build a fresh shared current-turn prompt before node message shaping."""
        prompt = Prompt()
        instruction = node.instruction(self)
        if instruction:
            prompt.add_section(
                "turn.instruction",
                instruction,
                order=1000,
                source=type(node).__name__,
            )
        profile = self.profile(_stepping_agent())
        self._modify(prompt, node, self.user_prompt_modifiers, profile.user_modifiers)
        return prompt

    def _modify(
        self,
        prompt: Prompt,
        node: Node,
        shared: PromptModifierSource | None,
        own: PromptModifierSource | None,
    ) -> None:
        """Flow-wide modifiers first, then the profile's, without running one twice."""
        if own is not shared:
            apply_prompt_modifiers(prompt, shared, self, node)
        apply_prompt_modifiers(prompt, own, self, node)

    def render_tools(self, node: Node) -> str:
        """The ``### Tools`` list: exactly the callables this node's REPL binds."""
        return render_tools(self, node)

    async def call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        **kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        client = self._create_pooled_llm(model, request_kwargs=kwargs)
        return await client.completion(messages)

    async def structured_query(
        self,
        state: object,
        *,
        output_schema: Schema,
        instruction: str | None = None,
        model: str | None = None,
        **options: Any,
    ) -> StructuredResult:
        """Produce and locally validate a structured value from supplied state."""
        schema = self.output_codec.normalize(output_schema)
        request = StructuredRequest(
            state=state,
            schema=schema,
            source_schema=output_schema,
            instruction=instruction,
            model=model,
            options=options,
        )
        reasons = tuple(self.structured_engine.supports(request))
        if reasons:
            raise UnsupportedStructuredOutputError(
                type(self.structured_engine).__name__,
                reasons,
            )
        result = await self.structured_engine.generate(request, self.output_codec)
        value = self.output_codec.validate(result.value, schema)
        metadata = as_finish_value(result.metadata)
        if not isinstance(metadata, dict):
            raise TypeError("structured engine metadata must be a JSON object")
        return StructuredResult(
            value=as_finish_value(value),
            metadata=metadata,
        )

    def _create_pooled_llm(
        self,
        model: str,
        *,
        request_kwargs: dict[str, Any] | None = None,
    ) -> PooledLLMClient:
        if model not in self._llm_clients:
            raise ValueError(f"unknown model {model!r}")
        return PooledLLMClient(
            self._llm_clients[model],
            self.pool,
            timeout=self.llm_request_timeout,
            request_kwargs=request_kwargs,
        )

    def llm_for_step(self, node: Node) -> PooledLLMClient:
        return self._create_pooled_llm(_stepping_agent().config.model)

    async def llm_query(
        self,
        prompt: str,
        *,
        model: str | None = None,
        schema: Any = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        cache: bool | None = None,
    ) -> Any:
        """Make one context-free model call from host code."""
        return await query(
            self,
            prompt,
            model=model,
            schema=schema,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            cache=cache,
        )

    def transition(
        self,
        *,
        source: type[Node] | tuple[type[Node], ...] | str,
        when: Callable[[Node], bool] | str | None = None,
    ):
        """Register a node-only automatic transition handler on this Flow."""

        def decorate(fn: Callable[[Node], Any]) -> TransitionFunction:
            transition_fn = TransitionFunction.create(
                fn,
                source=source,
                when=when,
                node_only=True,
            )
            self._transitions.append(transition_fn)
            return transition_fn

        return decorate

    # -- Steps ------------------------------------------------------------

    async def step(self, g: Graph, agent: AgentStart, node: Node | None = None) -> Node | None:
        """Take one step for ``agent`` from ``node`` (its frontier by default), outside a run."""
        step = Step(g, agent, node if node is not None else reads.frontier(g, agent))
        with running_step(step):
            return (await self._drive(step)).created

    async def _drive(self, step: Step) -> StepResult:
        """Queue entry: one step, with infrastructure failure as a terminal node."""
        g, agent, node = step.graph, step.agent, step.node
        error: BaseException | None = None
        finalizing = self.budget_exceeded(node)
        if finalizing:
            await self._wait_for_budget_descendants(g, agent)

        started = time.time()
        try:
            if finalizing:
                landed: Node | None = (
                    DoneOutput(result=MAX_BUDGET_EXCEEDED) if final_turn_spent() else FinalQuery()
                )
                anchor = node
            else:
                producer, anchor, source = await resolve_transition(
                    (*self._transitions, *DEFAULT_TRANSITIONS),
                    node,
                    partial(choose_transition, self),
                )
                landed = await producer.produce(self, source)
            _stamp(landed, started)
            ops.append(g, agent, landed, after=anchor)
        except asyncio.CancelledError:
            raise
        except StaleFrontier:
            landed = None  # a host write won; the agent is rescheduled from there
        except Exception as exc:  # noqa: BLE001 - a failed step is a transition
            error = exc
            detail = str(exc)
            text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            scope = "run" if reads.launcher(g, agent) is None else "child"
            landed = DoneOutput(content=text, result=f"[{scope} failed: {text}]")
            _stamp(landed, started)
            ops.append(g, agent, landed)

        return StepResult(step=step, created=landed, error=error)

    def budget_exhausted(self, g: Graph) -> bool:
        """Whether the stepping agent's run has reached its soft token threshold.

        A run is the stepping agent's top-level ancestor and everything it launched.
        """
        agent = _stepping_agent()
        limit = agent.config.max_budget
        if limit is None:
            return False
        run = reads.tree(g, reads.lineage(g, agent)[0])
        return reads.usage(g, run).total >= limit

    def budget_exceeded(self, node: Node) -> bool:
        """Whether this node must enter its agent's one final turn."""
        # Never discard a reply already paid for. The answer arrives in an LLMOutput
        # and only lands once its ExecAction runs submit(), so stopping at either one
        # spends the tokens and throws the result away. FinalQuery itself must also
        # run even though the threshold has already been crossed.
        if isinstance(node, (FinalQuery, LLMOutput, ExecAction)):
            return False
        return self.budget_exhausted(current_step().graph)

    async def _wait_for_budget_descendants(self, g: Graph, agent: AgentStart) -> None:
        """Let unfinished descendants finalize before their parent synthesizes."""
        queue = self.queue
        if queue is None:
            return
        descendants = reads.tree(g, agent)[1:]
        await queue.join(lambda: all(reads.terminal(g, child) for child in descendants))

    # -- Tools ------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
        namespace.inject(self, name, value)

    def add_tool(self, fn: Any, *, name: str | None = None) -> None:
        namespace.add_tool(self, fn, name=name)

    def remove_tool(self, name: str) -> Any:
        return namespace.remove_tool(self, name)

    def build_tools(self, node: Node) -> dict[str, Any]:
        """Build the complete model-visible tool surface for one node."""
        return {
            **builtin_tools(self, node),
            **namespace.application_tools(self, node),
        }

    def build_namespace(self, node: Node) -> dict[str, Any]:
        """Add non-tool runtime bindings to this node's tool surface."""
        return namespace.build_namespace(self, node, self.build_tools(node))

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        call_id: int,
    ) -> AgentStart:
        """Resolve one launch call to a child; subclasses hook per-child setup here."""
        return launch.resolve_child(self, action, spec, call_id)

    def backend_for(self, agent: AgentStart) -> AgentBackend:
        """The backend named by ``agent.config.backend``."""
        name = agent.config.backend
        if name not in self.backends:
            available = ", ".join(sorted(self.backends)) or "none"
            raise RuntimeError(
                f"agent {agent.config.name!r} requires backend {name!r}; this Flow has: {available}"
            )
        return self.backends[name]

    # -- Running ----------------------------------------------------------

    async def replay(self, g: Graph, agent: AgentStart | None = None) -> None:
        """Rebuild REPL namespaces from recorded code: one agent, or every unfinished one."""
        if agent is not None:
            await rebuild_repl(self, g, agent)
            return
        for agent in g.nodes(AgentStart):
            if not reads.terminal(g, agent):
                await rebuild_repl(self, g, agent)

    def running_steps(self) -> dict[str, Node]:
        """The node each in-flight step started from, by agent id."""
        if self.queue is None:
            return {}
        return {agent_id: step.node for agent_id, (step, _task) in self.queue.running.items()}

    async def run_streaming(
        self,
        g: Graph,
        *,
        until: StepUntil = "done",
        close_repls: bool = False,
    ) -> AsyncIterator[Node | Edge]:
        """Drive every ready agent in the graph and yield what each step adds.

        Yields graph items (nodes and edges) in log order.
        """
        if self.queue is not None:
            raise RuntimeError("this Flow is already driving a stream")
        boundary = resolve_boundary(until)
        cursor = len(g) - 1
        queue = TaskQueue()
        self.queue = queue
        self._active_loop = asyncio.get_running_loop()
        try:
            await rebuild_unfinished_repls(self, g)
            for agent in g.nodes(AgentStart):
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
                    yield item
                    stop = stop or (boundary is not None and boundary(item, g))
                if stop:
                    return
                await queue.notify()
                if (
                    result is not None
                    and result.error is not None
                    and reads.launcher(g, result.agent) is None
                ):
                    raise result.error
                await self._schedule(g)
        finally:
            interrupted = list(queue.running)
            await queue.cancel()
            # An interrupted step loses its REPL; replay it again on the next run.
            self._restored_agents.difference_update(
                agent_id for agent_id in interrupted if self.runtime.get(agent_id) is None
            )
            if self.queue is queue:
                self.queue = None
                self._active_loop = None
            if close_repls:
                for agent in g.nodes(AgentStart):
                    self.runtime.close_repl(agent)

    def _react(self, g: Graph, result: StepResult) -> None:
        """Flow policy after a step lands: release live state, deliver results."""
        agent = result.agent
        if (
            isinstance(result.created, DoneOutput)
            and agent.config.context_policy != ContextPolicy.RESIDENT
        ):
            self.runtime.close_repl(agent)
            self._restored_agents.discard(agent.id)
        handoff.react(g, agent)
        if isinstance(result.created, DoneOutput):
            for recipient in handoff.recipients(g, agent):
                if recipient not in self.queue:
                    handoff.react(g, recipient)

    async def _schedule(self, g: Graph) -> None:
        """Start one step for every ready agent that has none in flight."""
        queue = self.queue
        for agent in reads.ready(g):
            if agent in queue:
                continue
            if agent.config.context_policy == ContextPolicy.EPHEMERAL and isinstance(
                reads.frontier(g, agent), DoneOutput
            ):
                continue
            ops.deliver(g, agent)
            if reads.terminal(g, agent):
                continue
            node = reads.frontier(g, agent)
            if isinstance(node, ExecAction):
                await ensure_repl_rebuilt(self, g, agent)
            queue.submit(Step(g, agent, node), self._drive)

    async def arun(
        self,
        target: Graph | str,
        *,
        close_repls: bool = False,
        **args: Any,
    ) -> Any:
        """Run to quiescence. A graph returns itself; a task string returns its result."""
        if isinstance(target, str):
            g = Graph()
            agent = g.create_agent(target, config=self.root_config, **args)
        elif args:
            raise TypeError("agent arguments can only be passed with a task string")
        else:
            g, agent = target, None
        async for _item in self.run_streaming(g, close_repls=close_repls):
            pass
        return g if agent is None else reads.result(g, agent)

    def run(
        self,
        target: Graph | str,
        *,
        close_repls: bool = False,
        **args: Any,
    ) -> Any:
        return asyncio.run(self.arun(target, close_repls=close_repls, **args))

    async def aclose(self) -> None:
        if self.queue is not None:
            await self.queue.cancel()
            self.queue = None
        managed = [*self._llm_clients.values(), self.structured_engine, *self.backends.values()]
        owners = {id(item): item for item in managed}
        closing = [close() for item in owners.values() if (close := getattr(item, "aclose", None))]
        try:
            await asyncio.gather(*closing)
        finally:
            self.pool.close()
            self.runtime.close()


def _stamp(node: Node, started: float) -> None:
    node.started_at, node.finished_at = started, time.time()
    node.annotate("timing", node.timing())


def messages_to_query(messages: list[dict[str, Any]]) -> str:
    """Project chat messages into a single agent query."""

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content") or ""
        parts.append(f"{role}: {content if isinstance(content, str) else str(content)}")
    return "\n\n".join(parts)


class FlowLLM:
    """Expose a :class:`Flow` through a sync ``chat(messages)`` API."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow
        self.last_graph: Graph | None = None
        self.last_usage = LLMUsage()

    def chat(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        graph = Graph()
        agent = graph.create_agent(messages_to_query(messages), config=self.flow.root_config)
        self.flow.run(graph)
        self.last_graph = graph
        self.last_usage = reads.usage(graph)
        return reads.result(graph, agent)

    def completion(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], *args, **kwargs)

    def close(self) -> None:
        self.flow.runtime.close_repls()


__all__ = ["Flow", "FlowLLM", "StepUntil", "messages_to_query"]
