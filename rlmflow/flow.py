"""Run agents: prompt the model, execute its code, delegate to children."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from typing import Any, Literal

from rlmflow import boundaries
from rlmflow.engine import delegation, handoff, restore, semantic, transitions
from rlmflow.engine.boundaries import StepUntil
from rlmflow.engine.execution import Pool, StepResult, TaskQueue, ThreadPool
from rlmflow.engine.steps import DEFAULT_TRANSITIONS
from rlmflow.engine.transitions import TransitionFunction
from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import (
    AgentConfig,
    AgentStart,
    DoneOutput,
    ExecAction,
    FinalQuery,
    LLMOutput,
    LLMUsage,
    Node,
    Submission,
    TurnMode,
    build_agent,
)
from rlmflow.llm import PooledLLMClient
from rlmflow.prompts import (
    CORE_PROTOCOL_TEXT,
    PinnedPromptSectionError,
    Prompt,
    PromptModifierSource,
    PromptProfile,
    SystemPromptSource,
    add_argument_sections,
    apply_prompt_modifiers,
    messages,
    prompt_for,
)
from rlmflow.prompts.text import (
    AGENTS_TEXT,
    EVIDENCE_POLICY,
    EXECUTION_POLICY,
    LOCAL_WORK_EXAMPLE,
    WORKER_ADDENDUM,
)
from rlmflow.runtime import ExecutionGuard, LocalRuntime, Runtime, WrappedRuntime
from rlmflow.runtime.repl import Repl
from rlmflow.structured import DEFAULT_OUTPUT_CODEC, OutputCodec, Schema, as_finish_value
from rlmflow.structured_engines import (
    ChatStructuredEngine,
    StructuredEngine,
    StructuredRequest,
    StructuredResult,
    UnsupportedStructuredOutputError,
)
from rlmflow.tools import namespace
from rlmflow.tools.builtins import builtin_tools, query


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


def _normalize_schemas(codec: OutputCodec, config: AgentConfig) -> AgentConfig:
    schemas = {
        key: codec.normalize(value)
        for key in ("output_schema", "child_output_schema")
        if (value := getattr(config, key)) is not None
    }
    return replace(config, **schemas) if schemas else config


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
        delegation_model: str | None = None,
        delegation_prompt_profile: str | None = None,
        delegation_reuse_worker: bool = False,
        llm_request_timeout: float | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        use_agent_tree: bool = False,
        output_codec: OutputCodec | None = None,
        structured_engine: StructuredEngine | None = None,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        if pool is not None and workers is not None:
            raise ValueError("pass workers or pool, not both")
        self.output_codec = output_codec or DEFAULT_OUTPUT_CODEC
        self.root_config = _normalize_schemas(self.output_codec, root_config or AgentConfig())
        self.restore = restore
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = system_prompt
        self.system_prompt_modifiers = system_prompt_modifiers
        self.user_prompt_modifiers = user_prompt_modifiers
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.use_agent_tree = use_agent_tree
        self.use_llm_query = use_llm_query
        self._transitions: list[TransitionFunction] = []
        self.runtime = runtime or LocalRuntime()
        self.execution_guard = execution_guard
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._verifiers: dict[str, Callable[[object], Any]] = {}
        self._restored_agents: set[int] = set()
        self._restore_lock = asyncio.Lock()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        if delegation_model is not None and delegation_model not in self._llm_clients:
            available = ", ".join(sorted(self._llm_clients))
            raise ValueError(
                f"unknown delegation model {delegation_model!r}; available models: {available}"
            )
        self.delegation_model = delegation_model
        self.delegation_prompt_profile = delegation_prompt_profile
        self.delegation_reuse_worker = delegation_reuse_worker
        self.structured_engine = (
            structured_engine
            if structured_engine is not None
            else ChatStructuredEngine()
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
        return messages.build_messages(self, node)

    def default_system_prompt(self, node: Node) -> Prompt:
        """Build a fresh shared system prompt before node customization."""
        profile = self.profile(node.parent_agent)
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
            agent = node.parent_agent
            prompt.add_section(
                "policy.evidence",
                EVIDENCE_POLICY,
                order=150,
                source="Flow.default_system_prompt",
            )
            prompt.add_section(
                "policy.execution",
                EXECUTION_POLICY if can_spawn(agent) else WORKER_ADDENDUM,
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
        profile = self.profile(node.parent_agent)
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
        return messages.render_tools(self, node)

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
        agent = node.parent_agent
        return self._create_pooled_llm(agent.config.model)

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

    async def step(self, node: Node) -> Node:
        """Take one graph step and return the created node."""
        return (await self._drive(node)).created

    async def _drive(self, node: Node) -> StepResult:
        """Queue entry: one step, with infrastructure failure as a terminal node."""
        error: BaseException | None = None
        finalizing = self.budget_exceeded(node)
        if finalizing:
            await self._wait_for_budget_descendants(node.parent_agent)

        with timed(node):
            try:
                if finalizing:
                    landed = FinalQuery()
                    anchor = node
                else:
                    producer, anchor, source = await transitions.resolve(
                        (*self._transitions, *DEFAULT_TRANSITIONS),
                        node,
                        partial(semantic.choose, self),
                    )
                    landed = await producer.produce(self, source)
                anchor.append(landed)
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

        return StepResult(submitted=node, created=landed, error=error)

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
        # and only lands once its ExecAction runs submit(), so stopping at either one
        # spends the tokens and throws the result away. FinalQuery itself must also
        # run even though the threshold has already been crossed.
        if isinstance(node, (FinalQuery, LLMOutput, ExecAction, Submission)):
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
        return delegation.resolve_child(self, action, spec, call_id)

    # -- Running ----------------------------------------------------------

    async def replay(self, root: AgentStart) -> None:
        """Rebuild the namespaces of a graph we did not run, from its recorded code."""
        await restore.replay(self, root)

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
        await restore.resume(self, agents)
        queue = TaskQueue()
        self.queue = queue
        self._active_loop = asyncio.get_running_loop()
        driving = {id(agent) for agent in agents}

        try:
            for agent in agents:
                for leaf in agent.leaves():
                    await self._submit(leaf)

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
                        delegation.submit_child(self, dependent)
                    if not transition.is_agent_start and node.parent_agent.frontier is node:
                        await self._submit(node)
                    for leaf in extra:
                        delegation.submit_leaf(self, leaf)

                if root_error is not None:
                    raise root_error
        finally:
            await queue.cancel()
            if self.queue is queue:
                self.queue = None
                self._active_loop = None
            if close_repls:
                for agent in agents:
                    for node in agent.walk():
                        if isinstance(node, AgentStart):
                            self.runtime.close_repl(node)

    async def _submit(self, leaf: Node) -> None:
        """Queue a leaf of a live agent, restoring its namespace first when lazy."""
        owner = leaf.parent_agent
        if owner is None or owner.terminal:
            return
        if self.restore == "lazy" and isinstance(leaf, ExecAction):
            await restore.ensure_replayed(self, owner)
        delegation.submit_leaf(self, leaf)

    async def arun(
        self,
        root: AgentStart | str,
        *,
        close_repls: bool = False,
        **args: Any,
    ) -> Any:
        if not isinstance(root, str) and args:
            raise TypeError("agent arguments can only be passed with a task string")
        agent = self.start(root, **args) if isinstance(root, str) else root
        async for _node in self.run_streaming(agent, close_repls=close_repls):
            pass
        return agent.result()

    def run(
        self,
        root: AgentStart | str,
        *,
        close_repls: bool = False,
        **args: Any,
    ) -> Any:
        return asyncio.run(self.arun(root, close_repls=close_repls, **args))

    def start(
        self,
        task: str = "",
        *,
        config: AgentConfig | None = None,
        **args: Any,
    ) -> AgentStart:
        """Build a root agent whose keyword arguments become direct REPL variables."""
        agent = build_agent(task, args, config=config or self.root_config)
        agent.config = _normalize_schemas(self.output_codec, agent.config)
        return agent

    async def aclose(self) -> None:
        if self.queue is not None:
            await self.queue.cancel()
            self.queue = None
        owners = {id(item): item for item in (*self._llm_clients.values(), self.structured_engine)}
        closing = [
            close() for item in owners.values() if (close := getattr(item, "aclose", None))
        ]
        try:
            await asyncio.gather(*closing)
        finally:
            self.pool.close()
            self.runtime.close()


__all__ = ["Flow", "StepUntil"]
