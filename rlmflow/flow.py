"""Run agents: prompt the model, execute its code, delegate to children."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from rlmflow import boundaries
from rlmflow.engine.boundaries import StepUntil
from rlmflow.engine.execution import Pool, TaskQueue, ThreadPool, Transition
from rlmflow.graph.nodes import (
    DEFAULT_QUERY,
    AgentConfig,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    Node,
    UserQuery,
    start,
    validate_agent_name,
)
from rlmflow.prompts import DEFAULT_BUILDER, PromptProfile, SystemPromptSource, as_system_prompt_fn
from rlmflow.prompts.messages import (
    COLD_REPL_NOTE,
    CONTINUE_NUDGE,
    FINAL_ANSWER_ACTION,
    TRUNCATION_SUMMARY,
    UserPromptBuilder,
    UserPromptSource,
    as_user_prompt,
    coalesce_roles,
)
from rlmflow.runtime import LocalRuntime, Runtime
from rlmflow.runtime.env import RLMFLOW_REPLAY
from rlmflow.runtime.repl import DoneSignal, Repl, ReplStatus
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.structured import json_schema_for, parse_structured_answer
from rlmflow.tools import RESERVED_TOOLS, tool
from rlmflow.tools.agents import (
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    AgentHandle,
    build_agent_directory,
)
from rlmflow.tools.llm_query import llm_query_batched
from rlmflow.utils.helpers import (
    accepts_kwarg,
    code_block,
    tool_name,
    truncate_output,
    usage_from_client,
)

MAX_ITERS_EXCEEDED = "[max_iters exceeded]"
BUDGET_EXCEEDED = "[budget exceeded]"


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

    final_action = FINAL_ANSWER_ACTION
    continue_nudge = CONTINUE_NUDGE
    truncation_summary = TRUNCATION_SUMMARY
    cold_repl_note = COLD_REPL_NOTE

    def __init__(
        self,
        llm: Any,
        *,
        config: AgentConfig | None = None,
        restore: Literal["replay", "lazy"] = "replay",
        system_prompt: SystemPromptSource | None = None,
        user_prompt: UserPromptSource | None = None,
        prompt_profiles: dict[str, PromptProfile] | None = None,
        prompt_router: Callable[[Flow, AgentStart], str] | None = None,
        tools: list[Any] | None = None,
        runtime: Runtime | None = None,
        llm_clients: dict[str, Any] | None = None,
        llm_request_timeout: float | None = None,
        workers: int | None = None,
        pool: Pool | None = None,
        use_llm_query: bool = False,
        use_agent_tree: bool = False,
        enable_structured_output: bool = True,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        if pool is not None and workers is not None:
            raise ValueError("pass workers or pool, not both")
        self.llm = llm
        self.defaults = config or AgentConfig()
        self.restore = restore
        self.llm_request_timeout = llm_request_timeout
        self.system_prompt = system_prompt or DEFAULT_BUILDER
        self.user_prompt = (
            as_user_prompt(user_prompt) if user_prompt is not None else UserPromptBuilder()
        )
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.use_agent_tree = use_agent_tree
        self.enable_structured_output = enable_structured_output
        self.runtime = runtime or LocalRuntime()
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
        self._restored_agents: set[int] = set()
        self._restore_lock = asyncio.Lock()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.tools: dict[str, Any] = {}
        for fn in tools or []:
            self.add_tool(fn)
        if use_llm_query:
            self.add_tool(llm_query_batched(self), name="llm_query_batched")

    @property
    def max_depth(self) -> int:
        """Read by the shared prompt sections that describe delegation."""
        return self.defaults.max_depth

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
            return PromptProfile(self.system_prompt, self.user_prompt)
        raise ValueError(f"unknown prompt {name!r}")

    def user_builder(self, agent: AgentStart) -> UserPromptBuilder:
        """This agent's user prompt, from its profile or the flow's default.

        Normalized here rather than at construction, since a profile is a plain
        dataclass a caller can hand a bare ``(flow, node) -> str | None``.
        """
        source = self.profile(agent).user
        return self.user_prompt if source is None else as_user_prompt(source)

    def messages(self, node: Node) -> list[dict[str, str]]:
        """The prompt as of ``node``: system message, then that agent's turns."""
        agent = node.parent_agent
        profile = self.profile(agent)
        builder = self.user_builder(agent)
        keep = agent.config.keep_n_messages
        keep = None if keep is None else max(keep, 1)  # a prompt needs a user turn
        # One turn past the limit, so a full prompt can be told from a truncated one.
        turns = builder.project(self, node, keep=None if keep is None else keep + 1)
        if keep is not None and len(turns) > keep:
            turns = [
                {"role": "user", "content": self.truncation_summary},
                *turns[-keep:],
            ]
        system = as_system_prompt_fn(profile.system or self.system_prompt)(self, agent)
        return coalesce_roles([{"role": "system", "content": system}, *turns])

    def prepare_turn(self, node: Node) -> Node:
        """Commit this turn's user content, so the model always answers a user."""
        agent = node.parent_agent
        builder = self.user_builder(agent)
        content = builder.build(self, agent)
        if content:
            node = node.append(UserQuery(content=content))
        if agent.llm_turns() == agent.config.max_iters - 1:
            return node.append(UserQuery(content=self.final_action))
        turns = builder.project(self, node, keep=1)  # only the last turn's role matters
        if not turns or turns[-1]["role"] != "user":
            node = node.append(UserQuery(content=self.continue_nudge))
        return node

    async def call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        *,
        key: object | None = None,
        **kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        if model not in self._llm_clients:
            raise ValueError(f"unknown model {model!r}")
        client = self._llm_clients[model]
        timeout = self.llm_request_timeout
        if timeout is not None:
            if "timeout" not in kwargs and accepts_kwarg(client.chat, "timeout"):
                # Ours only frees the caller: a blocking call keeps its thread no
                # matter what we do, so the client's own timeout ends the request.
                kwargs["timeout"] = timeout
            call = asyncio.wait_for(
                self.pool.call(client.chat, messages, key=key, **kwargs),
                timeout,
            )
        else:
            call = self.pool.call(client.chat, messages, key=key, **kwargs)
        return await call, usage_from_client(client)

    @property
    def llm_query_batched(self):
        """The fan-out tool, bound to this flow — callable with or without opting in."""
        return llm_query_batched(self)

    # -- Steps ------------------------------------------------------------

    async def step(self, node: Node) -> Transition:
        """Take one complete graph step; only cancellation escapes as an exception."""
        error: BaseException | None = None

        with timed(node):
            try:
                if self.budget_exceeded(node):
                    landed = node.append(DoneOutput(result=BUDGET_EXCEEDED))
                elif isinstance(node, (AgentStart, UserQuery, ExecOutput, ErrorOutput)):
                    landed = await self.llm_step(node)
                elif isinstance(node, LLMOutput):
                    landed = node.append(ExecAction(code=node.code))
                elif isinstance(node, ExecAction):
                    landed = await self.exec_step(node)
                else:
                    raise TypeError(f"cannot step {type(node).__name__}")
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

        agent = landed.parent_agent
        if agent is not None and agent.terminal and agent is not agent.root:
            await asyncio.to_thread(self.runtime.close_repl, agent)
        return Transition(submitted=node, created=landed, error=error)

    async def llm_step(self, node: Node) -> Node:
        agent = node.parent_agent
        if agent.llm_turns() >= agent.config.max_iters:
            return node.append(DoneOutput(result=MAX_ITERS_EXCEEDED))
        turn = self.prepare_turn(node)
        messages = self.messages(turn)
        prompt_id = agent.record_prompt(messages[0]["content"])
        reply, usage = await self.call_chat(
            messages,
            agent.config.model,
            key=agent.id,
        )
        return turn.append(
            LLMOutput(content=reply, code=code_block(reply), usage=usage, prompt_id=prompt_id)
        )

    async def exec_step(self, action: ExecAction) -> Node:
        agent = action.parent_agent
        self.runtime.repl_for(agent).seed(self.build_tools(action), agent.config.inputs)
        run = await self.runtime.execute(action, action.code)
        output = truncate_output(run.output, agent.config.max_output_length)
        if run.status is ReplStatus.DONE:
            return action.append(DoneOutput(content=output, result=run.answer))
        if run.status is ReplStatus.OK:
            return action.append(ExecOutput(content=output or "(no output)"))
        if run.status is ReplStatus.ERROR:
            # The agent's own code raised; the traceback is what it needs to read.
            return action.append(ErrorOutput(content=output))
        if run.status is ReplStatus.DEAD:
            # A different failure, and a worse one: the next step opens an empty
            # namespace, so say so rather than leaving it to hit NameErrors.
            text = f"{output}\n{self.cold_repl_note}"
            return action.append(ErrorOutput(content=text, error="repl"))
        raise ValueError(f"unknown repl status {run.status!r}")

    def budget_exceeded(self, node: Node) -> bool:
        limit = node.parent_agent.config.max_budget
        if limit is None:
            return False
        return node.root.tokens().total >= limit

    # -- Tools ------------------------------------------------------------

    def inject(self, name: str, value: Any) -> None:
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
            "done": finish,  # compatibility for saved runs and existing agent code
            "launch_subagent": self.launch_tool(node),
            "asyncio": asyncio,
            "INPUTS": node.parent_agent.config.inputs,
            AGENTS_BINDING: agents,
            AGENT_WAIT_TOOL: self.wait_agent_tool(node),
        }
        if self.use_agent_tree:
            namespace["AGENTS"] = agents
        return namespace

    def wait_agent_tool(self, node: Node):
        @tool("Wait for an existing agent and return its result.", proxy=True)
        async def wait_agent(agent_id: str) -> Any:
            root = node.root
            target = next(
                (
                    candidate
                    for candidate in root.walk()
                    if isinstance(candidate, AgentStart) and candidate.id == agent_id
                ),
                None,
            )
            if target is None:
                raise KeyError(f"unknown agent {agent_id!r}")
            if not target.terminal:
                queue = self.queue
                if queue is None:
                    raise RuntimeError("waiting for an agent requires an active stream")
                await queue.join(target)
            return target.result()

        return wait_agent

    def finish_tool(self, node: Node):
        schema = node.parent_agent.config.output_schema

        @tool("Submit this agent's final answer and end its run.", proxy=True)
        def finish(answer: object) -> None:
            if schema is None:
                raise DoneSignal(str(answer))
            # Carry the parsed value, so the run records what callers receive.
            raise DoneSignal(parse_structured_answer(answer, schema))

        return finish

    def launch_tool(self, node: Node):
        mutation_lock = asyncio.Lock()

        @tool(
            "Spawn or resume one named child agent.",
            proxy=True,
        )
        async def launch_subagent(
            goal: str,
            *,
            name: str | None = None,
            inputs: dict[str, str] | None = None,
            model: str | None = None,
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

        queue = TaskQueue(self.pool)
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

    async def arun(self, root: AgentStart | str) -> Any:
        agent = self.start(root) if isinstance(root, str) else root
        async for _node in self.run_streaming(agent):
            pass
        return agent.result()

    def run(self, root: AgentStart | str) -> Any:
        return asyncio.run(self.arun(root))

    def start(self, query: str = "", **overrides: Any) -> AgentStart:
        """A root agent carrying this flow's defaults, which keyword overrides win against.

        The module-level ``start`` does the building; this only supplies the defaults
        the flow was constructed with, so ``Flow(config=...)`` reaches the roots you
        run on it.
        """
        return start(query, config=self.defaults, **overrides)

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


__all__ = ["Flow", "StepUntil", "code_block", "start"]
