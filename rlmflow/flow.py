"""Run minimal agents: prompt the model, execute its code, delegate to children."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
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
from rlmflow.structured import json_schema_for, parse_structured_output
from rlmflow.tools import RESERVED_TOOLS, tool
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
        self.enable_structured_output = enable_structured_output
        self.runtime = runtime or LocalRuntime()
        self.pool = pool or ThreadPool(workers)
        self.queue: TaskQueue | None = None
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
        repl = self.runtime.repl_for(agent)
        repl.seed(self.build_tools(action, repl), agent.config.inputs)
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

    def build_tools(self, node: Node, repl: Repl | None = None) -> dict[str, Any]:
        # Called without a repl to describe the tools in a prompt.
        repl = repl or SimpleNamespace(done_result=None)
        return {
            **self.tools,
            "done": self.done_tool(node, repl),
            "launch_subagents": self.launch_tool(node),
            "INPUTS": node.parent_agent.config.inputs,
        }

    def done_tool(self, node: Node, repl: Repl):
        schema = node.parent_agent.config.output_schema

        @tool("Submit this agent's final answer and end its run.", proxy=True)
        def done(answer: object) -> None:
            if schema is None:
                repl.done_result = str(answer)
            else:
                encoded = answer if isinstance(answer, str) else json.dumps(answer)
                # Keep the parsed value, so the run records what callers receive.
                repl.done_result = parse_structured_output(encoded, schema)
            raise DoneSignal

        return done

    def launch_tool(self, node: Node):
        @tool(
            "Spawn or resume child agents and await their results.",
            proxy=True,
        )
        async def launch_subagents(specs: list[dict[str, Any]]) -> list[Any]:
            if not isinstance(node, ExecAction):
                raise TypeError("launch_subagents requires an ExecAction")

            names = [spec.get("name") or f"child{index}" for index, spec in enumerate(specs)]
            if len(names) != len(set(names)):
                raise ValueError("duplicate child names in one launch_subagents call")

            existing = {id(child) for child in node.children if isinstance(child, AgentStart)}
            resolved = [self.resolve_child(node, spec, index) for index, spec in enumerate(specs)]
            children = [value for value in resolved if isinstance(value, AgentStart)]
            created = [child for child in children if id(child) not in existing]
            await self.run_children(children, submit=created)
            return [
                value.result() if isinstance(value, AgentStart) else value for value in resolved
            ]

        return launch_subagents

    def resolve_child(
        self,
        action: ExecAction,
        spec: dict[str, Any],
        index: int,
    ) -> AgentStart | str:
        """Resolve one launch spec to a direct child or refusal."""
        name = spec.get("name") or f"child{index}"
        validate_agent_name(name)

        existing = next(
            (
                child
                for child in action.children
                if isinstance(child, AgentStart) and child.config.name == name
            ),
            None,
        )
        if existing is not None:
            return existing

        parent = action.parent_agent
        if any(child.config.name == name for child in parent.sub_agents):
            raise ValueError(f"duplicate child name {name!r}")

        query = spec.get("query", "")
        if parent.config.depth >= parent.config.max_depth:
            return f"[refused: max depth {parent.config.max_depth}]"
        if len(query) > parent.config.max_query_chars:
            return f"[refused: query too long ({len(query)} chars)]"
        return self.new_child(action, name, spec)

    async def run_children(
        self,
        children: list[AgentStart],
        *,
        submit: list[AgentStart],
    ) -> None:
        """Submit newly created children and await every unfinished child."""
        unfinished = [child for child in children if not child.terminal]
        if not unfinished:
            return

        queue = self.queue
        if queue is None:
            raise RuntimeError("launch_subagents requires an active stream")
        for child in submit:
            if child.terminal:
                continue
            for leaf in child.leaves():
                owner = leaf.parent_agent
                if owner is not None and not owner.terminal:
                    queue.submit(
                        leaf,
                        self.step,
                        publish=isinstance(leaf, AgentStart),
                    )
        await asyncio.gather(*(queue.join(child) for child in unfinished))

    def new_child(self, node: Node, name: str, spec: dict[str, Any]) -> AgentStart:
        """Open a child agent of ``node``'s agent, attached to ``node``."""
        schema = spec.get("output_schema")
        overrides = {
            key: value
            for key, value in {
                "inputs": dict(spec.get("inputs") or {}),
                "model": spec.get("model"),
                "prompt_profile": spec.get("prompt_profile"),
                "output_schema": (json_schema_for(schema) if schema is not None else None),
            }.items()
            if value is not None
        }
        child = AgentStart(
            content=spec.get("query") or DEFAULT_QUERY,
            config=node.parent_agent.config.child(name, **overrides),
        )
        return node.append(child)

    # -- Running ----------------------------------------------------------

    async def replay(self, root: AgentStart) -> None:
        """Rebuild the namespaces of a graph we did not run, from its recorded code.

        Appends nothing: ``launch_subagents`` resolves the children already attached
        to each action and reads finished results from their terminal nodes. A block
        that failed the first time fails the same way here, leaving the same partial
        bindings, which is why output and errors are discarded.
        """
        for node in root.walk():
            if not isinstance(node, ExecAction) or node is node.parent_agent.frontier:
                continue  # the frontier action has not run yet; the first step runs it
            agent = node.parent_agent
            if agent.terminal:
                continue  # it answered; nothing will run in this namespace again
            repl = self.runtime.repl_for(agent)
            repl.seed(self.build_tools(node, repl), agent.config.inputs)
            repl.update_env({RLMFLOW_REPLAY: "1"})
            try:
                await self.runtime.execute(node, node.code)
            finally:
                repl.update_env({RLMFLOW_REPLAY: "0"})

    def note_cold(self, root: AgentStart) -> None:
        """Tell every unfinished agent that ran code that its namespace is gone."""
        for leaf in root.leaves():
            if isinstance(leaf, DoneOutput):
                continue
            if any(isinstance(node, ExecAction) for node in leaf.walk(reverse=True)):
                leaf.append(UserQuery(content=self.cold_repl_note))

    async def run_streaming(
        self,
        root: AgentStart | str,
        *roots: AgentStart | str,
        until: StepUntil = "done",
        close_repls: bool = False,
    ) -> AsyncIterator[Node]:
        """Drive one or more roots through one graph-agnostic queue."""
        agents = [self.new_root(item) if isinstance(item, str) else item for item in (root, *roots)]
        if self.queue is not None:
            raise RuntimeError("this Flow is already driving a stream")
        if len({id(agent) for agent in agents}) != len(agents):
            raise RuntimeError("the same root cannot be driven twice")

        boundary = boundaries.resolve(until)
        for agent in agents:
            if self.runtime.get(agent) is None:  # a graph this Flow has not run
                (await self.replay(agent) if self.restore == "replay" else self.note_cold(agent))
            else:
                for node in agent.walk():
                    if (
                        isinstance(node, AgentStart)
                        and not node.terminal
                        and self.runtime.get(node) is None
                    ):
                        (
                            await self.replay(node)
                            if self.restore == "replay"
                            else self.note_cold(node)
                        )

        queue = TaskQueue(self.pool)
        self.queue = queue
        active = {id(agent) for agent in agents}

        try:
            for agent in agents:
                for leaf in agent.leaves():
                    owner = leaf.parent_agent
                    if owner is not None and not owner.terminal:
                        queue.submit(leaf, self.step)

            while active and queue:
                transition = await queue.next()
                node = transition.created
                root = node.root
                if root is None or id(root) not in active:
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
                if stop or root.terminal:
                    active.discard(id(root))
                    await queue.cancel(root.walk())
                elif not transition.is_agent_start and not node.parent_agent.terminal:
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
        agent = self.new_root(root) if isinstance(root, str) else root
        async for _node in self.run_streaming(agent):
            pass
        return agent.result()

    def run(self, root: AgentStart | str) -> Any:
        return asyncio.run(self.arun(root))

    def new_root(self, query: str) -> AgentStart:
        return AgentStart(content=query or DEFAULT_QUERY, config=deepcopy(self.defaults))

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
