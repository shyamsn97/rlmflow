"""Run in-memory agents through model, REPL, and delegation steps."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from rlmflow.execution import CurrentResults, Pool, TaskQueue, ThreadPool
from rlmflow.graph import (
    DEFAULT_MAX_QUERY_CHARS,
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
    SupervisingOutput,
    UserQuery,
    boundaries,
    start,
    validate_agent_name,
)
from rlmflow.prompts import (
    DEFAULT_BUILDER,
    PromptProfile,
    SystemPromptSource,
    as_system_prompt_fn,
)
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
from rlmflow.runtime.repl import DoneSignal, Repl, ReplStatus
from rlmflow.structured import json_schema_for, parse_structured_output
from rlmflow.tools import RESERVED_TOOLS, tool
from rlmflow.tools.llm_query import llm_query_batched
from rlmflow.utils import (
    accepts_kwarg,
    code_block,
    tool_name,
    truncate_output,
    usage_from_client,
)

StepUntil = boundaries.StepUntil
MAX_ITERS_EXCEEDED = "[max_iters exceeded]"
BUDGET_EXCEEDED = "[budget exceeded]"


class Flow:
    """Own resources and execute fresh in-memory agent trees."""

    final_action = FINAL_ANSWER_ACTION
    continue_nudge = CONTINUE_NUDGE
    truncation_summary = TRUNCATION_SUMMARY

    def __init__(
        self,
        llm: Any,
        *,
        max_depth: int = 2,
        max_iters: int = 20,
        child_max_iters: int | None = None,
        keep_n_messages: int | None = None,
        max_output_length: int = 4_000,
        max_query_chars: int = DEFAULT_MAX_QUERY_CHARS,
        max_budget: int | None = None,
        llm_request_timeout: float | None = None,
        system_prompt: SystemPromptSource | None = None,
        user_prompt: UserPromptSource | None = None,
        prompt_profiles: dict[str, PromptProfile] | None = None,
        prompt_router: Callable[[Flow, AgentStart], str] | None = None,
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
        self.keep_n_messages = keep_n_messages
        self.max_output_length = max_output_length
        self.max_query_chars = max_query_chars
        self.max_budget = max_budget
        self.llm_request_timeout = llm_request_timeout
        self.defaults = AgentConfig(
            max_depth=max_depth,
            max_iters=max_iters,
            child_max_iters=child_max_iters,
            max_budget=max_budget,
            keep_n_messages=keep_n_messages,
            max_output_length=max_output_length,
            max_query_chars=max_query_chars,
        )

        self.system_prompt = system_prompt or DEFAULT_BUILDER
        self.user_prompt = (
            as_user_prompt(user_prompt) if user_prompt is not None else UserPromptBuilder()
        )
        self.prompt_profiles = {
            name: PromptProfile(
                system=profile.system,
                user=as_user_prompt(profile.user) if profile.user is not None else None,
                description=profile.description,
            )
            for name, profile in (prompt_profiles or {}).items()
        }
        self.prompt_router = prompt_router
        self.enable_structured_output = enable_structured_output

        self.runtime = runtime or LocalRuntime()
        self.pool = pool or ThreadPool(workers)
        self.tasks = TaskQueue()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.tools: dict[str, Any] = {}
        for fn in tools or []:
            self.add_tool(fn)
        if use_llm_query:
            self.add_tool(llm_query_batched(self), name="llm_query_batched")

    @property
    def repls(self) -> dict[str, Repl]:
        return self.runtime.repls

    # -- Prompt and model -------------------------------------------------

    def _profile(self, agent: AgentStart) -> PromptProfile:
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

    def messages(self, agent: AgentStart) -> list[dict[str, str]]:
        profile = self._profile(agent)
        builder = profile.user or self.user_prompt
        keep = agent.config.keep_n_messages
        keep = None if keep is None else max(keep, 1)
        turns = builder.project(self, agent.frontier, keep=None if keep is None else keep + 1)
        if keep is not None and len(turns) > keep:
            turns = [
                {"role": "user", "content": self.truncation_summary},
                *turns[-keep:],
            ]
        system = as_system_prompt_fn(profile.system or self.system_prompt)(self, agent)
        return coalesce_roles([{"role": "system", "content": system}, *turns])

    def prepare_turn(
        self,
        agent: AgentStart,
        current_results: CurrentResults | None = None,
    ) -> None:
        profile = self._profile(agent)
        builder = profile.user or self.user_prompt
        content = builder.build(self, agent)
        if content:
            agent.tail().submit(UserQuery(content=content), current_results)
        if agent.llm_turns() == agent.config.max_iters - 1:
            agent.tail().submit(UserQuery(content=self.final_action), current_results)
            return
        turns = builder.project(self, agent.frontier, keep=1)
        if not turns or turns[-1]["role"] != "user":
            agent.tail().submit(UserQuery(content=self.continue_nudge), current_results)

    async def call_chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        **kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        if model not in self._llm_clients:
            raise ValueError(f"unknown model {model!r}")
        client = self._llm_clients[model]
        if (
            self.llm_request_timeout is not None
            and "timeout" not in kwargs
            and not inspect.iscoroutinefunction(client.chat)
            and accepts_kwarg(client.chat, "timeout")
        ):
            kwargs["timeout"] = self.llm_request_timeout
        call = self.pool.call(client.chat, messages, **kwargs)
        if self.llm_request_timeout is not None:
            call = asyncio.wait_for(call, self.llm_request_timeout)
        return await call, usage_from_client(client)

    @property
    def llm_query_batched(self):
        return llm_query_batched(self)

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

    def tool_namespace_for_prompt(self, agent: AgentStart) -> dict[str, Any]:
        repl = self.runtime.namespace_for(agent)
        return repl if repl is not None else self.build_tools(agent)

    def build_tools(
        self,
        node: Node,
        repl: Repl | None = None,
        current_results: CurrentResults | None = None,
    ) -> dict[str, Any]:
        if node.agent is None:
            raise RuntimeError("node is detached")
        repl = repl or SimpleNamespace(done_result=None, read=lambda *, clear=False: "")
        return {
            **self.tools,
            "done": self.done_tool(node.agent, repl),
            "launch_subagents": self.launch_subagents(node, repl, current_results),
            "INPUTS": node.agent.config.inputs,
        }

    def done_tool(self, agent: AgentStart, repl: Repl):
        @tool("Submit this agent's final answer and end its run.", proxy=True)
        def done_tool(answer: object) -> None:
            schema = agent.config.output_schema
            if schema is not None:
                encoded = answer if isinstance(answer, str) else json.dumps(answer)
                parse_structured_output(encoded, schema)
                repl.done_result = encoded
            else:
                repl.done_result = str(answer)
            print(f"[done] {repl.done_result}")
            raise DoneSignal()

        return done_tool

    # -- Running ----------------------------------------------------------

    async def step(
        self,
        node: Node,
        current_results: CurrentResults | None = None,
    ) -> Node:
        if node.agent is None:
            raise RuntimeError("node is detached")
        agent = node.agent
        if node is not agent.tail():
            raise ValueError("step requires the current tail")
        if self._budget_exceeded(agent):
            return node.submit(DoneOutput(result=BUDGET_EXCEEDED), current_results)
        if isinstance(node, (AgentStart, UserQuery, ExecOutput, ErrorOutput)):
            return await self.llm_step(node, current_results)
        if isinstance(node, LLMOutput):
            return node.submit(ExecAction(code=node.code), current_results)
        if isinstance(node, ExecAction):
            return await self.exec_step(node, current_results)
        raise TypeError(f"cannot step {type(node).__name__}")

    async def llm_step(
        self,
        node: Node,
        current_results: CurrentResults | None = None,
    ) -> Node:
        assert node.agent is not None
        agent = node.agent
        if agent.llm_turns() >= agent.config.max_iters:
            return node.submit(DoneOutput(result=MAX_ITERS_EXCEEDED), current_results)
        self.prepare_turn(agent, current_results)
        reply, usage = await self.call_chat(self.messages(agent), agent.config.model)
        return agent.tail().submit(
            LLMOutput(content=reply, code=code_block(reply), usage=usage),
            current_results,
        )

    async def exec_step(
        self,
        action: ExecAction,
        current_results: CurrentResults | None = None,
    ) -> Node:
        assert action.agent is not None
        agent = action.agent
        repl = self.runtime.repl_for(agent)
        repl.seed(self.build_tools(action, repl, current_results), agent.config.inputs)
        run = await self.runtime.execute(action, action.code)
        output = truncate_output(run.output, agent.config.max_output_length)
        if run.status is ReplStatus.DONE:
            produced: Node = DoneOutput(content=output, result=run.answer)
        elif run.status is ReplStatus.OK:
            produced = ExecOutput(content=output or "(no output)")
        elif run.status is ReplStatus.ERROR:
            produced = ErrorOutput(content=output)  # the agent's code raised
        elif run.status is ReplStatus.DEAD:  # the REPL did, so its namespace is gone
            produced = ErrorOutput(content=f"{output}\n{COLD_REPL_NOTE}", error="repl")
        else:
            raise ValueError(f"unknown repl status {run.status!r}")
        return agent.tail().submit(produced, current_results)

    async def run_agent(
        self,
        agent: AgentStart,
        current_results: CurrentResults | None = None,
    ) -> Any:
        while not isinstance(agent.tail(), DoneOutput):
            await self.step(agent.tail(), current_results)
        return agent.result()

    async def _run_child(
        self,
        agent: AgentStart,
        current_results: CurrentResults | None,
    ) -> None:
        try:
            await self.run_agent(agent, current_results)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - child failures are values
            text = f"{type(exc).__name__}: {exc}"
            agent.tail().submit(
                DoneOutput(content=text, result=f"[child failed: {text}]"),
                current_results,
            )

    async def _drive(
        self,
        agent: AgentStart,
        current_results: CurrentResults,
    ) -> None:
        while not isinstance(agent.tail(), DoneOutput):
            await self.step(agent.tail(), current_results)
            if current_results.stopped:
                return

    async def run_streaming(
        self,
        root: AgentStart | str,
        *,
        until: StepUntil = "done",
        close_repls: bool = False,
    ) -> AsyncIterator[Node]:
        created = isinstance(root, str)
        agent = self._new_root(root) if created else root
        boundary = boundaries.resolve(until)
        if created:
            yield agent
            if boundary is not None and boundary(agent, agent.root):
                if close_repls:
                    self.runtime.close_repl(agent)
                return
        current_results = CurrentResults(
            stop=(None if boundary is None else lambda node: boundary(node, agent.root))
        )
        self.tasks.submit(
            self._drive(agent, current_results),
            current_results,
            key=agent.root.config.id,
        )
        try:
            async for node in current_results:
                yield node
            await current_results.wait()
        finally:
            await current_results.close()
            if close_repls:
                for child in agent.agents():
                    self.runtime.close_repl(child)

    async def arun(self, root: AgentStart | str) -> Any:
        agent = self._new_root(root) if isinstance(root, str) else root
        async for _ in self.run_streaming(agent):
            pass
        return agent.result()

    def run(self, root: AgentStart | str) -> Any:
        return asyncio.run(self.arun(root))

    # -- Delegation -------------------------------------------------------

    def launch_subagents(
        self,
        parent: Node,
        repl: Repl,
        current_results: CurrentResults | None = None,
    ):
        if parent.agent is None:
            raise RuntimeError("node is detached")

        @tool("Spawn child agents from specs and await their results.", proxy=True)
        async def launch(specs: list[dict[str, Any]]) -> list[Any]:
            agent = parent.agent
            results: list[Any] = [None] * len(specs)
            accepted: list[tuple[int, str, dict[str, Any]]] = []
            names = {child.config.name for child in agent.child_agents()}
            for index, spec in enumerate(specs):
                name = spec.get("name", f"child{index}")
                validate_agent_name(name)
                if name in names:
                    raise ValueError(f"duplicate child name {name!r}")
                names.add(name)
                query = spec.get("query", "")
                if agent.config.depth >= agent.config.max_depth:
                    results[index] = f"[refused: max depth {agent.config.max_depth}]"
                    continue
                if len(query) > agent.config.max_query_chars:
                    results[index] = f"[refused: query too long ({len(query)} chars)]"
                    continue
                accepted.append((index, name, spec))

            supervisor = agent.tail().submit(
                SupervisingOutput(content=repl.read(clear=True)),
                current_results,
            )
            assert isinstance(supervisor, SupervisingOutput)
            children = [
                (index, self.new_child(supervisor, name, spec, current_results))
                for index, name, spec in accepted
            ]
            await self.tasks.run_all(
                self._run_child(child, current_results) for _, child in children
            )
            for index, child in children:
                results[index] = self.child_result(child)
            return results

        return launch

    def new_child(
        self,
        supervisor: SupervisingOutput,
        name: str,
        spec: dict[str, Any],
        current_results: CurrentResults | None = None,
    ) -> AgentStart:
        assert supervisor.agent is not None
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
            content=spec.get("query", "") or DEFAULT_QUERY,
            config=supervisor.agent.config.child(name, **overrides),
        )
        return supervisor.spawn(child, current_results)

    def child_result(self, child: AgentStart) -> Any:
        value = child.result()
        schema = child.config.output_schema
        if schema is None:
            return value
        return parse_structured_output(str(value), schema)

    # -- Resources --------------------------------------------------------

    def _new_root(self, query: str) -> AgentStart:
        return AgentStart(content=query or DEFAULT_QUERY, config=deepcopy(self.defaults))

    def _budget_exceeded(self, agent: AgentStart) -> bool:
        limit = agent.config.max_budget
        return limit is not None and sum(agent.tokens(recursive=True)) >= limit

    async def aclose(self) -> None:
        await self.tasks.close()
        self.runtime.close()
        self.pool.close()


__all__ = ["Flow", "StepUntil", "code_block", "start"]
