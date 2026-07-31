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

from rlmflow.minimal import boundaries
from rlmflow.minimal.boundaries import StepUntil
from rlmflow.minimal.nodes import (
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
from rlmflow.minimal.task import TaskQueue
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
from rlmflow.runtime.env import RLMFLOW_REPLAY
from rlmflow.runtime.repl import DoneSignal, Repl, ReplStatus
from rlmflow.structured import json_schema_for, parse_structured_output
from rlmflow.tools import RESERVED_TOOLS, tool
from rlmflow.utils.helpers import (
    call_sync_or_async,
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
        enable_structured_output: bool = True,
    ) -> None:
        if restore not in ("replay", "lazy"):
            raise ValueError(f"restore must be 'replay' or 'lazy', not {restore!r}")
        self.llm = llm
        self.defaults = config or AgentConfig()
        self.restore = restore
        self.system_prompt = system_prompt or DEFAULT_BUILDER
        self.user_prompt = (
            as_user_prompt(user_prompt) if user_prompt is not None else UserPromptBuilder()
        )
        self.prompt_profiles = dict(prompt_profiles or {})
        self.prompt_router = prompt_router
        self.enable_structured_output = enable_structured_output
        self.runtime = runtime or LocalRuntime()
        self.queue = TaskQueue()
        self._llm_clients = {"default": llm, **(llm_clients or {})}
        self.tools: dict[str, Any] = {}
        for fn in tools or []:
            self.add_tool(fn)

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

    def messages(self, node: Node) -> list[dict[str, str]]:
        """The prompt as of ``node``: system message, then that agent's turns."""
        agent = node.parent_agent
        profile = self.profile(agent)
        builder = profile.user or self.user_prompt
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
        builder = self.profile(agent).user or self.user_prompt
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
        self, messages: list[dict[str, str]], model: str = "default"
    ) -> tuple[str, LLMUsage]:
        if model not in self._llm_clients:
            raise ValueError(f"unknown model {model!r}")
        client = self._llm_clients[model]
        reply = await call_sync_or_async(client.chat, messages)
        return reply, usage_from_client(client)

    # -- Steps ------------------------------------------------------------

    async def step(self, node: Node) -> Node:
        """Take one step from this node. Each branch lands what it produced."""
        with timed(node):
            if self.budget_exceeded(node):
                return node.append(DoneOutput(result=BUDGET_EXCEEDED))
            if isinstance(node, (AgentStart, UserQuery, ExecOutput, ErrorOutput)):
                return await self.llm_step(node)
            if isinstance(node, LLMOutput):
                return node.append(ExecAction(code=node.code))
            if isinstance(node, ExecAction):
                return await self.exec_step(node)
        raise TypeError(f"cannot step {type(node).__name__}")

    async def llm_step(self, node: Node) -> Node:
        agent = node.parent_agent
        if agent.llm_turns() >= agent.config.max_iters:
            return node.append(DoneOutput(result=MAX_ITERS_EXCEEDED))
        turn = self.prepare_turn(node)
        messages = self.messages(turn)
        prompt_id = agent.record_prompt(messages[0]["content"])
        reply, usage = await self.call_chat(messages, agent.config.model)
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
        spent = sum(
            spent.usage.input_tokens + spent.usage.output_tokens
            for spent in node.root.walk()
            if isinstance(spent, LLMOutput)
        )
        return spent >= limit

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
            "launch_subagents": self.launch_tool(node, repl),
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

    def launch_tool(self, node: Node, repl: Repl):
        agent = node.parent_agent

        @tool("Spawn child agents from specs and await their results.", proxy=True)
        async def launch_subagents(specs: list[dict[str, Any]]) -> list[Any]:
            answers: list[Any] = [None] * len(specs)
            accepted: list[tuple[int, str, dict[str, Any]]] = []
            names = {child.config.name for child in agent.sub_agents}
            answered = self.recorded_answers(agent, repl)
            for index, spec in enumerate(specs):
                name = spec.get("name") or f"child{index}"
                validate_agent_name(name)
                if name in answered:
                    answers[index] = answered[name]  # this launch already happened
                    continue
                if name in names:
                    raise ValueError(f"duplicate child name {name!r}")
                names.add(name)
                query = spec.get("query", "")
                if agent.config.depth >= agent.config.max_depth:
                    answers[index] = f"[refused: max depth {agent.config.max_depth}]"
                elif len(query) > agent.config.max_query_chars:
                    answers[index] = f"[refused: query too long ({len(query)} chars)]"
                else:
                    accepted.append((index, name, spec))
            if not accepted:
                return answers

            # Children hang off the action that launched them, which keeps the
            # parent's frontier where it was: this step has not landed yet.
            children = [(index, self.new_child(node, name, spec)) for index, name, spec in accepted]
            # Start them here: this step cannot finish until they answer, so the
            # run loop will not get a turn to pick them up on its own.
            for _index, child in children:
                self.submit(child)
            for index, child in children:
                answers[index] = await self.child_answer(child)
            return answers

        return launch_subagents

    def recorded_answers(self, agent: AgentStart, repl: Repl) -> dict[str, Any]:
        """What this agent's children already answered, by name — replay only.

        A finished child *is* its answer, so a replay of the code that launched it
        needs no cache: it reads the same value back through the same accessor the
        live run used. Empty during a live turn, where relaunching a name is an error
        rather than a repeat.
        """
        if repl.env.get(RLMFLOW_REPLAY) != "1":
            return {}
        return {child.config.name: child.result() for child in agent.sub_agents if child.terminal}

    def new_child(self, node: Node, name: str, spec: dict[str, Any]) -> AgentStart:
        """Open a child agent of ``node``'s agent, attached to ``node``."""
        schema = spec.get("output_schema")
        overrides = {
            key: value
            for key, value in {
                "inputs": dict(spec.get("inputs") or {}),
                "model": spec.get("model"),
                "prompt_profile": spec.get("prompt_profile"),
                "output_schema": json_schema_for(schema) if schema is not None else None,
            }.items()
            if value is not None
        }
        child = AgentStart(
            content=spec.get("query") or DEFAULT_QUERY,
            config=node.parent_agent.config.child(name, **overrides),
        )
        return node.append(child)

    async def child_answer(self, child: AgentStart) -> Any:
        """Wait for one child; a crash becomes its answer instead of the parent's."""
        try:
            await self.queue.result(child)
        except Exception as exc:  # noqa: BLE001 - child failures are values
            text = f"{type(exc).__name__}: {exc}"
            child.frontier.append(DoneOutput(content=text, result=f"[child failed: {text}]"))
        return child.result()

    # -- Running ----------------------------------------------------------

    def submit(self, node: Node) -> asyncio.Task[Node]:
        """Queue one step from this node, tagged with the node it steps.

        Work in flight is keyed by node, results by agent: a step is taken from
        somewhere, an answer belongs to someone.
        """
        return self.queue.submit(node, self.step, node)

    async def replay(self, root: AgentStart) -> None:
        """Rebuild the namespaces of a graph we did not run, from its recorded code.

        Appends nothing: ``launch_subagents`` reads recorded answers while
        ``RLMFLOW_REPLAY`` is set, so no child is launched and no node lands. A block
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

    def submit_leaves(self, root: AgentStart) -> None:
        """Step every leaf in the tree that is not already being stepped.

        A leaf is a step waiting to be taken unless its agent has answered — which
        its frontier says, or the queue does for the one outcome a graph cannot hold:
        a step that raised leaves no ``DoneOutput`` to read.
        """
        for leaf in root.leaves():
            agent = leaf.parent_agent
            if agent is None or agent.terminal or self.queue.completed(agent):
                continue
            if not self.queue.busy(leaf):
                self.submit(leaf)

    async def run_streaming(
        self,
        root: AgentStart | str,
        *,
        until: StepUntil = "done",
        close_repls: bool = False,
    ) -> AsyncIterator[Node]:
        """Step the tree's leaves until the boundary, yielding nodes as they land."""
        agent = self.new_root(root) if isinstance(root, str) else root
        boundary = boundaries.resolve(until)
        if self.runtime.get(agent) is None:  # a graph this Flow has not run
            await self.replay(agent) if self.restore == "replay" else self.note_cold(agent)
        seen = {node.id for node in agent.walk()}

        def mine(stepping: Node | None) -> bool:
            return stepping is not None and stepping.root is agent

        try:
            while True:
                self.submit_leaves(agent)
                if not self.queue.pending(mine):
                    break
                for stepped, handle in await self.queue.wave():
                    if not handle.cancelled() and handle.exception() is not None:
                        # A step that raised is its agent's outcome; it lands nothing.
                        self.queue.complete(stepped.parent_agent, error=handle.exception())
                stop = False
                for node in self.landed(agent, seen):
                    owner = node.parent_agent
                    if isinstance(node, DoneOutput) and owner is not None:
                        self.queue.complete(owner, owner.result())
                    yield node
                    stop = stop or (boundary is not None and boundary(node, agent))
                if stop or self.queue.completed(agent):
                    break
            if self.queue.completed(agent):
                # A crashed root step is the caller's problem, not a result.
                await self.queue.result(agent)
        finally:
            await self.queue.cancel(self.queue.pending(mine))
            for node in agent.walk():
                if isinstance(node, AgentStart):
                    self.queue.forget(node)
                    if close_repls:
                        self.runtime.close_repl(node)

    def landed(self, root: AgentStart, seen: set[str]) -> list[Node]:
        """The nodes attached since the last look, in tree order."""
        fresh = [node for node in root.walk() if node.id not in seen]
        seen.update(node.id for node in fresh)
        return fresh

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
        await self.queue.close()
        self.runtime.close()


__all__ = ["Flow", "StepUntil", "code_block", "start"]
