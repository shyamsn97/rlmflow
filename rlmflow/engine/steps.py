"""Replaceable class-based graph step functions."""

from __future__ import annotations

from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    LOCAL_PLANNING_ACTION,
    PLANNING_ACTION,
    AgentStart,
    ContinueQuery,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    LLMOutput,
    Node,
    PlanQuery,
    ReplDead,
    TruncationSummary,
    UserQuery,
)
from rlmflow.llm import LLMClient, join_chunks
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.runtime.runtime import WrappedRuntime
from rlmflow.utils.helpers import code_block, truncate_output

CONTROL_QUERIES = (
    PlanQuery,
    FinalQuery,
    ContinueQuery,
    TruncationSummary,
)
MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


def at_final(agent: AgentStart) -> bool:
    limit = agent.config.max_iters
    return limit is not None and agent.llm_turns() == limit - 1


def budget_nearly_spent(node: Node) -> bool:
    """Whether one more turn of the size seen so far would exhaust the token budget.

    The iteration limit warns at ``max_iters - 1``, leaving a turn to answer in. The
    token budget had no such warning: it ended the run mid-flight with no answer,
    which is how a task that burned 97k of its 100k tokens scored zero while holding
    13 unused iterations. Averaging turns so far avoids a fixed reserve, which is
    either too small for a long history or wasted on a short one.
    """
    limit = node.parent_agent.config.max_budget
    root = node.root
    if limit is None or root is None:
        return False
    turns = root.stats.node_counts.get(LLMOutput.type, 0)
    if turns == 0:
        return False
    spent = root.usage.total
    return spent + spent / turns >= limit


def needs_truncation(node: Node) -> bool:
    if isinstance(node, TruncationSummary):
        return False
    keep = node.parent_agent.config.keep_n_messages
    if keep is None:
        return False
    keep = max(keep, 1)
    return len(node.project(keep=keep + 1)) > keep


class MessageBuilder:
    """Assemble the chat the model sees at a node."""

    def build(self, node: Node) -> list[dict[str, str]]:
        return list(node.project())


class StepFunction:
    """One graph transition: an LLM, a message builder, and a wrapped runtime."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        messages: MessageBuilder,
        runtime: WrappedRuntime,
    ) -> None:
        self.llm = llm
        self.messages = messages
        self.runtime = runtime

    async def __call__(self, node: Node) -> Node:
        raise NotImplementedError


class LLMRequestStep(StepFunction):
    """Order all control nodes before the next model call."""

    def plan_query(self, node: Node) -> PlanQuery:
        agent = node.parent_agent
        can_spawn = agent.config.max_depth > 0 and agent.config.depth < agent.config.max_depth
        content = PLANNING_ACTION if can_spawn else LOCAL_PLANNING_ACTION
        return PlanQuery(content=content)

    async def __call__(self, node: Node) -> Node:
        if isinstance(node, CONTROL_QUERIES):
            return await self.chat(node)

        if isinstance(node, AgentStart):
            return node.append(self.plan_query(node))

        agent = node.parent_agent
        if at_final(agent) or budget_nearly_spent(node):
            return node.append(FinalQuery())

        if type(node) is UserQuery and isinstance(node.prev, DoneOutput):
            return node.append(self.plan_query(node))

        if needs_truncation(node):
            return node.append(TruncationSummary())

        messages = self.messages.build(node)
        if not messages or messages[-1]["role"] != "user":
            return node.append(ContinueQuery())

        return await self.chat(node, messages=messages)

    async def chat(
        self,
        node: Node,
        *,
        messages: list[dict[str, str]] | None = None,
    ) -> Node:
        agent = node.parent_agent
        limit = agent.config.max_iters
        if limit is not None and agent.llm_turns() >= limit:
            return node.append(DoneOutput(result=MAX_ITERS_EXCEEDED))

        messages = self.messages.build(node) if messages is None else messages
        reply, usage = await join_chunks(self.llm.stream(messages))
        return node.append(
            LLMOutput(
                content=reply,
                code=code_block(reply),
                usage=usage,
                prompt_id=agent.record_prompt(messages[0]["content"]),
            )
        )


class LLMOutputStep(StepFunction):
    async def __call__(self, node: Node) -> Node:
        if not isinstance(node, LLMOutput):
            raise TypeError(f"LLMOutputStep expected LLMOutput, got {type(node).__name__}")
        return node.append(ExecAction(code=node.code))


class ExecActionStep(StepFunction):
    async def __call__(self, node: Node) -> Node:
        if not isinstance(node, ExecAction):
            raise TypeError(f"ExecActionStep expected ExecAction, got {type(node).__name__}")
        run = await self.runtime.execute(node)
        return append_run_result(node, run)


def append_run_result(node: ExecAction, run: ReplRun) -> Node:
    output = truncate_output(
        run.output,
        node.parent_agent.config.max_output_length,
    )
    if run.status is ReplStatus.DONE:
        return node.append(DoneOutput(content=output, result=run.answer))
    if run.status is ReplStatus.OK:
        return node.append(ExecOutput(content=output or "(no output)"))
    if run.status is ReplStatus.ERROR:
        return node.append(ErrorOutput(content=output))
    if run.status is ReplStatus.DEAD:
        content = f"{output}\n{COLD_REPL_NOTE}" if output else COLD_REPL_NOTE
        return node.append(ReplDead(content=content))
    raise ValueError(f"unknown repl status {run.status!r}")


DEFAULT_STEPS: dict[type[Node], type[StepFunction]] = {
    AgentStart: LLMRequestStep,
    UserQuery: LLMRequestStep,
    ErrorOutput: LLMRequestStep,
    ExecOutput: LLMRequestStep,
    LLMOutput: LLMOutputStep,
    ExecAction: ExecActionStep,
}


__all__ = [
    "CONTROL_QUERIES",
    "DEFAULT_STEPS",
    "ExecActionStep",
    "LLMOutputStep",
    "MAX_ITERS_EXCEEDED",
    "MessageBuilder",
    "StepFunction",
    "LLMRequestStep",
    "append_run_result",
    "at_final",
    "budget_nearly_spent",
    "needs_truncation",
]
