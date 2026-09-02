"""Replaceable class-based graph step functions."""

from __future__ import annotations

from rlmflow.engine.transitions import (
    DEFAULT_TRANSITIONS,
    InvalidTransitionError,
    MissingTransitionError,
    TransitionProtocolError,
    Transitions,
    at_final,
    budget_nearly_spent,
    needs_truncation,
)
from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    Node,
    ReplDead,
    TurnMode,
    UserQuery,
    requested_transition,
)
from rlmflow.llm import LLMClient, join_chunks
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.runtime.runtime import WrappedRuntime
from rlmflow.utils.helpers import code_block, truncate_output

MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


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
        transitions: Transitions = DEFAULT_TRANSITIONS,
    ) -> None:
        self.llm = llm
        self.messages = messages
        self.runtime = runtime
        self.transitions = transitions

    async def __call__(self, node: Node) -> Node:
        raise NotImplementedError


class LLMRequestStep(StepFunction):
    """Order all control nodes before the next model call."""

    async def __call__(self, node: Node) -> Node:
        automatic = self.transitions.resolve_automatic(node)
        if automatic is not None:
            return node.append(automatic.target())

        if isinstance(node, ExecOutput):
            selected = requested_transition(node)
            if selected is None:
                action = node.prev
                if isinstance(action, ExecAction) and _turn_mode(action) is not TurnMode.ACTION:
                    return await self.chat(node)
                raise MissingTransitionError(
                    "a successful action must end with transition(...) or finish(...)"
                )
            option = self.transitions.resolve(node, selected)
            if option is None:
                raise InvalidTransitionError(
                    selected,
                    (choice.name for choice in self.transitions.available(node)),
                )
            if option.target is not None:
                return node.append(option.target())

        return await self.chat(node)

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
    if run.status is ReplStatus.TRANSITION:
        if _turn_mode(node) is not TurnMode.ACTION or not run.transition:
            raise TransitionProtocolError("transition(...) is only valid during an action turn")
        node.requested_transition = run.transition
        return node.append(ExecOutput(content=output or "(no output)"))
    if run.status is ReplStatus.OK:
        mode = _turn_mode(node)
        if mode is TurnMode.ACTION:
            raise MissingTransitionError(
                "a successful action must end with transition(...) or finish(...)"
            )
        if mode is TurnMode.FINAL:
            raise MissingTransitionError("a final action must call finish(answer)")
        return node.append(ExecOutput(content=output or "(no output)"))
    if run.status is ReplStatus.ERROR:
        return node.append(ErrorOutput(content=output))
    if run.status is ReplStatus.DEAD:
        content = f"{output}\n{COLD_REPL_NOTE}" if output else COLD_REPL_NOTE
        return node.append(ReplDead(content=content))
    raise ValueError(f"unknown repl status {run.status!r}")


def _turn_mode(node: ExecAction) -> TurnMode:
    output = node.prev
    source = output.prev if isinstance(output, LLMOutput) else None
    return source.turn_mode if source is not None else TurnMode.NONE


DEFAULT_STEPS: dict[type[Node], type[StepFunction]] = {
    AgentStart: LLMRequestStep,
    UserQuery: LLMRequestStep,
    ErrorOutput: LLMRequestStep,
    ExecOutput: LLMRequestStep,
    LLMOutput: LLMOutputStep,
    ExecAction: ExecActionStep,
}


__all__ = [
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
