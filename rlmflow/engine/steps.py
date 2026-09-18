"""Default ``@Flow.transitions.on`` producers and the guards they use.

Registration order is the policy. First matching row wins. Own rows, then ``_base``.

```text
AgentStart|UserQuery|ErrorOutput|ExecOutput + out_of_room      → to_final          → FinalQuery
AgentStart|UserQuery|ErrorOutput|ExecOutput + needs_truncation → to_summary        → TruncationSummary
AgentStart|ErrorOutput|ExecOutput                              → to_plan           → PlanQuery
UserQuery + child_returned (after DoneOutput or ExecOutput)    → to_plan           → PlanQuery
UserQuery                                                      → complete         → LLMOutput | DoneOutput
LLMOutput                                                      → to_action         → ExecAction
ExecAction                                                     → run_repl         → output or chosen UserQuery

transition("...")            REPL tool → TransitionSignal → run_repl resolves the choice
                             → selected UserQuery. Prompt footer is prompts/messages.py.
```
"""

from __future__ import annotations

from rlmflow.engine.transitions import InvalidTransitionError, TransitionProtocolError
from rlmflow.flow import Flow
from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    AgentStart,
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
    TurnMode,
    UserQuery,
)
from rlmflow.llm import join_chunks
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.utils.helpers import code_block, truncate_output

MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


def at_final(agent: AgentStart) -> bool:
    limit = agent.config.max_iters
    return limit is not None and agent.llm_turns() == limit - 1


def budget_nearly_spent(node: Node) -> bool:
    """Whether one more average turn would exhaust the token budget."""
    limit = node.parent_agent.config.max_budget
    root = node.root
    if limit is None or root is None:
        return False
    turns = root.stats.node_counts.get(LLMOutput.type, 0)
    if turns == 0:
        return False
    spent = root.usage.total
    return spent + spent / turns >= limit


def out_of_room(node: Node) -> bool:
    return not isinstance(node, FinalQuery) and (
        at_final(node.parent_agent) or budget_nearly_spent(node)
    )


def child_returned(node: Node) -> bool:
    """A host or child-handoff ``UserQuery`` after an observation, not a chat turn."""
    return type(node) is UserQuery and isinstance(node.prev, (DoneOutput, ExecOutput))


def needs_truncation(node: Node) -> bool:
    if isinstance(node, (FinalQuery, TruncationSummary)):
        return False
    keep = node.parent_agent.config.keep_n_messages
    if keep is None:
        return False
    keep = max(keep, 1)
    return len(node.project(keep=keep + 1)) > keep


@Flow.transitions.on(AgentStart, UserQuery, ErrorOutput, ExecOutput, when=out_of_room)
async def to_final(flow: Flow, node: Node) -> FinalQuery:
    return FinalQuery()


@Flow.transitions.on(AgentStart, UserQuery, ErrorOutput, ExecOutput, when=needs_truncation)
async def to_summary(flow: Flow, node: Node) -> TruncationSummary:
    return TruncationSummary()


@Flow.transitions.on(AgentStart, ErrorOutput, ExecOutput)
@Flow.transitions.on(UserQuery, when=child_returned)
async def to_plan(flow: Flow, node: Node) -> PlanQuery:
    return PlanQuery()


@Flow.transitions.on(UserQuery)
async def complete(flow: Flow, node: UserQuery) -> Node:
    agent = node.parent_agent
    if agent.config.max_iters is not None and agent.llm_turns() >= agent.config.max_iters:
        return DoneOutput(result=MAX_ITERS_EXCEEDED)
    messages = flow.build_messages(node)
    reply, usage = await join_chunks(flow.llm_for_step(node).stream(messages))
    return LLMOutput(
        content=reply,
        code=code_block(reply),
        usage=usage,
        prompt_id=agent.record_prompt(messages[0]["content"]),
    )


@Flow.transitions.on(LLMOutput)
async def to_action(flow: Flow, node: LLMOutput) -> ExecAction:
    return ExecAction(code=node.code)


@Flow.transitions.on(ExecAction)
async def run_repl(flow: Flow, node: ExecAction) -> Node:
    run = await flow.wrapped_runtime.execute(node)
    if run.status is ReplStatus.TRANSITION:
        return transition_target(flow, node, run)
    return node_for_run(node, run)


def transition_target(flow: Flow, node: ExecAction, run: ReplRun) -> UserQuery:
    """Resolve one REPL ``transition(name)`` directly to its selected query."""
    selected = run.transition
    if _turn_mode(node) is not TurnMode.ACTION or not selected:
        raise TransitionProtocolError("transition(...) is only valid during a working turn")
    option = flow.transitions.resolve_choice(node, selected)
    if option is None:
        raise InvalidTransitionError(
            selected,
            (choice.name for choice in flow.transitions.available(node)),
        )
    return option.target()


def node_for_run(node: ExecAction, run: ReplRun) -> Node:
    output = truncate_output(
        run.output,
        node.parent_agent.config.max_output_length,
    )
    if run.status is ReplStatus.DONE:
        return DoneOutput(content=output, result=run.answer)
    if run.status is ReplStatus.OK:
        if _turn_mode(node) is TurnMode.FINAL:
            raise TransitionProtocolError("a final action must call finish(answer)")
        return ExecOutput(content=output or "(no output)")
    if run.status is ReplStatus.ERROR:
        return ErrorOutput(content=output)
    if run.status is ReplStatus.DEAD:
        content = f"{output}\n{COLD_REPL_NOTE}" if output else COLD_REPL_NOTE
        return ReplDead(content=content)
    raise ValueError(f"unknown repl status {run.status!r}")


def _turn_mode(node: ExecAction) -> TurnMode:
    output = node.prev
    source = output.prev if isinstance(output, LLMOutput) else None
    return source.turn_mode if source is not None else TurnMode.NONE


__all__ = [
    "MAX_ITERS_EXCEEDED",
    "at_final",
    "budget_nearly_spent",
    "child_returned",
    "complete",
    "needs_truncation",
    "node_for_run",
    "out_of_room",
    "run_repl",
    "to_action",
    "to_final",
    "to_plan",
    "to_summary",
    "transition_target",
]
