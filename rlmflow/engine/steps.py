"""Default automatic transition producers and the guards they use.

``DEFAULT_TRANSITIONS`` order is the policy: application transitions first, then
these, and the first match wins.

```text
AgentStart|UserQuery|ErrorOutput|ExecOutput + out_of_room      → to_final           → FinalQuery
AgentStart|UserQuery|ErrorOutput|ExecOutput + needs_truncation → to_summary         → TruncationSummary
VerificationFailure                                            → to_repair          → RepairQuery
AgentStart|UserQuery|ErrorOutput|ExecOutput                    → complete           → LLMOutput | DoneOutput
LLMOutput + one non-empty repl block                           → to_action          → ExecAction
LLMOutput + invalid repl protocol                              → to_missing_action  → ExecAction
ExecAction                                                     → run_repl           → output node
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlmflow.engine.transitions import TransitionProtocolError, transition
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
    RepairQuery,
    ReplDead,
    TruncationSummary,
    TurnMode,
    UserQuery,
    VerificationFailure,
    turn_mode_for,
)
from rlmflow.llm import join_chunks
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.utils.helpers import repl_code_block, truncate_output

if TYPE_CHECKING:
    from rlmflow.flow import Flow

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


def needs_truncation(node: Node) -> bool:
    if isinstance(node, (FinalQuery, TruncationSummary)):
        return False
    keep = node.parent_agent.config.keep_n_messages
    if keep is None:
        return False
    keep = max(keep, 1)
    return len(node.project(keep=keep + 1)) > keep


@transition(
    source=(AgentStart, UserQuery, ErrorOutput, ExecOutput),
    when=out_of_room,
)
async def to_final(flow: Flow, node: Node) -> FinalQuery:
    return FinalQuery()


@transition(
    source=(AgentStart, UserQuery, ErrorOutput, ExecOutput),
    when=needs_truncation,
)
async def to_summary(flow: Flow, node: Node) -> TruncationSummary:
    return TruncationSummary()


@transition(source=VerificationFailure)
async def to_repair(flow: Flow, node: VerificationFailure) -> RepairQuery:
    return RepairQuery(
        content=(
            "Your previous finish candidate was rejected. Repair it and call "
            f"finish(...) again.\n\n{node.feedback}"
        )
    )


@transition(source=(AgentStart, UserQuery, ErrorOutput, ExecOutput))
async def complete(flow: Flow, node: Node) -> Node:
    agent = node.parent_agent
    if agent.config.max_iters is not None and agent.llm_turns() >= agent.config.max_iters:
        return DoneOutput(result=MAX_ITERS_EXCEEDED)
    messages = flow.build_messages(node)
    reply, usage = await join_chunks(flow.llm_for_step(node).stream(messages))
    return LLMOutput(
        content=reply,
        code=repl_code_block(reply),
        usage=usage,
        prompt_id=agent.record_prompt(messages[0]["content"]),
    )


def has_repl_action(node: LLMOutput) -> bool:
    """Whether the model explicitly chose the executable REPL path."""
    return bool(node.code)


@transition(source=LLMOutput, when=has_repl_action)
async def to_action(flow: Flow, node: LLMOutput) -> ExecAction:
    return ExecAction(code=node.code)


# Every invalid model response enters the runtime's established protocol-error
# recovery path. Only finish(value) inside a valid REPL action can complete a run.
@transition(source=LLMOutput)
async def to_missing_action(flow: Flow, node: LLMOutput) -> ExecAction:
    return ExecAction(code="")


@transition(source=ExecAction)
async def run_repl(flow: Flow, node: ExecAction) -> Node:
    return node_for_run(node, await flow.wrapped_runtime.execute(node))


def node_for_run(node: ExecAction, run: ReplRun) -> Node:
    output = truncate_output(
        run.output,
        node.parent_agent.config.max_output_length,
    )
    if run.status is ReplStatus.DONE:
        acceptance = run.acceptance or {}
        return DoneOutput(
            content=output,
            result=run.answer,
            schema_hash=acceptance.get("schema_hash"),
            verifier_hash=acceptance.get("verifier_hash"),
            verified=bool(acceptance.get("verified", False)),
        )
    if run.status is ReplStatus.REJECTED:
        failure = run.verification
        if failure is None:
            raise RuntimeError("rejected REPL run has no verification detail")
        return VerificationFailure(
            content=failure.feedback,
            candidate=failure.candidate,
            kind=failure.kind,
            feedback=failure.feedback,
            schema_hash=failure.schema_hash,
            verifier_hash=failure.verifier_hash,
            observations=output,
        )
    if run.status is ReplStatus.OK:
        if turn_mode_for(node) is TurnMode.FINAL:
            raise TransitionProtocolError("a final action must call finish(answer)")
        return ExecOutput(content=output or "(no output)")
    if run.status is ReplStatus.ERROR:
        return ErrorOutput(content=output)
    if run.status is ReplStatus.DEAD:
        content = f"{output}\n{COLD_REPL_NOTE}" if output else COLD_REPL_NOTE
        return ReplDead(content=content)
    raise ValueError(f"unknown repl status {run.status!r}")


DEFAULT_TRANSITIONS = (
    to_final,
    to_summary,
    to_repair,
    complete,
    to_action,
    to_missing_action,
    run_repl,
)


__all__ = ["DEFAULT_TRANSITIONS"]
