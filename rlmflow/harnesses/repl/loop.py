"""Our harness: one model call, then its cell, then the next model call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlmflow import schema
from rlmflow.graph.config import DEFAULT_MAX_ITERS
from rlmflow.graph.nodes import CODE, DONE, LLM_OUTPUT, Node, Step
from rlmflow.harnesses.base import Harness
from rlmflow.harnesses.repl.prompts import final_turn_spent, llm_turns, repl_messages
from rlmflow.utils import find_repl_blocks, has_repl_fence

if TYPE_CHECKING:
    from rlmflow.flow import Flow

MAX_ITERS_EXCEEDED = "[max_iters exceeded]"
MAX_BUDGET_EXCEEDED = "[max_budget exceeded]"


def protocol_error(reply: str) -> str:
    """What makes ``reply`` an invalid action."""
    blocks = find_repl_blocks(reply)
    if len(blocks) > 1:
        return f"found {len(blocks)} ```repl blocks; send exactly one per turn"
    if blocks:
        return "the ```repl block is empty"
    if has_repl_fence(reply):
        return "the ```repl block was never closed (the reply may have been cut off)"
    return "the reply is empty"


def malformed(reason: str) -> Node:
    """A ``code`` node that ran nothing, so the model sees the problem and retries."""
    text = (
        f"{reason}. Nothing ran. To act, reply with exactly one non-empty ```repl block. "
        "To finish, call submit(value) in that block, or reply in plain text with no "
        "```repl block."
    )
    return Node(CODE, code="", output=text, error=True)


class ReplHarness(Harness):
    """The REPL harness: a model call lands ``llm_output``; stepping that runs its cell."""

    name = "repl"
    handles = (*Harness.handles, LLM_OUTPUT)

    async def step(self, flow: Flow, step: Step) -> Node:
        if step.node.type == LLM_OUTPUT:
            return await self.run_reply(flow, step)
        return await self.call_model(flow, step)

    async def call_model(self, flow: Flow, step: Step) -> Node:
        config = step.agent.config
        if final_turn_spent(step):
            return Node(DONE, result=MAX_BUDGET_EXCEEDED)
        if llm_turns(step) >= (config.max_iters or DEFAULT_MAX_ITERS):
            return Node(DONE, result=MAX_ITERS_EXCEEDED)
        text, usage = await flow.complete(repl_messages(flow, step), config.model)
        return Node(LLM_OUTPUT, text=text, usage=usage.as_dict(), model=config.model)

    async def run_reply(self, flow: Flow, step: Step) -> Node:
        """Run the reply's one ``repl`` block, or take a reply without one as the answer."""
        text = step.node.data["text"]
        if text.strip() and not has_repl_fence(text):
            return self.text_answer(step, text.strip())
        blocks = find_repl_blocks(text)
        if len(blocks) != 1 or not blocks[0]:
            return malformed(protocol_error(text))
        cell, run = await flow.execute(step, blocks[0])
        if run.submitted:
            cell.type = DONE
            cell.data["result"] = run.answer
        return cell

    def text_answer(self, step: Step, text: str) -> Node:
        output_schema = step.agent.config.output_schema
        if output_schema is None:
            return Node(DONE, result=text)
        try:
            return Node(DONE, result=schema.check(text, output_schema))
        except (schema.SchemaError, TypeError) as exc:
            return Node(CODE, code="", output=f"Your reply is not a valid final answer.\n{exc}", error=True)


__all__ = ["MAX_BUDGET_EXCEEDED", "MAX_ITERS_EXCEEDED", "ReplHarness"]
