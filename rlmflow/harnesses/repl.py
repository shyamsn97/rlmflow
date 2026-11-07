"""Our harness: a model plus a persistent Python REPL, one node per call.

Each model call lands ``llm``; stepping that runs its one ``repl`` cell. The
system prompt is ``prompt_sections()`` with ``config.prompt`` on top; the messages
are the agent's history.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow import schema
from rlmflow.graph.config import DEFAULT_MAX_ITERS
from rlmflow.graph.nodes import (
    CODE,
    DONE,
    IDLE,
    IMAGES_KEY,
    LLM,
    MESSAGE,
    Agent,
    Node,
    Step,
)
from rlmflow.harnesses.base import Harness
from rlmflow.harnesses.prompts import argument_manifest, guidance
from rlmflow.utils import find_repl_blocks, has_repl_fence

if TYPE_CHECKING:
    from rlmflow.flow import Flow

PROTOCOL_TEXT = """\
You work through a persistent Python REPL, then give the final result.

To act, a response contains exactly one non-empty fenced ```repl``` block. Its code executes, the full response remains in the conversation, and the run continues from the observation. A response that opens more than one ```repl``` block, an empty one, or an unclosed one is a protocol error and does not execute. A response with no ```repl``` block is your final answer: its text completes the run. Plan the immediate next action from observations already obtained, then execute it. Compute answers from your inputs, not from assumptions. Avoid calling `submit(...)` in a block that prints output. When printed output must be inspected as evidence, end the block, read the observation, then submit in the next turn. Printing and submitting together is allowed only when the output is incidental and the answer is already determined. If you don't yet know the inputs' size, format, or contents, look first and decide after reading the observation. Prefer bounded, decision-relevant observations over broad dumps. When no next step remains, finish: call `submit(value)` inside a block to return a computed or structured value, or reply in plain text with no ```repl``` block. Do not reply in plain text while work remains.

The REPL persists across turns. The invocation `task` and any named JSON arguments are direct Python variables. Top-level `await` is available. Only printed stdout is returned, and outputs over ~20K characters are truncated, so inspect bounded samples or summaries."""

AGENTS_TEXT = """\
An `Agent` is what `launch_agent` returns and what `AGENTS[name]` looks up (by name, path or id). \
`await agent` waits for the result of an agent you launched. `agent.send(message)` queues a \
message for any agent, reopening it if it finished; for an agent you launched, \
`await agent.send(message)` also waits for its reply. Agents you didn't launch answer with a \
message at your next turn, so send and finish the block. `agent.status`, `agent.result()`, \
`agent.parent` and `agent.children` are as of when the block started.
`AGENTS` is this run's agents: `AGENTS.self` is you, `AGENTS.children` the agents you \
launched, and `AGENTS.print_graph(show_results=True)` prints the tree with statuses and results."""

FINAL_TURN_TEXT = """\
This is your last turn. Give the best supported final answer now: call `submit(value)` \
in exactly one fenced `repl` block, or reply in plain text with no `repl` block. Do not \
start work that requires another observation."""

TRUNCATION_TEXT = """\
[earlier turns omitted to fit the context window; the most recent turns follow. The \
REPL kept running, so variables, imports, and helpers defined in those turns are still \
bound — reuse them instead of redefining them.]"""


def with_images(text: str, paths: list[str]) -> list[dict[str, Any]]:
    """``text`` plus the images at ``paths``, as OpenAI-style content parts."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in paths:
        file = Path(path)
        if not file.is_file():
            parts.append({"type": "text", "text": f"[image not found: {path}]"})
            continue
        media_type = mimetypes.guess_type(file.name)[0] or "image/png"
        data = base64.b64encode(file.read_bytes()).decode()
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        )
    return parts


def append_text(content: str | list[dict[str, Any]], text: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return f"{content}\n\n{text}"
    return [*content, {"type": "text", "text": text}]


def node_messages(node: Node) -> list[dict[str, Any]]:
    """One history node as chat messages."""
    match node.type:
        case "message":
            return [{"role": "user", "content": node.data["text"]}]
        case "llm":
            return [{"role": "assistant", "content": node.data["text"]}]
        case "code" if "output" in node.data:
            output = node.data["output"] or "(no output)"
            images = node.data.get(IMAGES_KEY)
            return [{"role": "user", "content": with_images(output, images) if images else output}]
        case "idle" if node.data.get("text"):
            return [{"role": "assistant", "content": node.data["text"]}]
    return []


def history(step: Step) -> list[Node]:
    """The nodes the agent has seen, up to ``step.node``."""
    return step.graph.history(step.node) if isinstance(step.node, Node) else []


def origin(step: Step) -> Agent:
    """The agent whose task opens this history: a fork's furthest base agent."""
    agent = step.agent
    while agent.base is not None:
        agent = step.graph.owner(agent.base)
    return agent


def llm_turns(step: Step) -> int:
    return sum(node.type == LLM for node in history(step))


def history_messages(step: Step) -> list[dict[str, Any]]:
    """The task, then the agent's history as chat messages; the last ``keep_n_messages`` when set."""
    g, agent, config = step.graph, step.agent, step.agent.config
    messages = [{"role": "user", "content": origin(step).task}]
    messages += [message for node in history(step) for message in node_messages(node)]
    keep = config.keep_n_messages
    if keep is not None and len(messages) > keep:
        messages = [{"role": "user", "content": TRUNCATION_TEXT}, *messages[-keep:]] if keep else []
    notes = []
    children = g.children(agent)
    if children:
        notes.append(
            "\n".join(
                [
                    "Background agents:",
                    *(
                        f"- `{child.name}` ({g.path(child)}): "
                        + ("result ready" if g.status(child) == "done" else g.status(child))
                        for child in children
                    ),
                ]
            )
        )
    if llm_turns(step) == (config.max_iters or DEFAULT_MAX_ITERS) - 1 or final_turn(step):
        notes.append(FINAL_TURN_TEXT)
    if notes:
        note = "\n\n".join(notes)
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": append_text(messages[-1]["content"], note)}
        else:
            messages.append({"role": "user", "content": note})
    return messages


def final_turn(step: Step) -> bool:
    """Whether the latest message from rlmflow announced the final turn."""
    for node in reversed(history(step)):
        if node.type == LLM:
            return False
        if node.type == MESSAGE and node.data.get("final"):
            return True
    return False


def final_turn_spent(step: Step) -> bool:
    """The agent already had its final turn and didn't finish."""
    seen_reply = False
    for node in reversed(history(step)):
        if node.type == LLM:
            seen_reply = True
        elif node.type == MESSAGE and node.data.get("final"):
            return seen_reply
        elif node.type in (DONE, IDLE):
            return False
    return False


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
    """The REPL harness: a model call lands ``llm``; stepping that runs its cell."""

    name = "repl"
    handles = (*Harness.handles, LLM)

    def prompt_sections(self, flow: Flow, step: Step) -> dict[str, str | None]:
        from rlmflow.tools import format_tool_docs, repl_tools

        output_schema = step.agent.config.output_schema
        return {
            "protocol": PROTOCOL_TEXT,
            "tools": format_tool_docs(repl_tools(flow, step)),
            "agents": AGENTS_TEXT,
            **guidance(step),
            "arguments": argument_manifest(step),
            "schema": schema.contract(output_schema) if output_schema is not None else None,
        }

    def messages(self, flow: Flow, step: Step) -> list[dict[str, Any]]:
        """The system prompt, then the agent's history."""
        return [{"role": "system", "content": self.prompt(flow, step)}, *history_messages(step)]

    async def step(self, flow: Flow, step: Step) -> Node:
        if step.node.type == LLM:
            return await self.run_reply(flow, step)
        return await self.call_model(flow, step)

    async def call_model(self, flow: Flow, step: Step) -> Node:
        config = step.agent.config
        if final_turn_spent(step):
            return Node(DONE, result=MAX_BUDGET_EXCEEDED)
        if llm_turns(step) >= (config.max_iters or DEFAULT_MAX_ITERS):
            return Node(DONE, result=MAX_ITERS_EXCEEDED)
        text, usage = await flow.complete(self.messages(flow, step), config.model)
        return Node(LLM, text=text, usage=usage.as_dict(), model=config.model)

    async def run_reply(self, flow: Flow, step: Step) -> Node:
        """Run the reply's one ``repl`` block, or take a reply without one as the answer."""
        text = step.node.data["text"]
        if text.strip() and not has_repl_fence(text):
            return self.text_answer(step, text.strip())
        blocks = find_repl_blocks(text)
        if len(blocks) != 1 or not blocks[0]:
            return malformed(protocol_error(text))
        cell, run = await flow.execute(step, blocks[0])
        return Node(DONE, result=run.answer) if run.submitted else cell

    def text_answer(self, step: Step, text: str) -> Node:
        output_schema = step.agent.config.output_schema
        if output_schema is None:
            return Node(DONE, result=text)
        try:
            return Node(DONE, result=schema.check(text, output_schema))
        except (schema.SchemaError, TypeError) as exc:
            return Node(
                CODE, code="", output=f"Your reply is not a valid final answer.\n{exc}", error=True
            )


__all__ = [
    "MAX_BUDGET_EXCEEDED",
    "MAX_ITERS_EXCEEDED",
    "ReplHarness",
    "AGENTS_TEXT",
    "FINAL_TURN_TEXT",
    "PROTOCOL_TEXT",
    "TRUNCATION_TEXT",
    "final_turn",
    "final_turn_spent",
    "history",
    "history_messages",
    "llm_turns",
    "node_messages",
]
