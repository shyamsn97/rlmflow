"""Our harness: a model plus a persistent Python REPL, one node per model call.

Each model call is an assistant node: the reply is its text, and its one
``repl`` block is its code, which runs in place. The system prompt is
``prompt_sections()`` with ``config.prompt`` on top; the messages are the
agent's transcript.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow import schema
from rlmflow.graph.config import DEFAULT_MAX_ITERS
from rlmflow.graph.nodes import IMAGES_KEY, Agent, Node, Step
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


def node_messages(node: Node, *, first: bool = False) -> list[dict[str, Any]]:
    """One node as chat messages: its text, then what running it printed."""
    messages: list[dict[str, Any]] = []
    text = node.text
    if isinstance(node, Agent) and not first:
        text = ""
    if node.role == "user":
        if text:
            messages.append({"role": "user", "content": text})
    elif text or "model" in node.data:
        messages.append({"role": "assistant", "content": text})
    elif node.runs_code:
        messages.append({"role": "assistant", "content": f"```repl\n{node.code}\n```"})
    if node.runs_code or node.output:
        output = node.output or "(no output)"
        images = node.data.get(IMAGES_KEY)
        messages.append(
            {"role": "user", "content": with_images(output, images) if images else output}
        )
    return messages


def turns(step: Step) -> int:
    """How many turns the agent has taken."""
    return sum("seen" in node.data for node in step.node.transcript())


def transcript_messages(step: Step) -> list[dict[str, Any]]:
    """The agent's transcript as chat messages; the last ``keep_n_messages`` when set."""
    agent, config = step.agent, step.config
    nodes = step.node.transcript()
    messages = [
        message
        for index, node in enumerate(nodes)
        for message in node_messages(node, first=index == 0)
    ]
    keep = config.keep_n_messages
    if keep is not None and len(messages) > keep:
        messages = [{"role": "user", "content": TRUNCATION_TEXT}, *messages[-keep:]] if keep else []
    notes = []
    children = agent.children()
    if children:
        notes.append(
            "\n".join(
                [
                    "Background agents:",
                    *(
                        f"- `{child.name}` ({child.path()}): "
                        + ("result ready" if child.status() == "done" else child.status())
                        for child in children
                    ),
                ]
            )
        )
    if turns(step) == (config.max_iters or DEFAULT_MAX_ITERS) - 1 or final_turn(step):
        notes.append(FINAL_TURN_TEXT)
    if notes:
        note = "\n\n".join(notes)
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": append_text(messages[-1]["content"], note)}
        else:
            messages.append({"role": "user", "content": note})
    return messages


def final_turn(step: Step) -> bool:
    """Whether rlmflow announced the final turn since the agent's last one."""
    for node in reversed(step.node.transcript()):
        if "seen" in node.data:
            return False
        if node.data.get("final"):
            return True
    return False


def final_turn_spent(step: Step) -> bool:
    """The agent already had its final turn and didn't finish."""
    took_turn = False
    for node in reversed(step.node.transcript()):
        if node.data.get("final"):
            return took_turn
        if node.data.get("terminal") or node.data.get("idle"):
            return False
        if "seen" in node.data:
            took_turn = True
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


def malformed(reason: str) -> str:
    """What the model is told when its reply ran nothing."""
    return (
        f"{reason}. Nothing ran. To act, reply with exactly one non-empty ```repl block. "
        "To finish, call submit(value) in that block, or reply in plain text with no "
        "```repl block."
    )


class ReplHarness(Harness):
    """The REPL harness: each model call is one assistant node, whose ``repl`` block it runs."""

    name = "repl"

    def prompt_sections(self, flow: Flow, step: Step) -> dict[str, str | None]:
        from rlmflow.tools import format_tool_docs, repl_tools

        output_schema = step.config.output_schema
        return {
            "protocol": PROTOCOL_TEXT,
            "tools": format_tool_docs(repl_tools(flow, step)),
            "agents": AGENTS_TEXT,
            **guidance(step),
            "arguments": argument_manifest(step),
            "schema": schema.contract(output_schema) if output_schema is not None else None,
        }

    def messages(self, flow: Flow, step: Step) -> list[dict[str, Any]]:
        """The system prompt, then the agent's transcript."""
        return [{"role": "system", "content": self.prompt(flow, step)}, *transcript_messages(step)]

    async def step(self, flow: Flow, step: Step) -> Node:
        g, agent, config = step.graph, step.agent, step.config
        seen = len(step.node.transcript())
        if final_turn_spent(step):
            return flow.append(g, agent, final_node(MAX_BUDGET_EXCEEDED, seen))
        if turns(step) >= (config.max_iters or DEFAULT_MAX_ITERS):
            return flow.append(g, agent, final_node(MAX_ITERS_EXCEEDED, seen))
        started = time.time()
        text, usage = await flow.complete(self.messages(flow, step), config.model)
        node = Node(
            "assistant", text, data={"usage": usage.as_dict(), "model": config.model, "seen": seen}
        )
        node.started_at = started
        if text.strip() and not has_repl_fence(text):
            self.text_answer(step, node, text.strip())
            node.finished_at = time.time()
            return flow.append(g, agent, node)
        blocks = find_repl_blocks(text)
        if len(blocks) != 1 or not blocks[0]:
            node.output, node.error = malformed(protocol_error(text)), True
            node.finished_at = time.time()
            return flow.append(g, agent, node)
        node.code = blocks[0]
        flow.append(g, agent, node)
        await flow.run_node(g, agent, node)
        return node

    def text_answer(self, step: Step, node: Node, text: str) -> None:
        """A reply with no ``repl`` block is the answer, if it fits the output schema."""
        output_schema = step.config.output_schema
        try:
            node.result = text if output_schema is None else schema.check(text, output_schema)
        except (schema.SchemaError, TypeError) as exc:
            node.output, node.error = f"Your reply is not a valid final answer.\n{exc}", True
            return
        node.data["terminal"] = True


def final_node(result: str, seen: int) -> Node:
    """A turn that ends the agent without calling the model."""
    return Node("assistant", result=result, data={"seen": seen, "terminal": True})


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
    "final_node",
    "transcript_messages",
    "node_messages",
    "turns",
]
