"""The REPL harness's prompts: the system prompt and the messages built from the graph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import DEFAULT_MAX_ITERS, can_spawn
from rlmflow.graph.graph import DEPENDS_ON
from rlmflow.graph.nodes import AGENT_START, DONE, LLM_OUTPUT, MESSAGE, WAITING, Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow

CORE_PROTOCOL_TEXT = """\
You are a general-purpose Recursive Language Model (RLM). Work through the persistent Python REPL, then give the final result.

To act, a response contains exactly one non-empty fenced ```repl``` block. Its code executes, the full response remains in the conversation, and the run continues from the observation. A response that opens more than one ```repl``` block, an empty one, or an unclosed one is a protocol error and does not execute. A response with no ```repl``` block is your final answer: its text completes the run. Plan the immediate next action from observations already obtained, then execute it. Compute answers from your inputs, not from assumptions. Avoid calling `submit(...)` in a block that prints output. When printed output must be inspected as evidence, end the block, read the observation, then submit in the next turn. Printing and submitting together is allowed only when the output is incidental and the answer is already determined. If you don't yet know the inputs' size, format, or contents, look first and decide after reading the observation. Prefer bounded, decision-relevant observations over broad dumps. When no next step remains, finish: call `submit(value)` inside a block to return a computed or structured value, or reply in plain text with no ```repl``` block. Do not reply in plain text while work remains.

The REPL persists across turns. The invocation `task` and any named JSON arguments are direct Python variables. Top-level `await` is available. Only printed stdout is returned, and outputs over ~20K characters are truncated, so inspect bounded samples or summaries."""

EVIDENCE_POLICY = """\
Evidence and completion policy:
- Use Python to narrow, count, and aggregate exact candidates. Use semantic model
  queries only for semantic work, then combine and check compact results in Python.
- Before submitting, check the candidate against the inputs, in code where possible.
- Do not convert missing data, access failures, parse failures, or tool errors into a
  plausible answer. Surface the failure unless you validate a recovery."""

EXECUTION_POLICY = """\
Work allocation and graph policy:
- Subagents start with fresh context, can use cheaper models, and can run independent
  work concurrently. Use those advantages to reduce token cost and wall-clock time.
- When work has multiple substantial outputs or separable components with clear
  interfaces, assign them to different agents before implementing them yourself.
- Launch independent owners together, using the cheapest capable config. Keep work
  local only when it is small or tightly coupled enough that delegation costs more.
- Give every agent one bounded, non-overlapping deliverable and the arguments it needs.
- When a known work product depends on another, launch it with the prerequisite
  handle as a named argument; do not wait before declaring a known edge.
- Do not duplicate delegated work locally or create an agent for light synthesis.
- Consume and validate results before finishing.
- Stop launching when the answer is supported."""

WORKER_ADDENDUM = """\
Assigned-work policy:
- Own the requested deliverable and use the provided arguments.
- Do not leave TODOs, stubs, or work for a parent to complete.
- Validate the result."""

AGENTS_TEXT = """\
`AGENTS` is a read-only snapshot of this run's agents, refreshed before each block:
`AGENTS.get()` is you, `AGENTS.get_children()` your agents, and
`AGENTS.print_graph(show_results=True)` prints the tree with statuses and results.
`send_agent(target, message)` queues a message for another agent, and
`ask_agent(target, message)` sends one and waits for its reply."""

FINAL_TURN_TEXT = """\
This is your last turn. Give the best supported final answer now: call `submit(value)` \
in exactly one fenced `repl` block, or reply in plain text with no `repl` block. Do not \
start work that requires another observation."""

TRUNCATION_TEXT = """\
[earlier turns omitted to fit the context window; the most recent turns follow. The \
REPL kept running, so variables, imports, and helpers defined in those turns are still \
bound — reuse them instead of redefining them.]"""


def argument_manifest(step: Step) -> str:
    """Arguments by name and size; the values stay in the REPL."""
    g, agent = step.graph, step.agent
    values = {**agent.resolved_args(), **g.inputs(agent)}
    if not values:
        return ""
    descriptions = {
        **agent.argument_descriptions(),
        **{
            edge.data["as"]: edge.data["description"]
            for edge in g.edges(kind=DEPENDS_ON, source=agent.id)
            if edge.data.get("as") and edge.data.get("description")
        },
    }
    lines = []
    for name, value in values.items():
        if isinstance(value, str):
            size = f"str, {len(value)} chars"
        elif isinstance(value, list):
            size = f"list, {len(value)} items"
        elif isinstance(value, dict):
            size = f"object, {len(value)} fields"
        else:
            size = type(value).__name__
        line = f"- `{name}`: {size}"
        if description := descriptions.get(name):
            line += f" — {' '.join(description.split())}"
        lines.append(line)
    return "Your REPL has these task arguments as direct variables:\n" + "\n".join(lines)


def system_prompt(flow: Flow, step: Step) -> str:
    """``config.system_prompt`` (``{tools}`` is filled in), or the default prompt."""
    from rlmflow import schema
    from rlmflow.harnesses.repl.tools import format_tool_docs, repl_tools

    config = step.agent.config
    tools = format_tool_docs(repl_tools(flow, step))
    if config.system_prompt is not None:
        text = config.system_prompt
        return text.replace("{tools}", tools) if "{tools}" in text else text
    spawns = can_spawn(step.graph, step.agent)
    parts = [
        CORE_PROTOCOL_TEXT,
        EVIDENCE_POLICY,
        EXECUTION_POLICY if spawns else WORKER_ADDENDUM,
        tools,
        AGENTS_TEXT,
        argument_manifest(step),
    ]
    if config.output_schema is not None:
        parts.append(schema.contract(config.output_schema))
    if config.max_depth > 0:
        depth = step.graph.depth(step.agent)
        text = (
            f"You are using model key **`{config.model}`** at recursion depth "
            f"**{depth}** of max **{config.max_depth}**."
        )
        parts.append(text + ("" if spawns else " You cannot start agents."))
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def node_messages(node: Node) -> list[dict[str, str]]:
    """One history node as chat messages."""
    match node.type:
        case "agent_start":
            return [{"role": "user", "content": node.task}]
        case "message":
            return [{"role": "user", "content": node.data["text"]}]
        case "llm_output":
            return [{"role": "assistant", "content": node.data["text"]}]
        case "code" | "done" if "output" in node.data:
            return [{"role": "user", "content": node.data["output"] or "(no output)"}]
        case "waiting" if node.data.get("text"):
            return [{"role": "assistant", "content": node.data["text"]}]
    return []


def llm_turns(step: Step) -> int:
    return sum(node.type == LLM_OUTPUT for node in step.graph.history(step.node))


def repl_messages(flow: Flow, step: Step) -> list[dict[str, str]]:
    """System prompt, then the agent's history; the last ``keep_n_messages`` when set."""
    g, agent, config = step.graph, step.agent, step.agent.config
    messages = [
        message
        for node in g.history(step.node)
        # A fork's own agent_start sits mid-history and repeats the task.
        if not (node.type == AGENT_START and g.prev(node) is not None)
        for message in node_messages(node)
    ]
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
                        + ("result ready" if g.terminal(child) else "running")
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
            messages[-1] = {"role": "user", "content": f"{messages[-1]['content']}\n\n{note}"}
        else:
            messages.append({"role": "user", "content": note})
    return [{"role": "system", "content": system_prompt(flow, step)}, *messages]


def final_turn(step: Step) -> bool:
    """Whether the latest message from rlmflow announced the final turn."""
    for node in reversed(step.graph.history(step.node)):
        if node.type == LLM_OUTPUT:
            return False
        if node.type == MESSAGE and node.data.get("final"):
            return True
    return False


def final_turn_spent(step: Step) -> bool:
    """The agent already had its final turn and didn't finish."""
    seen_reply = False
    for node in reversed(step.graph.history(step.node)):
        if node.type == LLM_OUTPUT:
            seen_reply = True
        elif node.type == MESSAGE and node.data.get("final"):
            return seen_reply
        elif node.type in (DONE, WAITING):
            return False
    return False


def preview(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "AGENTS_TEXT",
    "CORE_PROTOCOL_TEXT",
    "EVIDENCE_POLICY",
    "EXECUTION_POLICY",
    "FINAL_TURN_TEXT",
    "TRUNCATION_TEXT",
    "WORKER_ADDENDUM",
    "argument_manifest",
    "final_turn",
    "final_turn_spent",
    "llm_turns",
    "node_messages",
    "preview",
    "repl_messages",
    "system_prompt",
]
