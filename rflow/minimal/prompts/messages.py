"""Chat-message projection, the inputs manifest, and state-dependent nudges.

Keeps message shaping out of ``flow.py``. Durable guidance lives in the system
prompt (``prompts.py``); this module only holds what the system prompt cannot
carry: the projection of a graph trajectory into chat messages, the dynamic
inputs manifest (names + sizes), and the nudges tied to run state (continue,
forced-final, truncation). Uses the **inputs-as-`INPUTS`** model: each agent's
inputs live in a single ``INPUTS`` dict so a key never shadows a REPL variable.
"""

from __future__ import annotations

from rflow.minimal.graph import (
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    Node,
    SupervisingOutput,
    UserQuery,
)

FINAL_ANSWER_ACTION = (
    "You have used the full iteration budget without calling done(). Based on the "
    "work above, call done(answer) now with only the final answer, in the exact "
    "form the query requested. Do not investigate further."
)

CONTINUE_NUDGE = "Continue. Reply with one ```repl``` block, or call done(...)."

TRUNCATION_SUMMARY = (
    "[earlier turns omitted to fit the context window; the most recent turns follow]"
)


def build_inputs_manifest(inputs: dict[str, str]) -> str:
    """List inputs by name + size (no value dumps), with a big-input chunk hint.

    Lets the model gauge how large each REPL-visible input is before choosing a
    chunking / fanout strategy. Returns ``""`` when there are no inputs.
    """
    if not inputs:
        return ""
    lines = [f"- {name}: str, {len(value)} chars" for name, value in inputs.items()]
    total = sum(len(value) for value in inputs.values())
    manifest = (
        "Your REPL `INPUTS` contain:\n"
        + "\n".join(lines)
        + f"\nTotal input chars: {total}. Read small instruction-like inputs fully; "
        "for large inputs keep the value in a variable and print targeted windows."
    )
    if total > 50_000:
        manifest += (
            f" (~{total // 4} tokens — process focused pieces in later turns or "
            "subagents rather than dumping the payload.)"
        )
    return manifest


def build_messages(
    graph: Graph,
    system_prompt: str,
    *,
    max_messages: int | None = None,
    force_final: bool = False,
) -> list[dict[str, str]]:
    """Project a graph's node stream into chat messages for one LLM call.

    User queries are delivered verbatim (the model's task); assistant turns and
    REPL outputs follow. When the trajectory grows past ``max_messages`` the
    middle is elided with a marker. A trailing nudge is appended so the model
    always ends on an actionable user turn (forced final on budget exhaustion,
    otherwise a continue nudge if the last message was not already a user turn).
    """
    msgs = [{"role": "system", "content": system_prompt}]
    for node in graph.nodes:
        if isinstance(node, UserQuery):
            msgs.append({"role": "user", "content": node.content})
        elif isinstance(node, LLMOutput):
            msgs.append({"role": "assistant", "content": node.content})
        elif isinstance(node, (ExecOutput, ErrorOutput, SupervisingOutput)):
            msgs.append({"role": "user", "content": node.content or node.output})

    if max_messages is not None and len(msgs) > max_messages:
        head, body = (msgs[:1], msgs[1:]) if msgs[0]["role"] == "system" else ([], msgs)
        keep = max(1, max_messages - len(head) - 1)
        summary = {"role": "user", "content": TRUNCATION_SUMMARY}
        msgs = [*head, summary, *body[-keep:]]

    if force_final:
        msgs.append({"role": "user", "content": FINAL_ANSWER_ACTION})
    elif msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": CONTINUE_NUDGE})
    return msgs


def merge_summary(child: Graph, delta: list[Node]) -> str:
    """Default one-line prompt summary for a merged branch's delta."""
    result = child.result() or "(no result)"
    n_exec = sum(1 for node in delta if isinstance(node, ExecAction))
    return f"[merged branch {child.graph_id}] {result} (folded {n_exec} exec step(s))"


__all__ = [
    "CONTINUE_NUDGE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "build_inputs_manifest",
    "build_messages",
    "merge_summary",
]
