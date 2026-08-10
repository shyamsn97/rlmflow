"""Chat-message projection, the inputs manifest, and state-dependent nudges.

Keeps message shaping out of ``flow.py``. Durable guidance lives in the system
prompt (``prompts.py``); this module only holds what the system prompt cannot
carry: the projection of a trajectory into chat messages, the dynamic
inputs manifest (names + sizes), and the nudges tied to run state (continue,
forced-final, truncation). Uses the **inputs-as-`INPUTS`** model: each agent's
inputs live in a single ``INPUTS`` dict so a key never shadows a REPL variable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlmflow.graph.nodes import AgentStart, ErrorOutput, ExecOutput, LLMOutput, Node, UserQuery

FINAL_ANSWER_ACTION = (
    "You have used the full iteration budget without calling finish(). Based on the "
    "work above, call finish(answer) now with only the final answer, in the exact "
    "form the query requested. Do not investigate further."
)

CONTINUE_NUDGE = "Continue. Reply with one ```repl``` block, or call finish(...)."

TRUNCATION_SUMMARY = (
    "[earlier turns omitted to fit the context window; the most recent turns follow]"
)

COLD_REPL_NOTE = (
    "[this agent's REPL was restarted, so variables and imports from earlier turns "
    "are gone. Re-derive whatever you need before using it.]"
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


class PromptBuilder:
    """Shared base for the two prompt sides.

    A ``PromptBuilder`` is a ``(flow, root) -> list[dict]`` callable: it renders
    one side of the conversation into chat messages. ``SystemPromptBuilder``
    returns a single ``system`` message; ``UserPromptBuilder`` returns the
    ``user``/``assistant`` turns. ``Flow.messages`` just concatenates the two
    (then truncates the user side and coalesces same-role turns), so both sides
    share this one shape.
    """

    def __call__(self, flow: Any, agent: Node) -> list[dict[str, str]]:
        raise NotImplementedError


class UserPromptBuilder(PromptBuilder):
    """Project one agent transcript into conversation turns.

    Projection is pure. ``Flow.llm_turn`` commits ``build()`` output and the
    continue/forced-final nudge before asking this builder to render the tree.

    Override ``build(flow, agent) -> str | None`` to inject per-turn content (an
    observation, an injection); it returns ``None`` by default (nudge only). A
    bare ``(flow, agent) -> str | None`` can be passed as ``build_fn=`` (or as
    ``user=`` on a profile — see ``as_user_prompt``). Override one ``render_*``
    to reshape how a node type projects (return ``None`` to drop it), or
    ``render_node`` to change dispatch wholesale.
    """

    def __init__(self, build_fn: Callable[[Any, Node], str | None] | None = None) -> None:
        self._build_fn = build_fn

    def __call__(self, flow: Any, node: Node) -> list[dict[str, str]]:
        return self.project(flow, node)

    def build(self, flow: Any, agent: Node) -> str | None:
        """Per-turn content to add to the conversation, or ``None`` for nothing.

        Uses ``build_fn`` when one was passed at construction; otherwise override
        this method. The framework commits the returned string as a ``UserQuery``
        node before projecting, so it lands in the durable trajectory.
        """
        if self._build_fn is not None:
            return self._build_fn(flow, agent)
        return None

    def project(self, flow: Any, node: Node, keep: int | None = None) -> list[dict[str, str]]:
        """The conversation as of ``node``: its own agent's nodes, as turns.

        Walks backwards, so with ``keep`` the work is the size of the prompt rather
        than the size of the history: it stops as soon as that many turns are in
        hand instead of rendering a whole transcript to drop most of it.
        """
        turns: list[dict[str, str]] = []
        for current in node.walk(reverse=True):
            msg = self.render_node(current)
            if msg is None:
                continue
            turns.append(msg)
            if keep is not None and len(turns) >= keep:
                break
        return turns[::-1]

    # Dispatch: one renderer per node type. Nodes that are tree bookkeeping
    # rather than turns (``ExecAction``, ``DoneOutput``) fall through to ``None``.
    def render_node(self, node: Node) -> dict[str, str] | None:
        if isinstance(node, AgentStart):
            return {"role": "user", "content": node.content}
        if isinstance(node, UserQuery):
            return self.render_user_query(node)
        if isinstance(node, LLMOutput):
            return self.render_llm_output(node)
        if isinstance(node, ExecOutput):
            return self.render_exec_output(node)
        if isinstance(node, ErrorOutput):
            return self.render_error_output(node)
        return None

    def render_user_query(self, node: UserQuery) -> dict[str, str] | None:
        return {"role": "user", "content": node.content}

    def render_llm_output(self, node: LLMOutput) -> dict[str, str] | None:
        return {"role": "assistant", "content": node.content}

    def render_exec_output(self, node: ExecOutput) -> dict[str, str] | None:
        return {"role": "user", "content": node.content}

    def render_error_output(self, node: ErrorOutput) -> dict[str, str] | None:
        return {"role": "user", "content": node.content}


#: Per-turn content: return a string to commit as a ``UserQuery``, or ``None``.
#: Bare functions are wrapped via ``as_user_prompt`` into a ``UserPromptBuilder``.
UserPromptBuildFn = Callable[[Any, Node], str | None]
UserPromptSource = UserPromptBuildFn | UserPromptBuilder
UserPromptFn = UserPromptBuildFn  # alias


def as_user_prompt(source: Any) -> UserPromptBuilder:
    """Normalize a user-prompt source to a ``UserPromptBuilder``.

    A bare ``(flow, agent) -> str | None`` becomes ``UserPromptBuilder(build_fn=…)``
    so ``PromptProfile(user=fn)`` / ``Flow(user_prompt=fn)`` just work.
    """
    if isinstance(source, UserPromptBuilder):
        return source
    if callable(source):
        return UserPromptBuilder(build_fn=source)
    raise TypeError(f"unsupported user prompt source: {type(source)!r}")


def coalesce_roles(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge adjacent messages that share a role into one, joining their content
    with a blank line. Chat APIs expect strictly alternating user/assistant
    turns; injected instructions, REPL outputs, truncation markers and nudges can
    otherwise land two user turns in a row."""
    merged: list[dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            prev, cur = merged[-1]["content"], message["content"]
            merged[-1]["content"] = "\n\n".join(p for p in (prev, cur) if p)
        else:
            merged.append(dict(message))
    return merged


__all__ = [
    "COLD_REPL_NOTE",
    "CONTINUE_NUDGE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "PromptBuilder",
    "UserPromptBuildFn",
    "UserPromptBuilder",
    "UserPromptFn",
    "UserPromptSource",
    "as_user_prompt",
    "build_inputs_manifest",
    "coalesce_roles",
]
