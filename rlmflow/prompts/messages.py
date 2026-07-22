"""Chat-message projection, the inputs manifest, and state-dependent nudges.

Keeps message shaping out of ``flow.py``. Durable guidance lives in the system
prompt (``prompts.py``); this module only holds what the system prompt cannot
carry: the projection of a graph trajectory into chat messages, the dynamic
inputs manifest (names + sizes), and the nudges tied to run state (continue,
forced-final, truncation). Uses the **inputs-as-`INPUTS`** model: each agent's
inputs live in a single ``INPUTS`` dict so a key never shadows a REPL variable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlmflow.graph import (
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


class PromptBuilder:
    """Shared base for the two prompt sides.

    A ``PromptBuilder`` is a ``(flow, graph) -> list[dict]`` callable: it renders
    one side of the conversation into chat messages. ``SystemPromptBuilder``
    returns a single ``system`` message; ``UserPromptBuilder`` returns the
    ``user``/``assistant`` turns. ``Flow.messages`` just concatenates the two
    (then truncates the user side and coalesces same-role turns), so both sides
    share this one shape.
    """

    def __call__(self, flow: Any, graph: Graph) -> list[dict[str, str]]:
        raise NotImplementedError


class UserPromptBuilder(PromptBuilder):
    """Prepare the graph for this turn, then project it into conversation turns.

    ``__call__`` runs the fixed machinery: it commits ``build()``'s per-turn
    content as a ``UserQuery`` node, materializes the continue/forced-final nudge
    (``ensure_nudge``), and only then projects the graph to chat messages
    (``project``). The graph is the single source of truth — everything the model
    sees this turn is a committed node before projection, which invents nothing.

    Override ``build(flow, graph) -> str | None`` to inject per-turn content (an
    observation, an injection); it returns ``None`` by default (nudge only). A
    bare ``(flow, graph) -> str | None`` can be passed as ``build_fn=`` (or as
    ``user=`` on a profile — see ``as_user_prompt``). Override one ``render_*``
    to reshape how a node type projects (return ``None`` to drop it), or
    ``render_node`` to change dispatch wholesale.
    """

    def __init__(
        self, build_fn: Callable[[Any, Graph], str | None] | None = None
    ) -> None:
        self._build_fn = build_fn

    def __call__(self, flow: Any, graph: Graph) -> list[dict[str, str]]:
        content = self.build(flow, graph)
        if content:
            flow.append_node(graph, UserQuery(content=content))
        self.ensure_nudge(flow, graph)
        return self.project(flow, graph)

    def build(self, flow: Any, graph: Graph) -> str | None:
        """Per-turn content to add to the conversation, or ``None`` for nothing.

        Uses ``build_fn`` when one was passed at construction; otherwise override
        this method. The framework commits the returned string as a ``UserQuery``
        node before projecting, so it lands in the durable trajectory.
        """
        if self._build_fn is not None:
            return self._build_fn(flow, graph)
        return None

    def ensure_nudge(self, flow: Any, graph: Graph) -> None:
        """Materialize the trailing nudge as a real ``UserQuery`` node.

        Formerly ``Flow._ensure_nudge``: on the last allowed turn commit the
        forced-final instruction; otherwise commit the continue nudge unless the
        trajectory already ends on a user turn (e.g. ``build()`` just added one).
        Making the nudge a node keeps the transcript self-contained.
        """
        llm_turns = sum(isinstance(node, LLMOutput) for node in graph.nodes)
        if llm_turns == flow.max_iters_for(graph) - 1:
            flow.append_node(graph, UserQuery(content=flow.final_action))
            return
        turns = self.project(flow, graph)
        if not turns or turns[-1]["role"] != "user":
            flow.append_node(graph, UserQuery(content=flow.continue_nudge))

    def project(self, flow: Any, graph: Graph) -> list[dict[str, str]]:
        """Pure projection of the graph's nodes into conversation turns."""
        turns: list[dict[str, str]] = []
        for node in graph.nodes:
            msg = self.render_node(node)
            if msg is not None:
                turns.append(msg)
        return turns

    # Dispatch: one renderer per node type. Nodes that are graph bookkeeping
    # rather than turns (``ExecAction``, ``DoneOutput``) fall through to ``None``.
    def render_node(self, node: Node) -> dict[str, str] | None:
        if isinstance(node, UserQuery):
            return self.render_user_query(node)
        if isinstance(node, LLMOutput):
            return self.render_llm_output(node)
        if isinstance(node, ExecOutput):
            return self.render_exec_output(node)
        if isinstance(node, ErrorOutput):
            return self.render_error_output(node)
        if isinstance(node, SupervisingOutput):
            return self.render_supervising_output(node)
        return None

    def render_user_query(self, node: UserQuery) -> dict[str, str] | None:
        return {"role": "user", "content": node.content}

    def render_llm_output(self, node: LLMOutput) -> dict[str, str] | None:
        return {"role": "assistant", "content": node.content}

    def render_exec_output(self, node: ExecOutput) -> dict[str, str] | None:
        return {"role": "user", "content": node.content or node.output}

    def render_error_output(self, node: ErrorOutput) -> dict[str, str] | None:
        return {"role": "user", "content": node.content or node.output}

    def render_supervising_output(
        self, node: SupervisingOutput
    ) -> dict[str, str] | None:
        return {"role": "user", "content": node.content or node.output}


#: Per-turn content: return a string to commit as a ``UserQuery``, or ``None``.
#: Bare functions are wrapped via ``as_user_prompt`` into a ``UserPromptBuilder``.
UserPromptBuildFn = Callable[[Any, Graph], str | None]
UserPromptSource = UserPromptBuildFn | UserPromptBuilder
UserPromptFn = UserPromptBuildFn  # alias


def as_user_prompt(source: Any) -> UserPromptBuilder:
    """Normalize a user-prompt source to a ``UserPromptBuilder``.

    A bare ``(flow, graph) -> str | None`` becomes ``UserPromptBuilder(build_fn=…)``
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


def merge_summary(child: Graph, delta: list[Node]) -> str:
    """Default one-line prompt summary for a merged branch's delta."""
    result = child.result() or "(no result)"
    n_exec = sum(1 for node in delta if isinstance(node, ExecAction))
    return f"[merged branch {child.graph_id}] {result} (folded {n_exec} exec step(s))"


__all__ = [
    "CONTINUE_NUDGE",
    "FINAL_ANSWER_ACTION",
    "TRUNCATION_SUMMARY",
    "PromptBuilder",
    "UserPromptBuilder",
    "UserPromptBuildFn",
    "UserPromptFn",
    "UserPromptSource",
    "as_user_prompt",
    "build_inputs_manifest",
    "coalesce_roles",
    "merge_summary",
]
