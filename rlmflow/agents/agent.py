"""Agents: the harnesses that drive a model for one run.

``Agent`` is rlmflow's own: a model plus a persistent Python REPL, one node per
model call. It is also the base of every harness. Everything an agent is told is
``system_prompt``, everything it sees is ``messages``, and everything it does
is ``step``, which returns the node flow appends.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow import schema
from rlmflow.agents.prompts import (
    AGENTS_HELP,
    FINAL_TURN,
    LOCATION,
    PROTOCOL,
    REPL_TOOL,
    STATE,
    TRUNCATED,
    USE_RLMFLOW_REPL,
    VERIFY,
    argument_manifest,
    skill,
)
from rlmflow.graph.config import DEFAULT_MAX_ITERS
from rlmflow.graph.nodes import IMAGES_KEY, Node
from rlmflow.llm.base import Reply, ToolCall

if TYPE_CHECKING:
    from rlmflow.repl import Repl

#: ``await llm(model, messages, tools=...) -> Reply``: the model call flow hands a step.
LLM = Callable[..., Awaitable[Reply]]

#: A prompt section: its text, ``None`` to drop it, or ``(node, repl) -> text``.
Section = str | None | Callable[[Node, "Repl"], "str | None"]

MAX_ITERS_EXCEEDED = "[max_iters exceeded]"


class Agent:
    """rlmflow's REPL harness, and the base of every harness: what every run of one kind shares.

    One agent serves many runs, so it holds no run's state: ``step`` gets the
    run's latest node, the model call and the run's ``Repl``.

    - ``system``: the whole system prompt, replacing the sections.
    - ``prompt``: ``{name: text | None | (node, repl) -> text}`` merged over
      ``sections()``: adds, replaces or (``None``) drops a section by name.
    - ``keep``: keep only the last ``keep`` messages of history.
    - ``max_iters``: steps a run may take; ``None`` for no limit.
    - ``description``: shown to agents choosing what to launch.
    - ``max_depth``: levels of runs it may launch below itself.
    - ``output_schema``: a JSON Schema (or a type) the result must match.
    - ``reasoning_effort``: OpenAI reasoning effort. ``"auto"`` uses ``"low"``
      for GPT-6 (the lowest value every GPT-6 tier accepts) and the provider
      default for other models.
    """

    #: How this harness reaches the REPL: ``PROTOCOL``, or ``USE_RLMFLOW_REPL`` for a CLI agent.
    protocol = PROTOCOL
    #: The agent ``launch_agent`` starts when none is named; ``None`` for the launcher's own.
    child_agent: str | None = None

    def __init__(
        self,
        model: str | None = "gpt-6-sol",
        *,
        system: str | None = None,
        prompt: dict[str, Section] | None = None,
        keep: int | None = None,
        max_iters: int | None = DEFAULT_MAX_ITERS,
        description: str = "",
        max_depth: int = 1,
        output_schema: Any = None,
        reasoning_effort: str | None = "auto",
    ) -> None:
        self.model = model
        self.system = system
        self.prompt = dict(prompt or {})
        self.keep = keep
        self.max_iters = max_iters
        self.description = description
        self.max_depth = max_depth
        self.output_schema = (
            schema.json_schema_for(output_schema) if output_schema is not None else None
        )
        if reasoning_effort == "auto":
            reasoning_effort = "low" if model is not None and model.startswith("gpt-6") else None
        self.reasoning_effort = reasoning_effort

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def summary(self) -> str:
        """How other agents see it when choosing what to launch."""
        text = f"{type(self).__name__}, model `{self.model}`"
        return f"{text}; {self.description}" if self.description else text

    @property
    def cli(self) -> bool:
        """Whether it runs code through ``rlmflow repl`` rather than in its replies."""
        return self.protocol is USE_RLMFLOW_REPL

    # -- what it is told --------------------------------------------------------

    def sections(self, node: Node, repl: Repl) -> dict[str, Section]:
        """The default system prompt, as named sections in order."""
        output_schema = self.output_schema
        return {
            "protocol": self.protocol,
            "location": LOCATION.format(location=repl.location)
            if self.cli and repl.location
            else None,
            "tools": repl.tool_docs(own_only=self.cli),
            "agents": (launchable(repl) if repl.can_launch() else None)
            if self.cli
            else AGENTS_HELP,
            "verify": VERIFY,
            "state": STATE,
            "orchestrate": skill("orchestrate") if repl.can_launch() else None,
            "arguments": argument_manifest(node),
            "schema": schema.contract(output_schema) if output_schema is not None else None,
        }

    def system_prompt(self, node: Node, repl: Repl) -> str:
        """``system`` if set; otherwise ``sections()`` with ``prompt`` merged over them."""
        if self.system is not None:
            return self.system
        parts = {**self.sections(node, repl), **self.prompt}
        texts = [value(node, repl) if callable(value) else value for value in parts.values()]
        return "\n\n".join(text.strip() for text in texts if text and text.strip())

    # -- what it sees -----------------------------------------------------------

    def messages(self, node: Node, repl: Repl, since: Node | None = None) -> list[dict[str, Any]]:
        """The system prompt, then the run's history up to ``node`` (after ``since``, if given).

        Only the last ``keep`` messages are kept when set. The runs it launched
        and how they are, and the last-step notice, end the last user message.
        """
        history = [m for n in node.transcript(after=since) for m in as_messages(n)]
        if self.keep is not None and len(history) > self.keep:
            kept = history[-self.keep :] if self.keep else []
            while kept and kept[0]["role"] == "tool":
                kept = kept[1:]
            history = [{"role": "user", "content": TRUNCATED}, *kept] if self.keep else []
        notes = [launched(node)]
        if final_turn(node, self.max_iters):
            notes.append(FINAL_TURN)
        if text := "\n\n".join(note for note in notes if note):
            if history and history[-1]["role"] == "user":
                history[-1] = {"role": "user", "content": _append(history[-1]["content"], text)}
            else:
                history.append({"role": "user", "content": text})
        return [{"role": "system", "content": self.system_prompt(node, repl)}, *history]

    # -- what it does -----------------------------------------------------------

    async def step(self, node: Node, llm: LLM, repl: Repl) -> Node:
        """One model call: the code its ``repl`` call passed, for flow to run, or the answer."""
        options = (
            {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort is not None else {}
        )
        reply = await llm(self.model, self.messages(node, repl), tools=[REPL_TOOL], **options)
        out = Node(
            "assistant", reply.text, data={"usage": reply.usage.as_dict(), "model": self.model}
        )
        code, error = parse_code(reply)
        if code or error:
            out.code, out.output, out.error = code or out.code, error, bool(error)
            return out
        return self.done(out, reply.text.strip(), repl, since=node, wait=False)

    def done(self, out: Node, answer: Any, repl: Repl, *, since: Node, wait: bool = True) -> Node:
        """A step that ended without code: the run's result, or a wait for the runs it launched.

        What ``submit()`` got since ``since`` wins. Otherwise ``answer``, checked
        against ``output_schema``, is the result, and runs it launched keep going.
        With ``wait``, a run whose launched runs are still going (or whose results
        it hasn't seen) waits for them instead: a CLI agent waits by ending its turn.
        """
        if (value := repl.submitted(since=since)) is not None:
            out.result = value
        elif not (wait and repl.running()):
            try:
                out.result = (
                    answer
                    if self.output_schema is None
                    else schema.check(answer, self.output_schema)
                )
            except (schema.SchemaError, TypeError) as exc:
                out.output, out.error = f"Your reply is not a valid final answer.\n{exc}", True
        return out


# -- shared helpers -------------------------------------------------------------


def steps(node: Node) -> int:
    """How many steps its run has taken, up to ``node``."""
    return sum(n.ends_turn for n in node.transcript())


def final_turn(node: Node, max_iters: int | None) -> bool:
    """Whether this is the run's last step: ``max_iters`` is one away, or a budget asked to finish."""
    if max_iters is not None and steps(node) == max_iters - 1:
        return True
    for earlier in reversed(node.transcript()):
        if earlier.ends_turn:
            return False
        if earlier.final:
            return True
    return False


def launchable(repl: Repl) -> str:
    """The Flow's agents, by name, as a run choosing one to launch sees them."""
    agents = ", ".join(f"`{name}` ({agent.summary()})" for name, agent in repl.flow.agents.items())
    return f"Agents: {agents}" if agents else ""


def launched(node: Node) -> str:
    """The runs ``node``'s run launched and how they are."""
    children = node.root.children()
    if not children:
        return ""
    lines = [
        f"- `{child.name}` ({child.path()}): "
        + ("result ready" if child.status() == "done" else child.status())
        for child in children
    ]
    return "\n".join(["Background agents:", *lines])


def as_messages(node: Node) -> list[dict[str, Any]]:
    """One node as chat messages: its text and ``repl`` call, then what running it printed.

    The call's id is the node's, so the history needs no provider ids.
    """
    messages: list[dict[str, Any]] = []
    images = node.data.get(IMAGES_KEY)
    if node.role == "user":
        if node.text:
            messages.append({"role": "user", "content": node.text})
    elif node.runs_code:
        call = {"name": REPL_TOOL["function"]["name"], "arguments": json.dumps({"code": node.code})}
        messages.append(
            {
                "role": "assistant",
                "content": node.text,
                "tool_calls": [{"id": node.id, "type": "function", "function": call}],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": node.id, "content": node.output or "(no output)"}
        )
        if images:
            messages.append({"role": "user", "content": with_images("Images it attached:", images)})
        return messages
    elif node.text or "model" in node.data:
        messages.append({"role": "assistant", "content": node.text})
    if node.output:
        content = with_images(node.output, images) if images else node.output
        messages.append({"role": "user", "content": content})
    return messages


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


def _append(content: str | list[dict[str, Any]], text: str) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return f"{content}\n\n{text}"
    return [*content, {"type": "text", "text": text}]


def repl_reply(code: str, text: str = "") -> Reply:
    """A reply that calls ``repl`` with ``code`` after ``text``: what a scripted model returns to act."""
    name = REPL_TOOL["function"]["name"]
    return Reply(text, (ToolCall(name, json.dumps({"code": code})),))


def parse_code(reply: Reply) -> tuple[str, str]:
    """``(code, error)`` for a reply: the code its one ``repl`` call passed, or why it ran nothing.

    A non-empty reply that calls no tool is an answer: both are empty.
    """
    if not reply.calls and reply.text.strip():
        return "", ""
    reason = "the reply is empty"
    if len(reply.calls) > 1:
        reason = f"it made {len(reply.calls)} tool calls; call `repl` once per response"
    elif reply.calls:
        call = reply.calls[0]
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            arguments = None
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if call.name != REPL_TOOL["function"]["name"]:
            reason = (
                f"`{call.name}` is not a tool: the only tool is `repl`, and the functions "
                "listed are called from the code you pass it"
            )
        elif not isinstance(code, str):
            reason = "the `repl` call has no `code` string (the reply may have been cut off)"
        elif not code.strip():
            reason = "the `repl` call's code is empty"
        else:
            return code, ""
    return "", (
        f"{reason}. Nothing ran. To act, call `repl` once with non-empty code. To finish, "
        "call submit(value) in that code, or reply in plain text without calling `repl`."
    )


def last_session(node: Node) -> Node | None:
    """The newest node of ``node``'s run that ended a session step, unless code ran after it."""
    for earlier in reversed(node.root.nodes()):
        if earlier.data.get("session") and earlier.ends_turn:
            return earlier
        if earlier.runs_code:
            return None
    return None


def forked(last: Node, repl: Repl) -> bool:
    """Whether ``last``'s session ran under another REPL: a fork gave the run a new one."""
    return last.data.get("repl", last.root_id) != repl.id


def as_prompt(messages: list[dict[str, Any]]) -> str:
    """Chat messages (without the system message) as one text prompt for a CLI agent."""
    parts = []
    for message in messages:
        content = message["content"] or ""
        if not isinstance(content, str):
            content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
        for call in message.get("tool_calls") or ():
            code = json.loads(call["function"]["arguments"]).get("code", "")
            content = f"{content}\n```python\n{code}\n```".strip()
        if not content:
            continue
        parts.append(content if message["role"] != "assistant" else f"[you]\n{content}")
    return "\n\n".join(parts) or "Continue."
