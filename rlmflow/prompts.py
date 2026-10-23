"""Prompts: model-facing prose, sections and profiles, the default system prompt, and message assembly."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from rlmflow.graph import reads
from rlmflow.graph.nodes import Node, TruncationSummary, current_step
from rlmflow.structured import DEFAULT_OUTPUT_CODEC
from rlmflow.tools.namespace import format_tool_docs

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
  interfaces, assign them to different children before implementing them yourself.
- Launch independent owners together, using the cheapest capable model. Keep work
  local only when it is small or tightly coupled enough that delegation costs more.
- Give a distinct multi-step investigation, implementation, search, or verification
  its own child when separate context or concurrent execution will help.
- Give every child one bounded, non-overlapping deliverable and the arguments it needs.
- When a known work product depends on another, launch it with the prerequisite
  `AgentHandle` as a named argument; do not wait before declaring a known edge.
- Do not duplicate delegated work locally or create a child for light synthesis.
- Consume and validate child results before finishing.
- Stop launching when the answer is supported."""

WORKER_ADDENDUM = """\
Assigned-work policy:
- Own the requested deliverable and use the provided arguments.
- Do not leave TODOs, stubs, or work for a parent to complete.
- Validate the result."""


LOCAL_WORK_EXAMPLE = """\
Example — one coherent task stays local:
```repl
print(task[:500])
```"""

AGENTS_TEXT = """\
`AGENTS` is a read-only snapshot of the agents in this run's graph:

- `AGENTS.get()` returns you; `AGENTS.get(id_or_path_or_name)` finds another agent.
- `AGENTS.get_parent()`, `.get_siblings()`, and `.get_children()` return
  `AgentInfo` objects relative to you (or pass an agent selector).
- `AgentInfo.status` is `blocked`, `running`, `waiting`, `idle`, or `completed`;
  `.dependency_paths`, `.dependent_paths`, and `.blocked_on_paths` expose dataflow.
  `.result()` returns an already-completed result and raises while one is still
  in flight; `await agent.wait_for_result()` waits automatically.
- `AGENTS.print_graph(show_results=True)` prints the whole tree with statuses and
  bounded result previews. Pass `view="dependencies"` for the consumer-rooted
  dependency projection.
- `AGENTS.edges(source=..., target=..., kind=...)` lists application edges
  between agents, such as `peer` links.

The snapshot is refreshed before each REPL action; repeated queries within one
action are not fresher. To act on other agents, use the tools: `send_agent`
queues a message, `ask_agent` sends one and waits for the reply, and
`connect_agent`/`disconnect_agent` add or remove edges."""


class PromptSectionError(ValueError):
    """Invalid section composition."""


class PinnedPromptSectionError(PromptSectionError):
    """A caller attempted to replace or remove a pinned section."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One named contribution to a prompt.

    ``key`` is the update identity; it is intentionally independent of display
    order. ``source`` is provenance for inspection and harness feedback and is
    not rendered into the prompt.
    """

    key: str
    content: str
    order: int = 0
    source: str = ""
    pinned: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise PromptSectionError("prompt section key cannot be empty")


class Prompt:
    """One mutable system or user prompt, assembled from named sections.

    System and user prompts use the same class. The chat-message role belongs
    to the caller that renders the prompt, not to a section.
    """

    def __init__(self, sections: Iterable[PromptSection] = ()) -> None:
        self._sections: dict[str, PromptSection] = {}
        self._positions: dict[str, int] = {}
        self._next_position = 0
        self.extend(sections)

    def __iter__(self) -> Iterator[PromptSection]:
        return iter(self.sections())

    def __contains__(self, key: object) -> bool:
        return key in self._sections

    def __str__(self) -> str:
        return self.render()

    def get(self, key: str) -> PromptSection | None:
        return self._sections.get(key)

    def sections(self) -> tuple[PromptSection, ...]:
        return tuple(
            sorted(
                self._sections.values(),
                key=lambda section: (section.order, self._positions[section.key]),
            )
        )

    def add_section(
        self,
        key: str,
        content: str,
        *,
        order: int = 0,
        source: str = "",
        pinned: bool = False,
    ) -> Prompt:
        """Add or replace a named section and return ``self``."""
        return self.add(
            PromptSection(
                key,
                content,
                order=order,
                source=source,
                pinned=pinned,
            )
        )

    def add(self, section: PromptSection) -> Prompt:
        """Add an already constructed section and return ``self``."""
        existing = self._sections.get(section.key)
        if existing is not None and existing.pinned and section != existing:
            raise PinnedPromptSectionError(
                f"cannot replace pinned prompt section {section.key!r} "
                f"from {existing.source or 'unknown source'}"
            )
        if existing is None:
            self._positions[section.key] = self._next_position
            self._next_position += 1
        self._sections[section.key] = section
        return self

    def extend(self, sections: Iterable[PromptSection]) -> Prompt:
        for section in sections:
            self.add(section)
        return self

    def update_section(
        self,
        key: str,
        content: str,
        *,
        order: int | None = None,
        source: str | None = None,
        pinned: bool | None = None,
    ) -> Prompt:
        """Update an existing section while retaining unspecified metadata."""
        existing = self._sections.get(key)
        if existing is None:
            raise KeyError(key)
        return self.add_section(
            key,
            content,
            order=existing.order if order is None else order,
            source=existing.source if source is None else source,
            pinned=existing.pinned if pinned is None else pinned,
        )

    def remove_section(self, key: str) -> PromptSection | None:
        """Remove and return one section; pinned sections cannot be removed."""
        existing = self._sections.get(key)
        if existing is None:
            return None
        if existing.pinned:
            raise PinnedPromptSectionError(
                f"cannot remove pinned prompt section {key!r} "
                f"from {existing.source or 'unknown source'}"
            )
        del self._sections[key]
        del self._positions[key]
        return existing

    def render(self) -> str:
        """Join non-empty sections with stable blank lines."""
        return "\n\n".join(
            section.content.strip() for section in self.sections() if section.content.strip()
        )

    def copy(self) -> Prompt:
        return Prompt(self.sections())


SystemPromptFn = Callable[[Any, Any], str]
SystemPromptSource = str | SystemPromptFn | Prompt
PromptModifier = Callable[[Prompt, Any, Any], None]
PromptModifierSource = PromptModifier | Iterable[PromptModifier]


def as_system_prompt_fn(source: Any) -> SystemPromptFn:
    """Normalize a system-prompt source to a ``(flow, node) -> str`` function."""
    if isinstance(source, Prompt):
        return lambda _flow=None, _node=None, prompt=source: prompt.render()
    if isinstance(source, str):
        if "{tools}" in source:

            def formatted(flow: Any = None, node: Any = None) -> str:
                tools = flow.render_tools(node) if flow is not None else ""
                return source.format(tools=tools)

            return formatted
        return lambda _flow=None, _node=None, text=source: text
    if callable(source):
        return source
    raise TypeError(f"unsupported system prompt source: {type(source)!r}")


def apply_prompt_modifiers(
    prompt: Prompt,
    source: PromptModifierSource | None,
    flow: Any = None,
    node: Any = None,
) -> None:
    """Apply one modifier or an ordered iterable of modifiers in place."""
    if source is None:
        return
    if callable(source):
        source(prompt, flow, node)
        return
    for modifier in source:
        modifier(prompt, flow, node)


def prompt_for(source: SystemPromptSource, flow: Any, node: Any) -> Prompt:
    """Build a fresh system prompt from a whole-prompt source."""
    if isinstance(source, Prompt):
        return source.copy()
    text = as_system_prompt_fn(source)(flow, node)
    prompt = Prompt()
    prompt.add_section(
        "system.override",
        text,
        order=0,
        source=type(source).__name__,
        pinned=True,
    )
    return prompt


@dataclass
class PromptProfile:
    """Named prompt customization for a child.

    Modifiers edit a system or user :class:`Prompt` in place. ``None`` on each
    field inherits the flow default.
    """

    system: SystemPromptSource | None = None
    system_modifiers: PromptModifierSource | None = None
    user_modifiers: PromptModifierSource | None = None


def build_args_manifest(
    args: dict[str, object],
    descriptions: dict[str, str] | None = None,
) -> str:
    """List arguments by name and size without copying values into the prompt.

    Arguments are direct REPL variables. Large values stay outside the context
    window until the model inspects them programmatically.
    """
    if not args:
        return ""

    def describe(value: object) -> str:
        if isinstance(value, str):
            return f"str, {len(value)} chars"
        if isinstance(value, list):
            return f"list, {len(value)} items"
        if isinstance(value, dict):
            return f"object, {len(value)} fields"
        return type(value).__name__

    descriptions = descriptions or {}
    lines = []
    for name, value in args.items():
        line = f"- `{name}`: {describe(value)}"
        if description := descriptions.get(name):
            line += f" — {' '.join(description.split())}"
        lines.append(line)
    return "Your REPL has these task arguments as direct variables:\n" + "\n".join(lines)


def add_argument_sections(prompt: Prompt, flow: Any, node: Any) -> None:
    """Add per-agent arguments, output schema, and useful depth information."""
    from rlmflow.graph import reads
    from rlmflow.graph.graph import DEPENDS_ON
    from rlmflow.graph.nodes import current_step

    step = current_step()
    if step is None:
        return
    g, agent = step.graph, step.agent
    bound = g.edges(kind=DEPENDS_ON, source=agent.id)
    manifest = build_args_manifest(
        {**agent.resolved_args(), **reads.inputs(g, agent)},
        {
            **agent.argument_descriptions(),
            **{
                edge.data["as"]: edge.data["description"]
                for edge in bound
                if edge.data.get("as") and edge.data.get("description")
            },
        },
    )
    if manifest:
        prompt.add_section(
            "context.arguments",
            manifest,
            order=700,
            source="add_argument_sections",
        )
    from rlmflow.graph.reads import ask_terms

    schema = (ask_terms(g, node) or {}).get("schema", agent.config.output_schema)
    if schema is not None:
        codec = getattr(flow, "output_codec", DEFAULT_OUTPUT_CODEC)
        prompt.add_section(
            "contract.output_schema",
            codec.render_contract(schema, source="submit").strip(),
            order=710,
            source="add_argument_sections",
            pinned=True,
        )
    max_depth = agent.config.max_depth
    if max_depth > 0:
        depth = reads.depth(g, agent)
        depth_text = (
            f"You are using model key **`{agent.config.model}`** at recursion depth "
            f"**{depth}** of max **{max_depth}**."
        )
        if depth >= max_depth:
            depth_text += " You cannot spawn sub-agents."
        prompt.add_section(
            "capabilities.depth",
            depth_text,
            order=720,
            source="add_argument_sections",
            pinned=True,
        )


def render_tools(flow: Any, node: Node) -> str:
    """The ``### Tools`` list for exactly the callables ``node``'s REPL binds."""
    return format_tool_docs(flow.build_tools(node))


def _append_to_last_user(
    current_messages: list[dict[str, str]],
    content: str,
) -> list[dict[str, str]]:
    """Add request-only context without changing a node's durable snapshot."""
    if not content:
        return current_messages
    current_messages = [dict(message) for message in current_messages]
    user_index = next(
        (
            index
            for index in range(len(current_messages) - 1, -1, -1)
            if current_messages[index]["role"] == "user"
        ),
        None,
    )
    if user_index is None:
        current_messages.append({"role": "user", "content": content})
    else:
        existing = current_messages[user_index]["content"].rstrip()
        current_messages[user_index]["content"] = (
            f"{existing}\n\n{content}" if existing else content
        )
    return current_messages


def build_messages(flow: Any, node: Node) -> list[dict[str, str]]:
    """The prompt as of ``node``: system message, then that agent's turns."""
    step = current_step()
    g, agent = step.graph, step.agent
    system_prompt = flow.build_system_prompt(node)
    if node.messages is None:
        user_prompt = flow.default_user_prompt(node)
        node.snapshot_messages(node.build_messages(user_prompt, flow))
    background = [
        f"- `{child.config.name}` ({reads.path(g, child)}): "
        + ("result ready" if reads.terminal(g, child) else "running")
        for child in reads.children(g, agent)
    ]
    current_messages = _append_to_last_user(
        node.message_dicts(),
        "\n".join(["Background subagents:", *background]) if background else "",
    )
    keep = agent.config.keep_n_messages
    if keep is not None:
        current_messages = current_messages[-keep:] if keep > 0 else []
    previous_keep = None if keep is None else max(keep - len(current_messages), 0)
    history = reads.history(g, node)
    previous_messages = reads.project_nodes(history[:-1], keep=previous_keep)
    if keep is not None:
        summary = next(
            (item for item in reversed(history) if isinstance(item, TruncationSummary)),
            None,
        )
        if summary is not None:
            notice = summary.message_dicts()
            visible = [*previous_messages, *current_messages]
            if any(message not in visible for message in notice):
                previous_messages = [*notice, *previous_messages]
    system = system_prompt.render()
    return [
        {"role": "system", "content": system},
        *previous_messages,
        *current_messages,
    ]


__all__ = [
    "AGENTS_TEXT",
    "CORE_PROTOCOL_TEXT",
    "EVIDENCE_POLICY",
    "EXECUTION_POLICY",
    "LOCAL_WORK_EXAMPLE",
    "PinnedPromptSectionError",
    "Prompt",
    "PromptModifierSource",
    "PromptProfile",
    "PromptSection",
    "PromptSectionError",
    "SystemPromptSource",
    "WORKER_ADDENDUM",
    "add_argument_sections",
    "apply_prompt_modifiers",
    "build_args_manifest",
    "build_messages",
    "prompt_for",
    "render_tools",
]
