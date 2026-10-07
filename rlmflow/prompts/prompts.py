"""Default system prompt construction and prompt customization hooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rlmflow.prompts.sections import Prompt
from rlmflow.prompts.text import CORE_PROTOCOL_TEXT
from rlmflow.structured import DEFAULT_OUTPUT_CODEC

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
    agent = node.parent_agent
    if agent is None:
        return
    manifest = build_args_manifest(
        agent.resolved_args(),
        agent.argument_descriptions(),
    )
    if manifest:
        prompt.add_section(
            "context.arguments",
            manifest,
            order=700,
            source="add_argument_sections",
        )
    schema = agent.config.output_schema
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
        depth_text = (
            f"You are using model key **`{agent.config.model}`** at recursion depth "
            f"**{agent.config.depth}** of max **{max_depth}**."
        )
        if agent.config.depth >= max_depth:
            depth_text += " You cannot spawn sub-agents."
        prompt.add_section(
            "capabilities.depth",
            depth_text,
            order=720,
            source="add_argument_sections",
            pinned=True,
        )


__all__ = [
    "CORE_PROTOCOL_TEXT",
    "PromptModifierSource",
    "PromptProfile",
    "SystemPromptSource",
    "add_argument_sections",
    "build_args_manifest",
    "apply_prompt_modifiers",
    "prompt_for",
]
