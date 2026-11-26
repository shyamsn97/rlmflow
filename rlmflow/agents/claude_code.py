"""Claude Code as a harness: one Claude Agent SDK ``query`` per step, resuming its session.

Needs ``rlmflow[claude]`` (``claude-agent-sdk``), which bundles the Claude Code
CLI; sign in once with ``claude`` (or set ``ANTHROPIC_API_KEY``).

Claude runs with your own settings (permissions, ``CLAUDE.md``, MCP servers,
plugins); rlmflow only allows ``rlmflow`` commands on top of them, and turns off
Claude Code's own cross-session messaging (``SendMessage``, ``ListAgents``), so
agents message each other with ``Agent.send`` and every message is in the graph.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from rlmflow.agents.agent import (
    LLM,
    Agent,
    as_prompt,
    forked,
    last_session,
)
from rlmflow.agents.prompts import USE_RLMFLOW_REPL
from rlmflow.graph.nodes import Node

if TYPE_CHECKING:
    from rlmflow.repl import Repl

NATIVE_MESSAGING_TOOLS = ("SendMessage", "ListAgents")
#: The SDK's default (1 MB) is less than one screenshot or large file read in a message.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
SESSION_NAMESPACE = uuid.UUID("5f0c1d1e-4b8e-4c1a-9a57-2f61b3f1e0aa")


class ClaudeCode(Agent):
    """Claude Code, through the Claude Agent SDK.

    ``options`` are ``ClaudeAgentOptions`` fields (``permission_mode``,
    ``cli_path``, ``max_turns``, ``plugins``, ...), applied last;
    ``allowed_tools`` and ``disallowed_tools`` there are extended, not
    replaced. ``mcp`` also offers the agent tools over MCP, with ``repl`` as
    ``python(code)``.
    """

    protocol = USE_RLMFLOW_REPL

    def __init__(
        self,
        model: str | None = None,
        *,
        options: dict[str, Any] | None = None,
        mcp: bool = False,
        **shared: Any,
    ) -> None:
        super().__init__(model, **shared)
        self.options = dict(options or {})
        self.mcp = mcp

    async def step(self, node: Node, llm: LLM, repl: Repl) -> Node:
        try:
            from claude_agent_sdk import ResultMessage, query
        except ImportError as exc:
            raise RuntimeError("ClaudeCode needs `pip install 'rlmflow[claude]'`") from exc

        last = last_session(node)
        system, *new = self.messages(node, repl, since=last)
        result = None
        options = self.sdk_options(system["content"], repl, last)
        async for message in query(prompt=as_prompt(new), options=options):
            if isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise RuntimeError("claude ended without a result")
        if result.is_error:
            raise RuntimeError(
                f"claude failed: {result.result or '; '.join(result.errors or []) or result.subtype}"
            )
        usage = result.usage or {}
        out = Node(
            "assistant",
            result.result or "",
            data={
                "session": result.session_id,
                "repl": repl.id,
                "usage": {
                    "input_tokens": int(usage.get("input_tokens") or 0)
                    + int(usage.get("cache_read_input_tokens") or 0)
                    + int(usage.get("cache_creation_input_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or 0),
                },
            },
        )
        if result.total_cost_usd is not None:
            out.data["cost_usd"] = result.total_cost_usd
        answer = result.structured_output
        return self.done(out, answer if answer is not None else out.text, repl, since=node)

    def sdk_options(self, system: str, repl: Repl, last: Node | None) -> Any:
        """The ``ClaudeAgentOptions`` for one step."""
        from claude_agent_sdk import ClaudeAgentOptions

        env = repl.environ
        fields: dict[str, Any] = {
            "system_prompt": {"type": "preset", "preset": "claude_code", "append": system},
            "cwd": repl.path,
            "env": env,
            "max_buffer_size": MAX_MESSAGE_BYTES,
        }
        if last is not None:
            fields.update(resume=last.data["session"], fork_session=forked(last, repl))
        else:
            run = repl.root
            fields["session_id"] = str(
                uuid.uuid5(SESSION_NAMESPACE, f"{repl.id}:{len(run.nodes())}")
            )
        if self.model is not None:
            fields["model"] = self.model
        if self.output_schema is not None:
            fields["output_format"] = {"type": "json_schema", "schema": self.output_schema}
        if self.mcp:
            from rlmflow.agent_tools.mcp import mcp_config

            fields["mcp_servers"] = mcp_config(env)["mcpServers"]
        fields.update(self.options)
        fields["allowed_tools"] = [
            *self.options.get("allowed_tools", []),
            "Bash(rlmflow:*)",
            *(["mcp__rlmflow"] if self.mcp else []),
        ]
        fields["disallowed_tools"] = [
            *self.options.get("disallowed_tools", []),
            *NATIVE_MESSAGING_TOOLS,
        ]
        return ClaudeAgentOptions(**fields)


__all__ = ["ClaudeCode"]
