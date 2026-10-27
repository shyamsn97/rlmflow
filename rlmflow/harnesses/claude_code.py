"""Claude Code as a harness: one Claude Agent SDK ``query`` per step.

Needs ``rlmflow[claude]`` (``claude-agent-sdk``), which bundles the Claude Code
CLI; sign in once with ``claude`` (or set ``ANTHROPIC_API_KEY``).

``harness_options``:

- ``tools``: ``"strict"`` (the default) lets Claude read with its own tools but
  change things only through ``rlmflow``, so every change is a ``code`` node.
  ``"loose"`` also allows its edit tools and any shell command; the workspace
  snapshot at each cell and at the turn's end still captures the files.
- ``mcp``: also offer the agent tools over MCP, with ``repl`` as ``python(code)``.
- ``command``: a ``claude`` executable to use instead of the SDK's bundled one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import Step
from rlmflow.harnesses.sessions import (
    Session,
    SessionHarness,
    SessionOptions,
    TurnResult,
    next_prompt,
    system_prompt,
)

if TYPE_CHECKING:
    from rlmflow.agent_tools.server import Turn
    from rlmflow.flow import Flow

READ_TOOLS = ["Read", "Grep", "Glob", "LS"]
EDIT_TOOLS = ["Edit", "Write", "NotebookEdit"]


@dataclass(frozen=True)
class ClaudeCodeOptions(SessionOptions):
    command: str | None = None
    mcp: bool = False

    def permissions(self) -> dict[str, Any]:
        """The SDK's ``allowed_tools``, ``disallowed_tools`` and ``permission_mode``."""
        extra = ["mcp__rlmflow"] if self.mcp else []
        if self.strict:
            return {"allowed_tools": ["Bash(rlmflow:*)", *READ_TOOLS, *extra], "disallowed_tools": EDIT_TOOLS}
        return {"allowed_tools": ["Bash", *READ_TOOLS, *EDIT_TOOLS, *extra], "permission_mode": "acceptEdits"}


class ClaudeCodeHarness(SessionHarness):
    name = "claude_code"
    Options = ClaudeCodeOptions

    async def run_turn(self, flow: Flow, step: Step, turn: Turn) -> TurnResult:
        try:
            from claude_agent_sdk import ResultMessage, query
        except ImportError as exc:
            raise RuntimeError("the claude_code harness needs `pip install 'rlmflow[claude]'`") from exc

        options = self.options(step)
        session = Session.for_step(step)
        prompt = await next_prompt(flow, step, session)
        result = None
        async for message in query(prompt=prompt, options=self.sdk_options(flow, step, options, session, turn.env)):
            if isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise RuntimeError("claude ended without a result")
        return self.parse(result)

    def sdk_options(
        self, flow: Flow, step: Step, options: ClaudeCodeOptions, session: Session, env: dict[str, str]
    ) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        config = step.agent.config
        prompt = system_prompt(flow, step, strict=options.strict)
        fields: dict[str, Any] = {
            "system_prompt": {"type": "preset", "preset": "claude_code", "append": prompt},
            "cwd": flow.workspaces.path(step.agent),
            "env": env,
            "cli_path": options.command,
            **options.permissions(),
        }
        if session.resume:
            fields.update(resume=session.id, fork_session=session.fork)
        else:
            fields["session_id"] = session.id
        if config.model != "default":
            fields["model"] = config.model
        if config.output_schema is not None:
            fields["output_format"] = {"type": "json_schema", "schema": config.output_schema}
        if options.mcp:
            from rlmflow.agent_tools.mcp import mcp_config

            fields["mcp_servers"] = mcp_config(env)["mcpServers"]
        return ClaudeAgentOptions(**fields)

    @staticmethod
    def parse(result: Any) -> TurnResult:
        """A ``TurnResult`` from the SDK's ``ResultMessage``."""
        if result.is_error:
            raise RuntimeError(f"claude failed: {result.result or '; '.join(result.errors or []) or result.subtype}")
        usage = result.usage or {}
        return TurnResult(
            session_id=result.session_id,
            text=result.result or "",
            structured=result.structured_output,
            cost_usd=result.total_cost_usd,
            usage={
                "input_tokens": int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            },
        )


__all__ = ["ClaudeCodeHarness", "ClaudeCodeOptions"]
