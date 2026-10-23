"""Claude Code's headless CLI as an agent backend."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph import reads
from rlmflow.graph.nodes import BackendRun, RepairQuery, VerificationFailure, current_step

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.nodes import AgentStart


class ClaudeCodeBackend:
    """Run agents with ``claude -p`` in a shared working directory."""

    def __init__(
        self,
        command: str | Sequence[str] = "claude",
        *,
        working_directory: str | Path | None = None,
        permission_mode: str | None = None,
        allowed_tools: Sequence[str] | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.command = (command,) if isinstance(command, str) else tuple(command)
        if not self.command or not all(isinstance(item, str) and item for item in self.command):
            raise ValueError("command must contain at least one non-empty argument")
        self._uses_flow_directory = working_directory is None
        self.working_directory = Path(working_directory) if working_directory is not None else None
        self.permission_mode = permission_mode
        self.allowed_tools = tuple(allowed_tools or ())
        self.extra_args = tuple(extra_args)

    def bind(self, flow: Flow) -> None:
        """Adopt the Flow runtime's working directory when none was supplied."""
        if self._uses_flow_directory:
            self.working_directory = flow.runtime.working_directory or Path.cwd()

    async def run(
        self,
        agent: AgentStart,
        action: BackendRun,
        *,
        resume: bool,
    ) -> Any:
        executable = self.command[0]
        if "/" not in executable and shutil.which(executable) is None:
            raise RuntimeError(
                f"Claude Code executable {executable!r} was not found; "
                "install Claude Code and authenticate it first"
            )

        prompt = self._prompt(agent, action, resume=resume)
        argv = [*self.command, "-p", prompt, "--output-format", "json"]
        if resume:
            argv += ["--resume", action.run_id]
        else:
            argv += ["--session-id", action.run_id]
        schema = agent.config.output_schema
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema, sort_keys=True, separators=(",", ":"))]
        if self.permission_mode is not None:
            argv += ["--permission-mode", self.permission_mode]
        if self.allowed_tools:
            argv += ["--allowedTools", *self.allowed_tools]
        argv += self.extra_args

        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.working_directory,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise

        text = stdout.decode(errors="replace").strip()
        error_text = stderr.decode(errors="replace").strip()
        if process.returncode:
            detail = error_text or text or f"exit status {process.returncode}"
            raise RuntimeError(f"Claude Code failed: {detail}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Claude Code returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Claude Code result must be a JSON object")
        if schema is not None:
            if "structured_output" not in payload:
                raise RuntimeError("Claude Code result has no structured_output")
            return payload["structured_output"]
        if "result" not in payload:
            raise RuntimeError("Claude Code result has no result")
        return payload["result"]

    @staticmethod
    def _prompt(
        agent: AgentStart,
        action: BackendRun,
        *,
        resume: bool,
    ) -> str:
        parts = [agent.content]
        if resume:
            parts.append(
                "Resume this interrupted run. Inspect the existing session state, "
                "do not repeat changes that are already complete, finish only "
                "remaining work, and return the final result."
            )
        graph = current_step().graph
        inputs = {**agent.resolved_args(), **reads.inputs(graph, agent)}
        if inputs:
            parts.append(
                "Inputs:\n```json\n" + json.dumps(inputs, indent=2, ensure_ascii=False) + "\n```"
            )
        descriptions = agent.argument_descriptions()
        if descriptions:
            parts.append(
                "Input descriptions:\n```json\n"
                + json.dumps(descriptions, indent=2, ensure_ascii=False)
                + "\n```"
            )
        history = reads.history(graph, action)
        source = history[-2] if len(history) > 1 else None
        failure = history[-3] if len(history) > 2 else None
        if isinstance(source, RepairQuery) and isinstance(failure, VerificationFailure):
            parts.append(
                "The previous candidate was rejected. Return a corrected result.\n"
                f"Feedback: {failure.feedback}\n"
                f"Previous candidate: {failure.candidate!r}"
            )
        return "\n\n".join(parts)


__all__ = ["ClaudeCodeBackend"]
