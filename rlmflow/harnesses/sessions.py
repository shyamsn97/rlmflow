"""What external harnesses share: their prompts, which session a turn runs in, and what a turn meant.

A ``SessionHarness`` runs one session turn per step. During the turn the session
reaches the agent tools through ``rlmflow``; each ``rlmflow repl`` call is its
own node. When the turn ends, ``outcome`` appends the node that ends it:
terminal, or idle until something arrives.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlmflow import schema
from rlmflow.graph.nodes import SELF, Agent, Node, Step
from rlmflow.graph.queries import unseen_results
from rlmflow.harnesses.base import Harness
from rlmflow.harnesses.prompts import (
    LOCATION_PROMPT,
    RECORD_PROMPT,
    RESUME_PROMPT,
    STATE_PROMPT,
    TOOLS_PROMPT,
    argument_manifest,
    arguments,
    guidance,
    preview,
)
from rlmflow.runtime.runtime import repl_key

if TYPE_CHECKING:
    from rlmflow.agent_tools.server import Turn
    from rlmflow.flow import Flow

SESSION_NAMESPACE = uuid.UUID("5f0c1d1e-4b8e-4c1a-9a57-2f61b3f1e0aa")

#: Prints the user's REPL variables with their types and sizes; never recorded.
DESCRIBE_VARIABLES = """\
def _rlmflow_describe(names):
    import types
    rows = []
    for name, value in list(globals().items()):
        if name.startswith("_") or name in names or callable(value) or isinstance(value, types.ModuleType):
            continue
        kind = type(value).__name__
        if isinstance(value, str):
            kind += ", %d chars" % len(value)
        elif hasattr(value, "__len__"):
            try:
                kind += "[%d]" % len(value)
            except TypeError:
                pass
        rows.append("%s (%s)" % (name, kind))
    print(", ".join(rows) or "(none)")
_rlmflow_describe({hidden!r})
del _rlmflow_describe"""


def turn_end(step: Step) -> Node | None:
    """The node that ended the latest session turn, if no code ran after it."""
    for node in reversed(step.agent.nodes()):
        if node.data.get("session_id") and "seen" in node.data:
            return node
        if node.runs_code:
            return None
    return None


@dataclass(frozen=True)
class Session:
    """The session a turn runs in, and whether it resumes or forks an earlier one."""

    id: str
    resume: bool = False
    fork: bool = False

    @classmethod
    def for_step(cls, step: Step) -> Session:
        """A turn resumes the session that ended the previous turn, forking it when
        that turn ran under another ``repl_id`` (a fork gave this agent a new one).
        Otherwise it starts a fresh session whose id is derived from the ``repl_id``
        and how many nodes the agent has, so a retried turn reuses the id.
        """
        repl = repl_key(step.agent)
        last = turn_end(step)
        if last is not None:
            return cls(
                last.data["session_id"], resume=True, fork=last.data.get("repl", last.owner) != repl
            )
        return cls(str(uuid.uuid5(SESSION_NAMESPACE, f"{repl}:{len(step.agent.nodes())}")))


@dataclass(frozen=True)
class TurnResult:
    """What a finished session turn reported."""

    session_id: str
    text: str = ""
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    #: The answer a session gave against ``output_schema``, if it gave one.
    structured: Any = None


@dataclass(frozen=True)
class SessionOptions:
    """``harness_options`` every command-line session harness takes."""

    #: The executable to run instead of the harness's default.
    command: str = ""


def config_list(flow: Flow) -> str:
    return "; ".join(
        f"`{name}`" + (f" ({config.description})" if config.description else "")
        for name, config in flow.configs.items()
    )


def first_prompt(step: Step) -> str:
    """The task, and the arguments by name and size; their values are REPL variables."""
    task = step.agent.task
    manifest = argument_manifest(step)
    return f"{task}\n\n{manifest}" if manifest else task


def pending_messages(step: Step) -> list[str]:
    """What others wrote since the agent's last turn began."""
    turn = step.agent.last_turn()
    start = turn.data["seen"] if turn is not None else 1
    return [
        node.text
        for node in step.agent.nodes()[start:]
        if node.producer != SELF and node.text and not isinstance(node, Agent)
    ]


async def describe_variables(flow: Flow, step: Step) -> str:
    await flow.prepare_repl(step.graph, step.agent)
    hidden = sorted(flow.tools) + ["ENV", "AGENTS", "Input", "task"]
    code = DESCRIBE_VARIABLES.replace("{hidden!r}", repr(hidden))
    run = await flow.runtime.execute(step.agent, code, step.graph)
    return run.output.strip() or "(none)"


async def resume_prompt(flow: Flow, step: Step) -> str:
    """A fresh session on an existing REPL: the code run so far and the variables they left."""
    ran = [n for n in step.agent.nodes() if n.runs_code]
    lines = [
        f"[{i}] {preview(c.code, 80)} -> {preview(c.output or '(no output)', 100)}"
        for i, c in enumerate(ran, 1)
    ]
    return RESUME_PROMPT.format(
        ran="\n".join(lines), variables=await describe_variables(flow, step)
    )


async def next_prompt(flow: Flow, step: Step, session: Session) -> str:
    """The prompt that opens this turn."""
    messages = pending_messages(step)
    if session.resume:
        return "\n\n".join(messages) or "Continue."
    has_code = any(n.runs_code for n in step.agent.nodes())
    if not has_code and step.agent.last_turn() is None:
        return first_prompt(step)
    opening = await resume_prompt(flow, step) if has_code else ""
    return "\n\n".join(
        part for part in (f"Your task: {step.agent.task}", opening, *messages) if part
    )


class SessionHarness(Harness):
    """A harness that runs one session turn per step; subclasses implement ``run_turn``."""

    def prompt_sections(self, flow: Flow, step: Step) -> dict[str, str | None]:
        """Appended to the session's own system prompt."""
        from rlmflow.tools import format_tool_line, tool_allowed

        variables = ", ".join(f"`{name}`" for name in ["task", *arguments(step)])
        own_tools = [
            line
            for fn in flow.tools.values()
            if tool_allowed(fn, flow, step.node) and (line := format_tool_line(fn))
        ]
        location = flow.runtime.location
        output_schema = step.config.output_schema
        return {
            "protocol": TOOLS_PROMPT.format(arguments=variables),
            "configs": f"Configs for launch_agent (default `default`): {config_list(flow)}",
            "tools": "Also in the REPL:\n" + "\n".join(own_tools) if own_tools else None,
            "location": LOCATION_PROMPT.format(location=location) if location is not None else None,
            "record": RECORD_PROMPT,
            **guidance(step),
            "schema": schema.contract(output_schema) if output_schema is not None else None,
        }

    async def step(self, flow: Flow, step: Step) -> Node:
        seen = len(step.agent.nodes())
        async with flow.agent_tools.turn(step) as turn:
            result = await self.run_turn(flow, step, turn)
        return await self.outcome(flow, step, turn, result, seen)

    async def run_turn(self, flow: Flow, step: Step, turn: Turn) -> TurnResult:
        raise NotImplementedError

    async def outcome(
        self, flow: Flow, step: Step, turn: Turn, result: TurnResult, seen: int
    ) -> Node:
        """End the turn: terminal if the agent submitted or has nothing to wait for, else idle."""
        g, agent = step.graph, step.agent
        node = Node(
            "assistant",
            result.text,
            data={"session_id": result.session_id, "repl": repl_key(agent), "seen": seen},
        )
        if result.cost_usd is not None:
            node.data["cost_usd"] = result.cost_usd
        if result.usage:
            node.data["usage"] = result.usage
        workspace = await flow.workspaces.snapshot(agent)
        if workspace is not None:
            node.data["workspace"] = workspace
        if turn.submitted:
            node.result, node.data["terminal"] = turn.result, True
        elif unseen_results(g, agent) or not all(c.status() == "done" for c in agent.children()):
            node.data["idle"] = True
        else:
            answer = result.structured if result.structured is not None else result.text
            output_schema = step.config.output_schema
            node.result = schema.check(answer, output_schema) if output_schema else answer
            node.data["terminal"] = True
        return flow.append(g, agent, node)

    async def run_command(
        self, flow: Flow, step: Step, argv: list[str], env: dict[str, str]
    ) -> str:
        """Run one headless session turn in the agent's workspace and return its stdout; cancelling kills it."""
        if "/" not in argv[0] and shutil.which(argv[0], path=env.get("PATH")) is None:
            raise RuntimeError(f"{argv[0]!r} was not found; install it and sign in first")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=flow.workspaces.path(step.agent),
            env={**os.environ, **env},
            stdin=asyncio.subprocess.DEVNULL,
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
        out, err = stdout.decode(errors="replace").strip(), stderr.decode(errors="replace").strip()
        if process.returncode:
            raise RuntimeError(
                f"{argv[0]} failed: {err or out or f'exit status {process.returncode}'}"
            )
        return out


def json_payload(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"{what} returned invalid JSON: {exc}") from exc


__all__ = [
    "LOCATION_PROMPT",
    "RECORD_PROMPT",
    "RESUME_PROMPT",
    "STATE_PROMPT",
    "TOOLS_PROMPT",
    "Session",
    "SessionHarness",
    "SessionOptions",
    "TurnResult",
    "json_payload",
    "next_prompt",
    "pending_messages",
    "resume_prompt",
]
