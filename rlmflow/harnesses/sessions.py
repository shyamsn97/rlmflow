"""What external harnesses share: their prompts, which session a turn runs in, and what a turn meant.

A ``SessionHarness`` runs one session turn per step. During the turn the session
reaches the agent tools through ``rlmflow``; each ``rlmflow repl`` cell lands its
own ``code`` node. When the turn ends, ``outcome`` turns it into a ``done`` or
``waiting`` node.
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
from rlmflow.engine import handoff
from rlmflow.graph.nodes import AGENT_START, CODE, DONE, MESSAGE, WAITING, Node, Step
from rlmflow.harnesses.base import Harness
from rlmflow.harnesses.prompts import argument_manifest, preview

if TYPE_CHECKING:
    from rlmflow.agent_tools.server import Turn
    from rlmflow.flow import Flow

SESSION_NAMESPACE = uuid.UUID("5f0c1d1e-4b8e-4c1a-9a57-2f61b3f1e0aa")

TOOLS_PROMPT = """\
You are one agent in a multi-agent run, with a persistent Python REPL.

Run code with `rlmflow repl 'CODE'`, or for several lines:
    rlmflow repl <<'EOF'
    ...
    EOF
Quote the heredoc marker ('EOF') so the shell leaves the code alone. Variables persist
between calls. Your inputs are already variables: {arguments}.
Only printed output comes back, so print summaries, not whole values.

In the REPL:
- launch_agent(task, config=NAME, **args) starts an agent and returns a handle. Configs (default `default`): {configs}
  A handle's `.status`, `.done()` and `.result()` are as of when the cell started, so they
  never change inside a loop; `await handle` waits for its result.
- llm_query(prompt) asks a model and returns its answer.
- AGENTS shows the agents you started and their status.
- submit(value) records your final result. Call it exactly once, when you're finished.
From the shell, `rlmflow agents` prints the tree and `rlmflow done VALUE` is submit.

To wait for agents you started, end your turn without submitting. You'll be resumed
with their results. Don't poll; `rlmflow wait` gives up after a minute. A cell that
awaits long work can hit your tool timeout, so launch agents and end the turn instead."""

STATE_PROMPT = """\
Keep your working state in REPL variables, not only in your replies: put intermediate
results in clearly named variables (for example `candidate_files`, `matches_by_chunk`),
and keep a `notes` string with your plan and what you've ruled out. Update `notes`
before ending a turn. A later session may continue from this REPL without your
conversation."""

RECORD_PROMPT = """\
Cells you run with `rlmflow repl` are recorded in this run, so it can be replayed and
forked from them; changes you make with your own tools are not."""

LOCATION_PROMPT = """\
Your REPL runs in {location}. Your own tools run on this machine; reach the REPL's
files and programs through `rlmflow repl`, with paths as the REPL sees them."""

RESUME_PROMPT = """\
You are continuing an earlier session on this task. Its conversation is gone, but its
REPL is intact. Cells run so far (outputs truncated):

{cells}

REPL variables: {variables}
Start with `rlmflow repl 'print(notes)'`."""

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


def own_history(step: Step) -> list[Node]:
    return step.graph.history(step.node)


def turn_end(step: Step) -> Node | None:
    """The latest node that ended a session turn, if no cell ran after it."""
    for node in reversed(own_history(step)):
        if node.type in (WAITING, DONE) and node.data.get("session_id"):
            return node
        if node.type == CODE:
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
        that turn belongs to another agent (this agent is a fork). Otherwise it starts
        a fresh session whose id is derived from the agent and its place in history,
        so a retried turn reuses the id.
        """
        last = turn_end(step)
        if last is not None:
            return cls(last.data["session_id"], resume=True, fork=last.agent_id != step.agent.id)
        return cls(str(uuid.uuid5(SESSION_NAMESPACE, f"{step.agent.id}:{len(own_history(step))}")))


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


def system_prompt(flow: Flow, step: Step) -> str:
    """The prompt appended to every session: the tools, then keeping state, then ``config.system_prompt``."""
    names = list({**step.agent.resolved_args(), **step.graph.inputs(step.agent)})
    arguments = ", ".join(f"`{name}`" for name in ["task", *names])
    from rlmflow.tools import format_tool_line, tool_allowed

    parts = [TOOLS_PROMPT.format(arguments=arguments, configs=config_list(flow))]
    own_tools = [line for fn in flow.tools.values() if tool_allowed(fn, flow, step.node) and (line := format_tool_line(fn))]
    if own_tools:
        parts.append("Also in the REPL:\n" + "\n".join(own_tools))
    if flow.runtime.location is not None:
        parts.append(LOCATION_PROMPT.format(location=flow.runtime.location))
    parts += [STATE_PROMPT, RECORD_PROMPT]
    if step.agent.config.output_schema is not None:
        parts.append(schema.contract(step.agent.config.output_schema))
    if step.agent.config.system_prompt:
        parts.append(step.agent.config.system_prompt)
    return "\n\n".join(parts)


def first_prompt(step: Step) -> str:
    """The task, and the arguments by name and size; their values are REPL variables."""
    manifest = argument_manifest(step)
    return f"{step.agent.task}\n\n{manifest}" if manifest else step.agent.task


def pending_messages(step: Step) -> list[str]:
    """Messages that arrived since the agent's last turn."""
    texts: list[str] = []
    for node in reversed(own_history(step)):
        if node.type != MESSAGE:
            break
        texts.append(node.data["text"])
    return texts[::-1]


async def describe_variables(flow: Flow, step: Step) -> str:
    await flow.prepare_repl(step.graph, step.agent)
    hidden = sorted(flow.tools) + ["ENV", "AGENTS", "Input", "task"]
    code = DESCRIBE_VARIABLES.replace("{hidden!r}", repr(hidden))
    run = await flow.runtime.execute(step.agent, code, step.graph)
    return run.output.strip() or "(none)"


async def resume_prompt(flow: Flow, step: Step) -> str:
    """A fresh session on an existing REPL: the cells so far and the variables they left."""
    cells = [n for n in own_history(step) if n.type in (CODE, DONE) and n.data.get("code")]
    lines = [
        f"[cell {i}] {preview(c.data['code'], 80)} -> {preview(c.data.get('output') or '(no output)', 100)}"
        for i, c in enumerate(cells, 1)
    ]
    return RESUME_PROMPT.format(cells="\n".join(lines), variables=await describe_variables(flow, step))


async def next_prompt(flow: Flow, step: Step, session: Session) -> str:
    """The prompt that opens this turn."""
    messages = pending_messages(step)
    if session.resume:
        return "\n\n".join(messages) or "Continue."
    has_cells = any(n.type == CODE for n in own_history(step))
    if not has_cells and step.node.type == AGENT_START:
        return first_prompt(step)
    opening = await resume_prompt(flow, step) if has_cells else ""
    return "\n\n".join(part for part in (f"Your task: {step.agent.task}", opening, *messages) if part)


class SessionHarness(Harness):
    """A harness that runs one session turn per step; subclasses implement ``run_turn``."""

    async def step(self, flow: Flow, step: Step) -> Node:
        async with flow.agent_tools.turn(step) as turn:
            result = await self.run_turn(flow, step, turn)
        return await self.outcome(flow, step, turn, result)

    async def run_turn(self, flow: Flow, step: Step, turn: Turn) -> TurnResult:
        raise NotImplementedError

    async def outcome(self, flow: Flow, step: Step, turn: Turn, result: TurnResult) -> Node:
        """``done`` if the agent submitted or has nothing to wait for, else ``waiting`` on its children."""
        g, agent = step.graph, step.agent
        data: dict[str, Any] = {"session_id": result.session_id, "text": result.text}
        if result.cost_usd is not None:
            data["cost_usd"] = result.cost_usd
        if result.usage:
            data["usage"] = result.usage
        workspace = await flow.workspaces.snapshot(agent)
        if workspace is not None:
            data["workspace"] = workspace
        if turn.submitted:
            return Node(DONE, result=turn.result, **data)
        # Children still running, or finished with results this agent hasn't seen.
        unseen = {child.id for child in handoff.pending(g, agent)}
        on = [c.id for c in g.launched(agent) if c.id in unseen or not g.terminal(c)]
        if on:
            return Node(WAITING, on=on, **data)
        answer = result.structured if result.structured is not None else result.text
        output_schema = agent.config.output_schema
        return Node(DONE, result=schema.check(answer, output_schema) if output_schema else answer, **data)

    async def run_command(self, flow: Flow, step: Step, argv: list[str], env: dict[str, str]) -> str:
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
            raise RuntimeError(f"{argv[0]} failed: {err or out or f'exit status {process.returncode}'}")
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
    "system_prompt",
]
