"""Delegation and REPL access as plain functions of ``(flow, step)``.

``step.agent`` is the caller. No function knows which harness called it: every
agent reaches them through its REPL, ours or the one external harnesses run
with ``rlmflow repl``. Launches and messages hang off the running cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import Agent, Node, Step, current_code

if TYPE_CHECKING:
    from rlmflow.flow import Flow


def spawn(
    flow: Flow,
    step: Step,
    task: str,
    *,
    config: str | None = None,
    name: str | None = None,
    **args: Any,
) -> Agent:
    """Start an agent launched by the caller's running cell."""
    g, parent = step.graph, step.agent
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if not can_spawn(g, parent):
        raise ValueError(f"cannot start agents beyond max depth {parent.config.max_depth}")
    if len(task) > parent.config.max_query_chars:
        raise ValueError(f"task exceeds {parent.config.max_query_chars} characters")
    if flow.budget_exhausted(g, parent):
        raise RuntimeError("cannot start an agent after max_budget is exhausted")
    if config is not None and not isinstance(config, str):
        raise TypeError("agents started by an agent take a config name, not an AgentConfig")
    return flow.spawn(step, task, config=config or "default", name=name, **args)


def send(flow: Flow, step: Step, target: object, text: str) -> Node:
    """Queue a ``message`` for another agent; it reopens a finished agent."""
    from rlmflow.agent_tools.view import resolve_agent

    if not isinstance(text, str):
        raise TypeError("message text must be a string")
    agent = resolve_agent(step.graph, target)
    message = step.graph.send(agent, text, by=current_code() or step.agent)
    flow.wake()
    return message


def done(flow: Flow, step: Step, result: Any) -> Any:
    """Check ``result`` against the caller's ``output_schema`` and return the value to record."""
    from rlmflow.schema import as_json_value, check

    schema = step.agent.config.output_schema
    return check(result, schema) if schema is not None else as_json_value(result)


async def repl(flow: Flow, step: Step, code: str) -> dict[str, Any]:
    """Run ``code`` in the caller's REPL; its ``code`` node lands in the caller's history."""
    node, run = await flow.execute(step, code)
    flow.wake()
    return {
        "output": node.data["output"],
        "error": node.data["error"],
        "submitted": run.submitted,
        "result": run.answer if run.submitted else None,
    }


__all__ = ["done", "repl", "send", "spawn"]
