"""Delegation and REPL access as plain functions of ``(flow, step)``.

``step.agent`` is the caller and ``step.node`` its latest node. No function
knows which harness called it: every agent reaches them through its REPL, ours
or the one external harnesses run with ``rlmflow repl``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import AGENT_START, Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow


def spawn(
    flow: Flow,
    step: Step,
    task: str,
    *,
    config: str | None = None,
    name: str | None = None,
    inputs: Mapping[str, Any] | None = None,
    **args: Any,
) -> Node:
    """Start an agent created by ``step.node``."""
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
    return flow.spawn(step, task, config=config or "default", name=name, inputs=inputs, **args)


def resolve(g: Any, selector: object) -> Node:
    """An agent by node, id, path, or unique name."""
    from rlmflow.agent_tools.view import resolve_agent

    if isinstance(selector, Node) and selector.type == AGENT_START and selector.id in g:
        return selector
    return resolve_agent(g, selector)


def send(flow: Flow, step: Step, target: object, text: str) -> Node:
    """Queue a ``message`` for another agent; it reopens a finished agent."""
    if not isinstance(text, str):
        raise TypeError("message text must be a string")
    agent = resolve(step.graph, target)
    message = step.graph.send(agent, text, sender=step.node)
    flow.wake()
    return message


def done(flow: Flow, step: Step, result: Any) -> Any:
    """Check ``result`` against the caller's ``output_schema`` and return the value to record."""
    from rlmflow.schema import as_json_value, check

    schema = step.agent.config.output_schema
    return check(result, schema) if schema is not None else as_json_value(result)


async def repl(flow: Flow, step: Step, code: str) -> dict[str, Any]:
    """Run ``code`` in the caller's REPL and land its ``code`` node."""
    node, run = await flow.execute(step, code)
    flow.land(step.graph, step.agent, node)
    return {
        "output": node.data["output"],
        "error": node.data["error"],
        "submitted": run.submitted,
        "result": run.answer if run.submitted else None,
    }


__all__ = [
    "done",
    "repl",
    "resolve",
    "send",
    "spawn",
]
