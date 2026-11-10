"""Delegation and REPL access as plain functions of ``(flow, step)``.

``step.agent`` is the caller and ``step.node`` the node whose code is running.
No function knows which harness called it: every agent reaches them through
its REPL, ours or the one external harnesses run with ``rlmflow repl``.
Launches and messages hang off the running node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import Node, Step

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
) -> Node:
    """Start an agent launched by the caller's running node."""
    g, parent = step.graph, step.agent
    limits = step.config
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if not can_spawn(g, parent):
        raise ValueError(f"cannot start agents beyond max depth {limits.max_depth}")
    if len(task) > limits.max_query_chars:
        raise ValueError(f"task exceeds {limits.max_query_chars} characters")
    if flow.budget_exhausted(g, parent):
        raise RuntimeError("cannot start an agent after max_budget is exhausted")
    if config is not None and not isinstance(config, str):
        raise TypeError("agents started by an agent take a config name, not an AgentConfig")
    return flow.spawn(step, task, config=config or "default", name=name, **args)


def send(flow: Flow, step: Step, target: object, text: str, **inputs: Any) -> Node:
    """Attach a message to another agent, binding ``inputs`` there; it reopens a finished agent."""
    from rlmflow.agent_tools.view import resolve_agent

    if not isinstance(text, str):
        raise TypeError("message text must be a string")
    g = step.graph
    agent = resolve_agent(g, target)
    message = g.send(agent, f"From {step.agent.path()}: {text}", by=step.node, **inputs)
    flow._land(message)
    return message


def done(flow: Flow, step: Step, result: Any) -> Any:
    """Check ``result`` against the caller's ``output_schema`` and return the value to record."""
    from rlmflow.schema import as_json_value, check

    schema = step.config.output_schema
    return check(result, schema) if schema is not None else as_json_value(result)


async def repl(flow: Flow, step: Step, code: str) -> dict[str, Any]:
    """Run ``code`` as a new node after the caller's last node."""
    g, agent = step.graph, step.agent
    node = flow.append(g, agent, Node("assistant", code=code))
    run = await flow.run_node(g, agent, node)
    return {
        "output": node.output,
        "error": node.error,
        "submitted": bool(run and run.submitted),
        "result": run.answer if run and run.submitted else None,
    }


__all__ = ["done", "repl", "send", "spawn"]
