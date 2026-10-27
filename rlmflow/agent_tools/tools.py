"""Delegation and REPL access as plain functions of ``(flow, step)``.

``step.agent`` is the caller and ``step.node`` its latest node. No function
knows which harness called it: our REPL binds them as Python functions, and the
agent-tools server exposes them to external harnesses over HTTP.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import HARNESSES, AgentConfig, can_spawn
from rlmflow.graph.nodes import AGENT_START, Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow

#: ``rlmflow wait`` returns after at most this many seconds.
WAIT_LIMIT = 60.0
STILL_RUNNING = "Still running. End your turn; you'll be resumed when they finish."


def child_config(flow: Flow, step: Step, config: str | None) -> AgentConfig:
    """A config name on the graph, or the caller's own config when ``None``."""
    if config is not None and not isinstance(config, str):
        raise TypeError("agents started by an agent take a config name, not an AgentConfig")
    resolved = step.agent.config if config is None else step.graph.config(config)
    if resolved.harness == HARNESSES["repl"] and resolved.model not in flow.llms:
        available = ", ".join(sorted(flow.llms))
        raise ValueError(f"config {resolved.name!r} uses unknown model {resolved.model!r}; available: {available}")
    return resolved


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
    resolved = child_config(flow, step, config)
    return flow.spawn(step, task, config=resolved, name=name, inputs=inputs, **args)


def resolve(g: Any, selector: object) -> Node:
    """An agent by node, id, path, or unique name."""
    from rlmflow.agent_tools.view import resolve_agent

    if isinstance(selector, Node) and selector.type == AGENT_START and selector.id in g:
        return selector
    return resolve_agent(g, selector)


async def wait(
    flow: Flow,
    step: Step,
    agent_ids: list[str],
    *,
    timeout: float = WAIT_LIMIT,
) -> dict[str, Any]:
    """Each agent's result, or ``"running"`` once ``timeout`` (at most 60 s) passes."""
    g = step.graph
    targets = [resolve(g, agent_id) for agent_id in agent_ids] or [
        child for child in g.launched(step.agent) if not g.terminal(child)
    ]
    timeout = min(max(timeout, 0.0), WAIT_LIMIT)
    if flow.queue is not None and targets:
        try:
            await asyncio.wait_for(
                flow.queue.join(lambda: all(g.terminal(t) for t in targets)), timeout
            )
        except TimeoutError:
            pass
    return {
        target.id: g.result(target) if g.terminal(target) else "running"
        for target in targets
    }


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


def agents(flow: Flow, step: Step) -> str:
    """The caller's run as a tree with each agent's config, status and result preview."""
    from rlmflow.agent_tools.view import build_agent_directory

    return build_agent_directory(step.graph, step.agent).render_graph(show_results=True)


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
    "STILL_RUNNING",
    "WAIT_LIMIT",
    "agents",
    "child_config",
    "done",
    "repl",
    "resolve",
    "send",
    "spawn",
    "wait",
]
