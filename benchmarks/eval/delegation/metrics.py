"""Deterministic trajectory diagnostics derived from a persisted node graph."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from rlmflow import AgentStart, ErrorOutput, ExecAction, ExecOutput, LLMOutput, ReplDead


def _timestamp(node) -> float | None:
    value = getattr(node, "created_at", None)
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _result_text(agent: AgentStart) -> str:
    result = agent.result()
    return "" if result is None else str(result)


def _hash(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode()).hexdigest()


def _child_telemetry(child: AgentStart) -> dict[str, Any]:
    launched_at = _timestamp(child)
    completed_at = _timestamp(child.frontier) if child.terminal else None
    result = _result_text(child)
    parent = child.parent.parent_agent if child.parent is not None else None
    retrieved = parent is not None and child.id in parent.retrieved_agent_ids()
    echoed = False
    if result and parent is not None:
        echoed = any(
            isinstance(node, ExecOutput)
            and (completed_at is None or (_timestamp(node) or 0) >= completed_at)
            and result in node.content
            for node in parent.transcript()
        )
    return {
        "agent_id": child.id,
        "parent_agent_id": parent.id if parent is not None else None,
        "path": child.config.path,
        "depth": child.config.depth,
        "model": child.config.model,
        "launch_call_id": child.config.launch_call_id,
        "goal_hash": _hash(child.content),
        "result_hash": _hash(result) if result else None,
        "launched_at": launched_at,
        "completed_at": completed_at,
        "active_seconds": (
            max(completed_at - launched_at, 0.0)
            if launched_at is not None and completed_at is not None
            else None
        ),
        "terminal": child.terminal,
        "failed": result.startswith(("[child failed:", "[run failed:")),
        "result_retrieved": retrieved,
        "result_echoed": echoed,
    }


def _routing_decision(agent: AgentStart) -> dict[str, Any]:
    """Return the routing outcome directly represented by one agent's graph."""
    children = list(agent.sub_agents)
    launch_actions = [child.parent for child in children if child.parent is not None]
    first_launch = min(launch_actions, key=lambda action: action.seq, default=None)
    retrieved = agent.retrieved_agent_ids()
    return {
        "agent_id": agent.id,
        "path": agent.config.path,
        "depth": agent.config.depth,
        "can_delegate": agent.config.depth < agent.config.max_depth,
        "outcome": "delegated" if children else "local",
        "children_launched": len(children),
        "first_launch_turn": first_launch.seq if first_launch is not None else None,
        "prelaunch_llm_turns": (
            sum(
                isinstance(node, LLMOutput) and node.seq < first_launch.seq
                for node in agent.transcript()
            )
            if first_launch is not None
            else None
        ),
        "retrieved_child_results": sum(child.id in retrieved for child in children),
        "unretrieved_child_results": sum(
            child.terminal and child.id not in retrieved for child in children
        ),
    }


def delegation_metrics(root: AgentStart) -> dict[str, Any]:
    """Return behavior metrics that require no semantic or LLM judge."""

    agents = list(root.iter_agents())
    children = agents[1:]
    launch_batches: dict[str, list[AgentStart]] = defaultdict(list)
    for child in children:
        parent_id = child.parent.id if child.parent is not None else ""
        launch_batches[parent_id].append(child)

    terminals = [agent for agent in agents if agent.terminal]
    unfinished = [agent for agent in children if not agent.terminal]
    failed = [
        agent
        for agent in children
        if any(isinstance(node, (ErrorOutput, ReplDead)) for node in agent.transcript())
        or _result_text(agent).startswith(("[child failed:", "[run failed:"))
    ]
    budget_stops = [
        agent
        for agent in children
        if _result_text(agent) in {"[budget exceeded]", "[max_iters exceeded]"}
    ]

    root_finished_at = _timestamp(root.frontier) if root.terminal else None
    post_root_nodes = 0
    if root_finished_at is not None:
        post_root_nodes = sum(
            timestamp is not None and timestamp > root_finished_at
            for node in root.walk()
            if (timestamp := _timestamp(node)) is not None
        )

    goals = Counter(" ".join(child.content.lower().split()) for child in children)
    duplicate_goals = sum(count - 1 for count in goals.values() if count > 1)
    model_calls = sum(isinstance(node, LLMOutput) for node in root.walk())
    actions = [node for node in root.walk() if isinstance(node, ExecAction)]
    waits = sum("wait_for_result" in node.code for node in actions)
    spent = root.usage
    child_runs = [_child_telemetry(child) for child in children]
    routing_decisions = [_routing_decision(agent) for agent in agents]

    return {
        "agents": len(agents),
        "children": len(children),
        "completed_children": max(len(terminals) - int(root.terminal), 0),
        "unfinished_children": len(unfinished),
        "failed_children": len(failed),
        "budget_stopped_children": len(budget_stops),
        "launch_batches": len(launch_batches),
        "max_launch_batch": max((len(batch) for batch in launch_batches.values()), default=0),
        "concurrent_launch_batches": sum(len(batch) > 1 for batch in launch_batches.values()),
        "duplicate_child_goals": duplicate_goals,
        "launch_events": len(children),
        "wait_events": waits,
        "retrieved_child_results": sum(run["result_retrieved"] for run in child_runs),
        "unretrieved_child_results": sum(
            run["terminal"] and not run["result_retrieved"] for run in child_runs
        ),
        "echoed_child_results": sum(run["result_echoed"] for run in child_runs),
        "local_routing_decisions": sum(
            decision["outcome"] == "local" for decision in routing_decisions
        ),
        "delegated_routing_decisions": sum(
            decision["outcome"] == "delegated" for decision in routing_decisions
        ),
        "max_depth": max((agent.config.depth for agent in agents), default=0),
        "model_calls": model_calls,
        "input_tokens": spent.input_tokens,
        "output_tokens": spent.output_tokens,
        "post_root_nodes": post_root_nodes,
        "root_finished_with_unfinished_children": bool(root.terminal and unfinished),
        "routing_decisions": routing_decisions,
        "child_runs": child_runs,
    }


__all__ = ["delegation_metrics"]
