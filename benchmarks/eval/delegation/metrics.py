"""Deterministic trajectory diagnostics derived from a persisted node graph."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from rlmflow import (
    AgentStart,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    ReplDead,
    TransitionDecision,
    VerificationFailure,
)


def _timestamp(node) -> float | None:
    value = getattr(node, "created_at", None)
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
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
    root = child.root
    launch_action = (
        root._index.launch_actions_by_agent_id.get(child.id) if root is not None else None
    )
    launch_call_id = (
        launch_action.launch_call_id_for(child.id) if launch_action is not None else None
    )
    retrieval_actions = (
        [
            node
            for node in root.walk()
            if isinstance(node, ExecAction) and child.id in node.retrieved_agent_ids
        ]
        if root is not None
        else []
    )
    first_retrieval = min(retrieval_actions, key=lambda action: action.seq, default=None)
    launch_parent = launch_action.parent_agent if launch_action is not None else None
    intervening_parent_actions = (
        sum(
            isinstance(node, ExecAction)
            and node.seq > launch_action.seq
            and (first_retrieval is None or node.seq < first_retrieval.seq)
            for node in launch_parent.transcript()
        )
        if launch_action is not None and launch_parent is not None
        else 0
    )
    return {
        "agent_id": child.id,
        "parent_agent_id": parent.id if parent is not None else None,
        "name": child.config.name,
        "path": child.config.path,
        "depth": child.config.depth,
        "model": child.config.model,
        "launch_call_id": launch_call_id,
        "launch_action_id": launch_action.id if launch_action is not None else None,
        "retrieval_action_ids": [action.id for action in retrieval_actions],
        "intervening_parent_actions": intervening_parent_actions,
        "argument_keys": sorted(child.args),
        "dependency_argument_keys": sorted(
            name for name, _source_id in child.argument_references
        ),
        "goal_hash": _hash(child.content),
        "prompt_sha256": hashlib.sha256(child.content.encode()).hexdigest(),
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


def _dependency_chain_depth(agents: list[AgentStart]) -> int:
    """Longest handle-valued input chain, independent of ownership depth."""
    by_id = {agent.id: agent for agent in agents}
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(agent_id: str) -> int:
        if agent_id in memo:
            return memo[agent_id]
        if agent_id in visiting:
            return 0
        visiting.add(agent_id)
        agent = by_id.get(agent_id)
        sources = [
            source_id
            for _argument_name, source_id in (
                agent.argument_references if agent is not None else ()
            )
            if source_id in by_id
        ]
        value = 0 if not sources else 1 + max(depth(source_id) for source_id in sources)
        visiting.remove(agent_id)
        memo[agent_id] = value
        return value

    return max((depth(agent.id) for agent in agents), default=0)


def delegation_metrics(root: AgentStart) -> dict[str, Any]:
    """Return behavior metrics that require no semantic or LLM judge."""

    agents = list(root.iter_agents())
    children = agents[1:]
    launch_batches: dict[str, list[AgentStart]] = defaultdict(list)
    for child in children:
        launch_action = root._index.launch_actions_by_agent_id.get(child.id)
        launch_batches[launch_action.id if launch_action is not None else ""].append(child)

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
    query_calls = sum(
        len(re.findall(r"\bllm_query(?:_batched)?\s*\(", node.code)) for node in actions
    )
    decisions = [
        node for node in root.walk() if isinstance(node, TransitionDecision)
    ]
    verification_failures = [
        node for node in root.walk() if isinstance(node, VerificationFailure)
    ]
    spent = root.usage
    child_runs = [_child_telemetry(child) for child in children]
    routing_decisions = [_routing_decision(agent) for agent in agents]
    dependency_edges = [
        {
            "source_agent_id": source_id,
            "target_agent_id": agent.id,
            "argument_name": argument_name,
        }
        for agent in children
        for argument_name, source_id in agent.argument_references
    ]
    launch_batch_records = []
    for action_id, batch in sorted(
        launch_batches.items(),
        key=lambda item: min(
            (
                root._index.launch_actions_by_agent_id[child.id].seq
                for child in item[1]
                if child.id in root._index.launch_actions_by_agent_id
            ),
            default=0,
        ),
    ):
        action = root._index.launch_actions_by_agent_id.get(batch[0].id) if batch else None
        launch_batch_records.append(
            {
                "launch_action_id": action_id or None,
                "parent_agent_id": action.parent_agent.id if action is not None else None,
                "child_agent_ids": [child.id for child in batch],
                "call_ids": {
                    child.id: action.launch_call_id_for(child.id)
                    for child in batch
                    if action is not None
                },
                "size": len(batch),
                "ready_size": sum(not child.argument_references for child in batch),
            }
        )

    completed_children = max(len(terminals) - int(root.terminal), 0)
    retrieved_children = sum(run["result_retrieved"] for run in child_runs)
    return {
        "agents": len(agents),
        "children": len(children),
        "completed_children": completed_children,
        "unfinished_children": len(unfinished),
        "failed_children": len(failed),
        "budget_stopped_children": len(budget_stops),
        "launch_batches": len(launch_batches),
        "max_launch_batch": max((len(batch) for batch in launch_batches.values()), default=0),
        "max_ready_launch_batch": max(
            (
                sum(not child.argument_references for child in batch)
                for batch in launch_batches.values()
            ),
            default=0,
        ),
        "concurrent_launch_batches": sum(len(batch) > 1 for batch in launch_batches.values()),
        "duplicate_child_goals": duplicate_goals,
        "launch_events": len(children),
        "query_calls": query_calls,
        "retrieval_events": sum(len(action.retrieved_agent_ids) for action in actions),
        "dependency_edges": dependency_edges,
        "dependency_edge_count": len(dependency_edges),
        "transition_decisions": len(decisions),
        "matched_transition_decisions": sum(
            decision.selected_rule_id is not None for decision in decisions
        ),
        "verification_failures": len(verification_failures),
        "schema_failures": sum(
            failure.kind == "schema" for failure in verification_failures
        ),
        "verifier_failures": sum(
            failure.kind == "verifier" for failure in verification_failures
        ),
        "launch_batch_records": launch_batch_records,
        "wait_events": waits,
        "retrieved_child_results": retrieved_children,
        "retrieval_ratio": (retrieved_children / completed_children if completed_children else 0.0),
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
        "recursive_launch_count": sum(run["depth"] > 1 for run in child_runs),
        "dependency_chain_depth": _dependency_chain_depth(agents),
        "model_calls": model_calls,
        "input_tokens": spent.input_tokens,
        "output_tokens": spent.output_tokens,
        "post_root_nodes": post_root_nodes,
        "root_finished_with_unfinished_children": bool(root.terminal and unfinished),
        "routing_decisions": routing_decisions,
        "child_runs": child_runs,
    }


__all__ = ["delegation_metrics"]
