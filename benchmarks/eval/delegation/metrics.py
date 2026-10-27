"""Deterministic trajectory diagnostics derived from a run graph."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from rlmflow import CODE, DEPENDS_ON, DONE, FROM, LLM_OUTPUT, Graph, Node
from rlmflow.engine.handoff import AWAITED_KEY
from rlmflow.flow import launched_from


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


def _result_text(g: Graph, agent: Node) -> str:
    result = g.result(agent)
    return "" if result is None else str(result)


def _hash(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode()).hexdigest()


def _launch_action(g: Graph, child: Node) -> Node | None:
    edges = g.edges(kind=FROM, source=child.id)
    return g.node(edges[0].target) if edges else None


def _launch_call_id(g: Graph, action: Node | None, child: Node) -> int | None:
    if action is None:
        return None
    siblings = [agent.id for agent in launched_from(g, action)]
    return siblings.index(child.id) if child.id in siblings else None


def _cells(g: Graph, agent: Node) -> list[Node]:
    """The agent's nodes that ran code, in order."""
    return [
        node
        for node in g.nodes(agent=agent.id)
        if node.type in (CODE, DONE) and node.data.get("code")
    ]


def _references(g: Graph, agent: Node) -> list[tuple[str, str]]:
    """``(argument name, producer id)`` for each named ``depends_on`` input."""
    return [
        (edge.data["as"], edge.target)
        for edge in g.edges(kind=DEPENDS_ON, source=agent.id)
        if edge.data.get("as")
    ]


def _retrieved(g: Graph, agent: Node) -> set[str]:
    return {agent_id for node in _cells(g, agent) for agent_id in _waited(node)}


def _waited(cell: Node) -> set[str]:
    return set(cell.data.get(AWAITED_KEY, ()))


def _child_telemetry(g: Graph, child: Node, cells: list[Node]) -> dict[str, Any]:
    terminal = g.terminal(child)
    launched_at = _timestamp(child)
    completed_at = _timestamp(g.frontier(child)) if terminal else None
    result = _result_text(g, child)
    parent = g.launcher(child)
    retrieved = parent is not None and child.id in _retrieved(g, parent)
    echoed = False
    if result and parent is not None:
        echoed = any(
            (completed_at is None or (_timestamp(node) or 0) >= completed_at)
            and result in node.data.get("output", "")
            for node in _cells(g, parent)
        )
    launch_action = _launch_action(g, child)
    retrieval_actions = [node for node in cells if child.id in _waited(node)]
    first_retrieval = min(retrieval_actions, key=lambda action: action.seq, default=None)
    intervening_parent_actions = (
        sum(
            node.seq > launch_action.seq
            and (first_retrieval is None or node.seq < first_retrieval.seq)
            for node in _cells(g, parent)
        )
        if launch_action is not None and parent is not None
        else 0
    )
    return {
        "agent_id": child.id,
        "parent_agent_id": parent.id if parent is not None else None,
        "name": child.config.name,
        "path": g.path(child),
        "depth": g.depth(child),
        "model": child.config.model,
        "launch_call_id": _launch_call_id(g, launch_action, child),
        "launch_action_id": launch_action.id if launch_action is not None else None,
        "retrieval_action_ids": [action.id for action in retrieval_actions],
        "intervening_parent_actions": intervening_parent_actions,
        "argument_keys": sorted(child.args),
        "dependency_argument_keys": sorted(name for name, _source in _references(g, child)),
        "goal_hash": _hash(child.task),
        "prompt_sha256": hashlib.sha256(child.task.encode()).hexdigest(),
        "result_hash": _hash(result) if result else None,
        "launched_at": launched_at,
        "completed_at": completed_at,
        "active_seconds": (
            max(completed_at - launched_at, 0.0)
            if launched_at is not None and completed_at is not None
            else None
        ),
        "terminal": terminal,
        "failed": result.startswith(("[child failed:", "[run failed:")),
        "result_retrieved": retrieved,
        "result_echoed": echoed,
    }


def _routing_decision(g: Graph, agent: Node) -> dict[str, Any]:
    """Return the routing outcome directly represented by one agent's graph."""
    children = g.children(agent)
    launch_actions = [action for child in children if (action := _launch_action(g, child))]
    first_launch = min(launch_actions, key=lambda action: action.seq, default=None)
    retrieved = _retrieved(g, agent)
    depth = g.depth(agent)
    return {
        "agent_id": agent.id,
        "path": g.path(agent),
        "depth": depth,
        "can_delegate": depth < agent.config.max_depth,
        "outcome": "delegated" if children else "local",
        "children_launched": len(children),
        "first_launch_turn": first_launch.seq if first_launch is not None else None,
        "prelaunch_llm_turns": (
            sum(node.seq < first_launch.seq for node in g.nodes(LLM_OUTPUT, agent=agent.id))
            if first_launch is not None
            else None
        ),
        "retrieved_child_results": sum(child.id in retrieved for child in children),
        "unretrieved_child_results": sum(
            g.terminal(child) and child.id not in retrieved for child in children
        ),
    }


def _dependency_chain_depth(g: Graph, agents: list[Node]) -> int:
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
        sources = [source for _name, source in _references(g, by_id[agent_id]) if source in by_id]
        value = 0 if not sources else 1 + max(depth(source) for source in sources)
        visiting.remove(agent_id)
        memo[agent_id] = value
        return value

    return max((depth(agent.id) for agent in agents), default=0)


def delegation_metrics(g: Graph, root: Node) -> dict[str, Any]:
    """Return behavior metrics for ``root``'s tree that require no judge."""

    agents = g.tree(root)
    children = agents[1:]
    nodes = [node for agent in agents for node in g.nodes(agent=agent.id)]
    nodes.sort(key=lambda node: node.seq)
    cells = sorted((node for agent in agents for node in _cells(g, agent)), key=lambda n: n.seq)
    launch_batches: dict[str, list[Node]] = defaultdict(list)
    launch_actions: dict[str, Node] = {}
    for child in children:
        action = _launch_action(g, child)
        if action is not None:
            launch_actions[action.id] = action
        launch_batches[action.id if action is not None else ""].append(child)

    terminal = {agent.id: g.terminal(agent) for agent in agents}
    unfinished = [agent for agent in children if not terminal[agent.id]]
    failed = [
        agent
        for agent in children
        if any(node.data.get("dead") for node in _cells(g, agent))
        or isinstance(g.frontier(agent).data.get("error"), str)
        or _result_text(g, agent).startswith(("[child failed:", "[run failed:"))
    ]
    budget_stops = [
        agent
        for agent in children
        if _result_text(g, agent) in {"[budget exceeded]", "[max_iters exceeded]"}
    ]

    root_finished_at = _timestamp(g.frontier(root)) if terminal[root.id] else None
    post_root_nodes = 0
    if root_finished_at is not None:
        post_root_nodes = sum(
            timestamp is not None and timestamp > root_finished_at
            for node in nodes
            if (timestamp := _timestamp(node)) is not None
        )

    goals = Counter(" ".join(child.task.lower().split()) for child in children)
    duplicate_goals = sum(count - 1 for count in goals.values() if count > 1)
    model_calls = sum(node.type == LLM_OUTPUT for node in nodes)
    waits = sum(bool(_waited(node)) for node in cells)
    query_calls = sum(
        len(re.findall(r"\bllm_query(?:_batched)?\s*\(", node.data["code"])) for node in cells
    )
    spent = g.usage(agents)
    child_runs = [_child_telemetry(g, child, cells) for child in children]
    routing_decisions = [_routing_decision(g, agent) for agent in agents]
    dependency_edges = [
        {
            "source_agent_id": source_id,
            "target_agent_id": agent.id,
            "argument_name": argument_name,
        }
        for agent in children
        for argument_name, source_id in _references(g, agent)
    ]
    launch_batch_records = []
    for action_id, batch in sorted(
        launch_batches.items(),
        key=lambda item: launch_actions[item[0]].seq if item[0] in launch_actions else 0,
    ):
        action = launch_actions.get(action_id)
        launch_batch_records.append(
            {
                "launch_action_id": action_id or None,
                "parent_agent_id": action.agent_id if action is not None else None,
                "child_agent_ids": [child.id for child in batch],
                "call_ids": {child.id: _launch_call_id(g, action, child) for child in batch}
                if action is not None
                else {},
                "size": len(batch),
                "ready_size": sum(not _references(g, child) for child in batch),
            }
        )

    completed_children = max(sum(terminal.values()) - int(terminal[root.id]), 0)
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
                sum(not _references(g, child) for child in batch)
                for batch in launch_batches.values()
            ),
            default=0,
        ),
        "concurrent_launch_batches": sum(len(batch) > 1 for batch in launch_batches.values()),
        "duplicate_child_goals": duplicate_goals,
        "launch_events": len(children),
        "query_calls": query_calls,
        "retrieval_events": sum(len(_waited(cell)) for cell in cells),
        "dependency_edges": dependency_edges,
        "dependency_edge_count": len(dependency_edges),
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
        "max_depth": max((g.depth(agent) for agent in agents), default=0),
        "recursive_launch_count": sum(run["depth"] > 1 for run in child_runs),
        "dependency_chain_depth": _dependency_chain_depth(g, agents),
        "model_calls": model_calls,
        "input_tokens": spent.input_tokens,
        "output_tokens": spent.output_tokens,
        "post_root_nodes": post_root_nodes,
        "root_finished_with_unfinished_children": bool(terminal[root.id] and unfinished),
        "routing_decisions": routing_decisions,
        "child_runs": child_runs,
    }


__all__ = ["delegation_metrics"]
