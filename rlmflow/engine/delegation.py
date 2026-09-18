"""Child launch: resolve, attach, submit, and the ``launch_subagent`` factory."""

from __future__ import annotations

import asyncio
from typing import Any

from rlmflow.graph.config import AGENT_REF_KEY, agent_ref_id, input_references
from rlmflow.graph.nodes import (
    DEFAULT_QUERY,
    AgentStart,
    DoneOutput,
    ExecAction,
    Node,
    validate_agent_name,
)
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.structured import json_schema_for
from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import tool


def launch_tool(flow: Any, node: Node):
    mutation_lock = asyncio.Lock()

    @tool(
        "Spawn or resume one focused child agent with an explicitly chosen model. "
        "Pass a child handle as an inputs value to wait for that child and inject "
        "its result into the new agent's INPUTS.",
        proxy=True,
    )
    async def launch_subagent(
        goal: str,
        *,
        model: str,
        name: str | None = None,
        inputs: dict[str, str | AgentHandle] | None = None,
        output_schema: Any = None,
        prompt_profile: str | None = None,
        reuse_repl: bool = False,
    ) -> AgentHandle:
        if not isinstance(node, ExecAction):
            raise TypeError("launch_subagent requires an ExecAction")
        call_id = current_rpc_call_id()
        async with mutation_lock:
            existing_id = node.launched_agent_ids.get(call_id)
            if existing_id is not None:
                root = node.root
                existing = root.find_agent(existing_id) if root is not None else None
                if existing is None:
                    raise ValueError(
                        f"launch receipt {node.id}:{call_id} points to missing "
                        f"agent {existing_id!r}"
                    )
                return _handle(existing)
            if flow.queue is None:
                raise ValueError(f"replayed launch call {node.id}:{call_id} has no launch receipt")

            if not isinstance(goal, str):
                raise TypeError("launch_subagent goal must be a string")
            spec = {
                "query": goal,
                "name": name,
                "inputs": resolve_inputs(node, inputs),
                "model": model,
                "output_schema": output_schema,
                "prompt_profile": prompt_profile,
                "reuse_repl": reuse_repl,
            }
            resolved = flow.resolve_child(node, spec, call_id)
            node.record_agent_launch(call_id, resolved)
            _sort_launched_children(node)
            flow.submit_child(resolved)
        return _handle(resolved)

    return launch_subagent


def _handle(agent: AgentStart) -> AgentHandle:
    return AgentHandle(
        agent_id=agent.id,
        name=agent.config.name,
        path=agent.config.path,
    )


def resolve_inputs(
    action: ExecAction,
    inputs: object,
) -> dict[str, str | dict[str, str]]:
    """Validate immediate strings and persist top-level handle-valued inputs."""
    if inputs is None:
        return {}
    if not isinstance(inputs, dict):
        raise TypeError("launch_subagent inputs must be a mapping")
    root = action.root
    owner = action.parent_agent
    if root is None or owner is None:
        raise RuntimeError("cannot resolve inputs from a detached action")

    resolved: dict[str, str | dict[str, str]] = {}
    sources: set[str] = set()
    for input_name, value in inputs.items():
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("input names must be non-empty strings")
        if isinstance(value, str):
            resolved[input_name] = value
            continue
        agent_id = value.agent_id if isinstance(value, AgentHandle) else agent_ref_id(value)
        if agent_id is None:
            raise TypeError(f"input {input_name!r} must be a string or top-level AgentHandle")

        source = root.find_agent(agent_id)
        if source is None:
            raise ValueError(f"input {input_name!r} refers to unknown agent {agent_id!r}")
        launch_action = root._index.launch_actions_by_agent_id.get(source.id)
        launch_owner = launch_action.parent_agent if launch_action is not None else None
        if launch_owner is not owner:
            raise ValueError(
                f"input {input_name!r} agent {source.config.path!r} is outside "
                "the current agent's delegation scope"
            )
        if source.id in sources:
            raise ValueError(f"duplicate agent input {source.config.path!r}")
        sources.add(source.id)
        resolved[input_name] = {AGENT_REF_KEY: source.id}
    return resolved


def resolve_child(
    flow: Any,
    action: ExecAction,
    spec: dict[str, Any],
    call_id: int,
) -> AgentStart:
    """Create one new direct child after receipt resolution has missed."""
    explicit_name = spec.get("name")
    if explicit_name is not None:
        root = action.root
        for agent_id in action.launched_agent_ids.values():
            existing = root.find_agent(agent_id) if root is not None else None
            if existing is not None and existing.config.name == explicit_name:
                references = input_references(spec.get("inputs") or {})
                if existing.input_references != references:
                    raise ValueError(
                        f"launch {existing.config.name!r} was repeated with "
                        "different handle-valued inputs"
                    )
                return existing

    parent = action.parent_agent
    query = spec.get("query", "")
    if explicit_name is None:
        used = {child.config.name for child in parent.sub_agents}
        index = call_id
        while f"child{index}" in used:
            index += 1
        name = f"child{index}"
    else:
        name = explicit_name
    validate_agent_name(name)
    if any(child.config.name == name for child in parent.sub_agents):
        raise ValueError(f"duplicate child name {name!r}")

    if parent.config.depth >= parent.config.max_depth:
        raise ValueError(f"cannot launch beyond max depth {parent.config.max_depth}")
    if len(query) > parent.config.max_query_chars:
        raise ValueError(f"subagent goal exceeds {parent.config.max_query_chars} characters")
    model = spec["model"]
    if model not in flow._llm_clients:
        available = ", ".join(sorted(flow._llm_clients))
        raise ValueError(f"unknown model {model!r}; available models: {available}")
    return flow.new_child(action, name, spec)


def submit_child(flow: Any, child: AgentStart) -> None:
    """Submit a newly attached child and all unfinished restored leaves."""
    queue = flow.queue
    if queue is None:
        raise RuntimeError("launch_subagent requires an active stream")
    if child.terminal:
        return
    for leaf in child.leaves():
        submit_leaf(flow, leaf, publish=isinstance(leaf, AgentStart))


def submit_leaf(flow: Any, leaf: Node, *, publish: bool = False) -> bool:
    """Submit a leaf unless its owning agent is still dependency-blocked."""
    queue = flow.queue
    if queue is None:
        raise RuntimeError("agent submission requires an active stream")
    owner = leaf.parent_agent
    if owner is None or owner.terminal:
        return False
    if leaf is owner and owner.input_references:
        if owner.dependency_scheduled or owner.blocked:
            return False
        owner.dependency_scheduled = True
    queue.submit(leaf, flow._drive, publish=publish)
    return True


def ready_dependents(node: Node) -> list[AgentStart]:
    """Return newly runnable agents when one upstream agent completes."""
    if not isinstance(node, DoneOutput):
        return []
    source = node.parent_agent
    root = node.root
    if source is None or root is None:
        return []
    return [
        agent
        for agent in root.iter_agents()
        if agent.frontier is agent
        and not agent.dependency_scheduled
        and any(source_id == source.id for _name, source_id in agent.input_references)
        and not agent.blocked
    ]


def new_child(
    flow: Any,
    node: Node,
    name: str,
    spec: dict[str, Any],
) -> AgentStart:
    """Open a child agent of ``node``'s agent, attached to ``node``."""
    if flow.budget_exhausted(node):
        raise RuntimeError("cannot launch a new subagent after max_budget is exhausted")
    schema = spec.get("output_schema")
    overrides = {
        key: value
        for key, value in {
            "inputs": dict(spec.get("inputs") or {}),
            "model": spec.get("model"),
            "prompt_profile": spec.get("prompt_profile"),
            "output_schema": (json_schema_for(schema) if schema is not None else None),
            "reuse_repl": spec.get("reuse_repl"),
        }.items()
        if value is not None
    }
    child = AgentStart(
        content=spec.get("query") or DEFAULT_QUERY,
        config=node.parent_agent.config.child(name, **overrides),
    )
    node.append(child)
    return child


def _sort_launched_children(action: ExecAction) -> None:
    """Keep concurrent proxy launches in source-call order using receipts."""
    root = action.root
    parent = action.parent_agent
    if root is None or parent is None:
        return

    def launch_order(agent: AgentStart) -> int:
        launch_action = root._index.launch_actions_by_agent_id.get(agent.id)
        if launch_action is None:
            return 0
        call_id = launch_action.launch_call_id_for(agent.id)
        return call_id if call_id is not None else 0

    action.children.sort(
        key=lambda value: (
            not isinstance(value, AgentStart),
            launch_order(value) if isinstance(value, AgentStart) else 0,
        )
    )
    parent.sub_agents.sort(
        key=lambda value: (
            value.parent.seq if value.parent is not None else 0,
            launch_order(value),
        )
    )


__all__ = [
    "launch_tool",
    "new_child",
    "ready_dependents",
    "resolve_inputs",
    "resolve_child",
    "submit_leaf",
    "submit_child",
]
