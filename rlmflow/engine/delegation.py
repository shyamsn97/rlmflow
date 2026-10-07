"""Child launch: resolve, attach, submit, and the ``launch_subagent`` factory."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from rlmflow.engine import verification
from rlmflow.graph.arguments import RECURSE, transform_argument
from rlmflow.graph.config import (
    AGENT_REF_KEY,
    Input,
    agent_ref_id,
    input_parts,
    validate_argument_name,
)
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ExecAction,
    Node,
    build_agent,
    validate_agent_name,
)
from rlmflow.runtime.repl_client import current_rpc_call_id
from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import tool

LAUNCH_DESCRIPTION = """\
Launch a recursive agent immediately. Pass `name=` for a stable child name, JSON \
keyword arguments or child handles, `schema=` for a validated result, and `verify=` \
for repair feedback. Wrap a known input as `Input(value, description="...")` to \
give the child its format or meaning without copying the value into its prompt.

Example — launch ready independent work together:
```repl
first = launch_subagent(name="first", task="Produce deliverable A.")
second = launch_subagent(name="second", task="Produce deliverable B.")
print(await asyncio.gather(first, second))
```
Example — declare a known dependency without waiting first:
```repl
source = launch_subagent(task="Produce the source artifact.")
review = launch_subagent(task="Validate artifact.", artifact=source)
print(await review)
```"""


def launch_tool(flow: Any, node: Node):
    mutation_lock = threading.Lock()

    def restore_launch(
        saved_result: object,
        *,
        task: str,
        name: str | None = None,
        schema: Any = None,
        verify: Any = None,
        **json_args: Any,
    ) -> AgentHandle:
        del task, schema, json_args
        if not isinstance(node, ExecAction):
            raise TypeError("launch_subagent requires an ExecAction")
        call_id = current_rpc_call_id()
        verifier_id, verifier_hash = _register_verifier(flow, node, call_id, verify)
        agent_id = agent_ref_id(saved_result)
        root = node.root
        existing = root.find_agent(agent_id) if root is not None and agent_id is not None else None
        if existing is None:
            raise ValueError(
                f"saved launch call {node.id}:{call_id} points to missing agent "
                f"{agent_id!r}"
            )
        if existing.config.verifier_id != verifier_id:
            raise ValueError(f"restored launch verifier changed for {existing.config.path!r}")
        if existing.config.verifier_hash != verifier_hash:
            raise ValueError(f"restored launch verifier changed for {existing.config.path!r}")
        if name is not None and name != existing.config.name:
            raise ValueError(
                f"restored launch name changed for {existing.config.path!r}: {name!r}"
            )
        return _handle(existing)

    @tool(
        LAUNCH_DESCRIPTION,
        proxy=True,
        is_async=False,
        restore=restore_launch,
    )
    def launch_subagent(
        *,
        task: str,
        name: str | None = None,
        schema: Any = None,
        verify: Any = None,
        **json_args: Any,
    ) -> AgentHandle:
        if not isinstance(node, ExecAction):
            raise TypeError("launch_subagent requires an ExecAction")
        call_id = current_rpc_call_id()

        def attach() -> AgentHandle:
            with mutation_lock:
                return _attach(
                    flow,
                    node,
                    call_id,
                    task=task,
                    name=name,
                    schema=schema,
                    verify=verify,
                    json_args=json_args,
                )

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        owner_loop = getattr(flow, "_active_loop", None)
        if current_loop is owner_loop or flow.queue is None:
            return attach()
        if owner_loop is None:
            raise RuntimeError("launch_subagent has no active Flow event loop")

        async def attach_on_owner() -> AgentHandle:
            return attach()

        return asyncio.run_coroutine_threadsafe(attach_on_owner(), owner_loop).result()

    return launch_subagent


def _attach(
    flow: Any,
    node: ExecAction,
    call_id: int,
    *,
    task: object,
    name: object,
    schema: Any,
    verify: Any,
    json_args: dict[str, Any],
) -> AgentHandle:
    verifier_id, verifier_hash = _register_verifier(
        flow,
        node,
        call_id,
        verify,
    )

    if not isinstance(task, str):
        raise TypeError("launch_subagent task must be a string")
    if name is not None:
        validate_agent_name(name)
    spec = {
        "task": task,
        "name": name,
        "args": resolve_args(node, json_args),
        "output_schema": (
            flow.output_codec.normalize(schema) if schema is not None else None
        ),
        "verifier_id": verifier_id,
        "verifier_hash": verifier_hash,
    }
    resolved = flow.resolve_child(node, spec, call_id)
    node.record_tool_call("launch_subagent", call_id, {AGENT_REF_KEY: resolved.id})
    _sort_launched_children(node)
    submit_child(flow, resolved)
    return _handle(resolved)


def _register_verifier(
    flow: Any,
    action: ExecAction,
    call_id: int,
    verify: Any,
) -> tuple[str | None, str | None]:
    if verify is None:
        return None, None
    verifier_id = f"{action.id}:{call_id}"
    if callable(verify):
        flow._verifiers[verifier_id] = verify
        return verifier_id, verification.fingerprint(verify)
    if (
        isinstance(verify, dict)
        and verify.get("$rlmflow_verifier") == verifier_id
        and isinstance(verify.get("fingerprint"), str)
    ):
        return verifier_id, verify["fingerprint"]
    raise TypeError("launch_subagent verify must be a callable")


def _handle(agent: AgentStart) -> AgentHandle:
    return AgentHandle(
        agent_id=agent.id,
        name=agent.config.name,
        path=agent.config.path,
    )


def resolve_args(
    action: ExecAction,
    args: object,
) -> dict[str, Any]:
    """Validate strict JSON arguments and persist nested handle references."""
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise TypeError("launch_subagent arguments must be a mapping")
    root = action.root
    owner = action.parent_agent
    if root is None or owner is None:
        raise RuntimeError("cannot resolve arguments from a detached action")

    resolved: dict[str, Any] = {}
    sources: set[str] = set()

    def resolve(path: str, value: Any) -> Any:
        wrapped = input_parts(value)
        if wrapped is not None:
            raw, description = wrapped
            return Input(
                transform_argument(raw, resolve, path=path),
                description=description,
            )
        agent_id = value.agent_id if isinstance(value, AgentHandle) else agent_ref_id(value)
        if agent_id is not None:
            source = root.find_agent(agent_id)
            if source is None:
                raise ValueError(f"argument {path!r} refers to unknown agent {agent_id!r}")
            launch_action = root._index.launch_actions_by_agent_id.get(source.id)
            launch_owner = launch_action.parent_agent if launch_action is not None else None
            if launch_owner is not owner:
                raise ValueError(
                    f"argument {path!r} agent {source.config.path!r} is outside "
                    "the current agent's delegation scope"
                )
            if source.id in sources:
                raise ValueError(f"duplicate agent argument {source.config.path!r}")
            sources.add(source.id)
            return {AGENT_REF_KEY: source.id}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            json.dumps(value, allow_nan=False)
            return value
        if isinstance(value, dict):
            if AGENT_REF_KEY in value:
                raise TypeError(f"argument {path!r} must contain an exact agent reference")
            return RECURSE
        if isinstance(value, list):
            return RECURSE
        raise TypeError(f"argument {path!r} must be JSON-compatible or an AgentHandle")

    for argument_name, value in args.items():
        validate_argument_name(argument_name)
        resolved[argument_name] = transform_argument(value, resolve, path=argument_name)
    return resolved


def resolve_child(
    flow: Any,
    action: ExecAction,
    spec: dict[str, Any],
    call_id: int,
) -> AgentStart:
    """Create one new direct child after receipt resolution has missed."""
    parent = action.parent_agent
    task = spec.get("task", "")
    requested_name = spec.get("name")
    used = {child.config.name for child in parent.sub_agents}
    if requested_name is None:
        index = call_id
        while f"child{index}" in used:
            index += 1
        name = f"child{index}"
    else:
        name = requested_name
    validate_agent_name(name)
    if any(child.config.name == name for child in parent.sub_agents):
        raise ValueError(f"duplicate child name {name!r}")

    if parent.config.depth >= parent.config.max_depth:
        raise ValueError(f"cannot launch beyond max depth {parent.config.max_depth}")
    if len(task) > parent.config.max_query_chars:
        raise ValueError(f"subagent goal exceeds {parent.config.max_query_chars} characters")
    model = flow.delegation_model or parent.config.model
    if model not in flow._llm_clients:
        available = ", ".join(sorted(flow._llm_clients))
        raise ValueError(f"unknown model {model!r}; available models: {available}")
    return new_child(flow, action, name, spec)


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
    if leaf is owner and owner.argument_references:
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
        and any(source_id == source.id for _name, source_id in agent.argument_references)
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
    overrides = {
        key: value
        for key, value in {
            "model": flow.delegation_model or node.parent_agent.config.model,
            "prompt_profile": (
                flow.delegation_prompt_profile
                or node.parent_agent.config.prompt_profile
            ),
            "reuse_worker": flow.delegation_reuse_worker,
            "output_schema": spec.get("output_schema"),
            "verifier_id": spec.get("verifier_id"),
            "verifier_hash": spec.get("verifier_hash"),
        }.items()
        if value is not None
    }
    child = build_agent(
        spec.get("task", ""),
        spec.get("args"),
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
    "resolve_args",
    "resolve_child",
    "submit_child",
    "submit_leaf",
]
