"""Child launch: the ``launch_agent`` tool over ``ops.spawn``."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from rlmflow.graph import ops, reads
from rlmflow.graph.config import (
    NATIVE_BACKEND,
    RECURSE,
    Input,
    input_parts,
    validate_argument_name,
    visit_argument,
)
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    AGENT_REF_KEY,
    ActionNode,
    AgentStart,
    ExecAction,
    Node,
    Step,
    agent_ref_id,
    current_step,
    validate_agent_name,
)
from rlmflow.runtime.repl_client import current_rpc_call_id, current_rpc_name
from rlmflow.runtime.repl_server import fingerprint
from rlmflow.tools.agents_view import AgentHandle, agent_handle, resolve_agent
from rlmflow.tools.namespace import tool

LAUNCH_DESCRIPTION = """\
Launch a recursive agent immediately. Pass `name=` for a stable child name, JSON \
keyword arguments or agent handles, `schema=` for a validated result, and `verify=` \
for repair feedback. Wrap a known input as `Input(value, description="...")` to \
give the child its format or meaning without copying the value into its prompt.

Example — launch ready independent work together:
```repl
first = launch_agent(name="first", task="Produce deliverable A.")
second = launch_agent(name="second", task="Produce deliverable B.")
print(await asyncio.gather(first, second))
```
Example — declare a known dependency without waiting first:
```repl
source = launch_agent(task="Produce the source artifact.")
review = launch_agent(task="Validate artifact.", artifact=source)
print(await review)
```"""


def launch_tool(flow: Any, node: Node):
    step = current_step()
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
            raise TypeError("launch_agent requires an ExecAction")
        call_id = current_rpc_call_id()
        verifier_id, verifier_hash = _register_verifier(flow, node, call_id, verify)
        agent_id = agent_ref_id(saved_result)
        existing = step.graph.node(agent_id) if agent_id in step.graph else None
        if not isinstance(existing, AgentStart):
            raise ValueError(
                f"saved launch call {node.id}:{call_id} points to missing agent {agent_id!r}"
            )
        label = existing.config.name
        if existing.config.verifier_id != verifier_id:
            raise ValueError(f"restored launch verifier changed for {label!r}")
        if existing.config.verifier_hash != verifier_hash:
            raise ValueError(f"restored launch verifier changed for {label!r}")
        if name is not None and name != existing.config.name:
            raise ValueError(f"restored launch name changed for {label!r}: {name!r}")
        return agent_handle(step.graph, existing)

    @tool(
        LAUNCH_DESCRIPTION,
        proxy=True,
        is_async=False,
        restore=restore_launch,
    )
    def launch_agent(
        *,
        task: str,
        name: str | None = None,
        schema: Any = None,
        verify: Any = None,
        **json_args: Any,
    ) -> AgentHandle:
        if not isinstance(node, ExecAction):
            raise TypeError("launch_agent requires an ExecAction")
        call_id = current_rpc_call_id()

        def launch() -> AgentHandle:
            with mutation_lock:
                return _launch(
                    flow,
                    step,
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
            return launch()
        if owner_loop is None:
            raise RuntimeError("launch_agent has no active Flow event loop")

        async def launch_on_owner() -> AgentHandle:
            return launch()

        return asyncio.run_coroutine_threadsafe(launch_on_owner(), owner_loop).result()

    return launch_agent


def _launch(
    flow: Any,
    step: Step,
    call_id: int,
    *,
    task: object,
    name: object,
    schema: Any,
    verify: Any,
    json_args: dict[str, Any],
) -> AgentHandle:
    action = step.node
    verifier_id, verifier_hash = _register_verifier(flow, action, call_id, verify)
    if not isinstance(task, str):
        raise TypeError("launch_agent task must be a string")
    if name is not None:
        validate_agent_name(name)
    args, inputs = split_args(step, json_args)
    spec = {
        "task": task,
        "name": name,
        "args": args,
        "inputs": inputs,
        "output_schema": (flow.output_codec.normalize(schema) if schema is not None else None),
        "verifier_id": verifier_id,
        "verifier_hash": verifier_hash,
    }
    child = flow.resolve_child(action, spec, call_id)
    if flow.queue is not None:
        flow.queue.wake()
    return agent_handle(step.graph, child)


def _register_verifier(
    flow: Any,
    action: ExecAction,
    call_id: int,
    verify: Any,
) -> tuple[str | None, str | None]:
    if verify is None:
        return None, None
    if callable(verify):
        verifier_id = f"{action.id}:{current_rpc_name() or 'launch_agent'}:{call_id}"
        flow._verifiers[verifier_id] = verify
        return verifier_id, fingerprint(verify)
    if (
        isinstance(verify, dict)
        and isinstance(verify.get("$rlmflow_verifier"), str)
        and verify["$rlmflow_verifier"].startswith(f"{action.id}:")
        and isinstance(verify.get("fingerprint"), str)
    ):
        return verify["$rlmflow_verifier"], verify["fingerprint"]
    raise TypeError("launch_agent verify must be a callable")


def _producer(step: Step, value: Any) -> AgentStart | None:
    agent_id = value.agent_id if isinstance(value, AgentHandle) else agent_ref_id(value)
    if agent_id is None:
        return None
    producer = step.graph.node(agent_id) if agent_id in step.graph else None
    if not isinstance(producer, AgentStart):
        raise ValueError(f"handle refers to unknown agent {agent_id!r}")
    return producer


def split_args(step: Step, args: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate JSON arguments from top-level handle inputs.

    A handle, or ``Input(handle, description=...)``, becomes a ``depends_on``
    input bound to its argument name. Handles nested inside other values are
    rejected.
    """
    if args is None:
        return {}, {}
    if not isinstance(args, dict):
        raise TypeError("launch_agent arguments must be a mapping")
    plain: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for argument_name, value in args.items():
        validate_argument_name(argument_name)
        wrapped = input_parts(value)
        raw, description = wrapped if wrapped is not None else (value, None)
        producer = _producer(step, raw)
        if producer is not None:
            inputs[argument_name] = (
                Input(producer, description=description) if description else producer
            )
            continue
        visit_argument(raw, _check_json, path=argument_name)
        plain[argument_name] = value
    return plain, inputs


def _check_json(path: str, value: Any) -> Any:
    if isinstance(value, AgentHandle) or agent_ref_id(value) is not None:
        raise TypeError(f"argument {path!r} nests a handle; pass handles as top-level arguments")
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


def resolve_child(
    flow: Any,
    action: ExecAction,
    spec: dict[str, Any],
    call_id: int,
) -> AgentStart:
    """Create one new direct child of the stepping agent."""
    step = current_step()
    g, parent = step.graph, step.agent
    task = spec.get("task", "")
    used = {child.config.name for child in reads.children(g, parent)}
    name = spec.get("name")
    if name is None:
        index = call_id
        while f"child{index}" in used:
            index += 1
        name = f"child{index}"
    validate_agent_name(name)
    if name in used:
        raise ValueError(f"duplicate child name {name!r}")
    if reads.depth(g, parent) >= parent.config.max_depth:
        raise ValueError(f"cannot launch beyond max depth {parent.config.max_depth}")
    if len(task) > parent.config.max_query_chars:
        raise ValueError(f"subagent goal exceeds {parent.config.max_query_chars} characters")
    return new_child(flow, action, name, spec)


def new_child(flow: Any, action: Node, name: str, spec: dict[str, Any]) -> AgentStart:
    """Spawn a child of the stepping agent, launched by ``action``."""
    step = current_step()
    parent = step.agent
    if flow.budget_exhausted(step.graph):
        raise RuntimeError("cannot launch a new subagent after max_budget is exhausted")
    overrides = {
        key: spec[key]
        for key in ("output_schema", "verifier_id", "verifier_hash")
        if spec.get(key) is not None
    }
    config = parent.config.child(name, **overrides)
    if config.backend == NATIVE_BACKEND and config.model not in flow._llm_clients:
        available = ", ".join(sorted(flow._llm_clients))
        raise ValueError(f"unknown model {config.model!r}; available models: {available}")
    return ops.spawn(
        step.graph,
        spec.get("task", ""),
        by=action,
        config=config,
        inputs=spec.get("inputs") or None,
        **(spec.get("args") or {}),
    )


def _fork_point(g: Graph, agent: AgentStart) -> Node:
    """The latest node of ``agent`` a fork can start from."""
    node = reads.frontier(g, agent)
    while isinstance(node, ActionNode):
        node = reads.prev(g, node)
    return node


def fork_tool(flow: Any, node: Node):
    step = current_step()
    g = step.graph

    def restore(saved: object, *args: Any, **kwargs: Any) -> AgentHandle:
        return agent_handle(g, resolve_agent(g, agent_ref_id(saved)))

    @tool(
        "Fork an agent's history into a new agent that continues independently. "
        "With no handle, forks yourself at this turn; `message=` is then required "
        "and becomes the fork's next instruction.",
        proxy=True,
        is_async=False,
        restore=restore,
    )
    def fork_agent(
        handle: Any = None,
        *,
        message: str | None = None,
        name: str | None = None,
    ) -> AgentHandle:
        if not isinstance(node, ExecAction):
            raise TypeError("fork_agent requires an ExecAction")
        parent = step.agent
        if handle is None:
            if not message:
                raise ValueError("forking yourself needs message=, or the fork reruns this block")
            at = reads.prev(g, node)
        else:
            at = _fork_point(g, resolve_agent(g, handle))
        if reads.depth(g, parent) >= parent.config.max_depth:
            raise ValueError(f"cannot fork beyond max depth {parent.config.max_depth}")
        if flow.budget_exhausted(g):
            raise RuntimeError("cannot fork after max_budget is exhausted")
        used = {child.config.name for child in reads.children(g, parent)}
        if name is None:
            owner = g.node(at.agent_id).config.name
            index = len(used)
            while f"{owner}_fork{index}" in used:
                index += 1
            name = f"{owner}_fork{index}"
        elif name in used:
            raise ValueError(f"duplicate child name {name!r}")
        fork = ops.fork(g, at, name=name, by=node)
        if message:
            ops.send(g, fork, message, sender=node)
        if flow.queue is not None:
            flow.queue.wake()
        return agent_handle(g, fork)

    return fork_agent


__all__ = ["fork_tool", "launch_tool", "new_child", "resolve_child", "split_args"]
