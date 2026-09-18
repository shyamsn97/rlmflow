"""REPL namespace: host tools, reserved builtins, and per-node factories."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from rlmflow.engine.delegation import launch_tool
from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import ExecAction, Node
from rlmflow.runtime.repl import DoneSignal, TransitionSignal
from rlmflow.structured import as_finish_value, parse_structured_answer
from rlmflow.tools import RESERVED_TOOLS
from rlmflow.tools.agents import (
    AGENT_OBSERVE_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    build_agent_directory,
)
from rlmflow.tools.tools import host_tools_for, is_toolset, tool, toolset_members
from rlmflow.utils.helpers import tool_name


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a list of tools and toolsets.

    A toolset is one item, same as a function. ``tools=[FILE_TOOLS, grep_extra]``
    and ``tools=FILE_TOOLS`` both work; the list is not "one toolset or many
    functions."
    """
    if tools is None:
        return []
    if is_toolset(tools) or callable(tools):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        return list(tools)
    raise TypeError(
        f"tools must be a tool, a toolset, or a sequence of them, not {type(tools).__name__}"
    )


def inject(flow: Any, name: str, value: Any) -> None:
    if name in RESERVED_TOOLS:
        raise ValueError(f"{name!r} is reserved")
    flow.tools[name] = value
    flow.runtime.inject_live(name, value)


def add_tool(flow: Any, fn: Any, *, name: str | None = None) -> None:
    if is_toolset(fn):
        instance = fn() if isinstance(fn, type) else fn
        bind_toolset(flow, instance)
        return
    inject(flow, name or tool_name(fn), fn)


def bind_toolset(flow: Any, instance: Any) -> None:
    for existing in flow.toolsets:
        if type(existing) is type(instance):
            raise ValueError(f"toolset {type(instance).__name__!r} is already bound")
    pending: list[tuple[str, Any]] = []
    for tool_name_, method in toolset_members(instance):
        meta = getattr(getattr(method, "__func__", method), "_tool_meta", None)
        if meta is None or not meta.inject:
            continue
        if tool_name_ in RESERVED_TOOLS:
            raise ValueError(f"{tool_name_!r} is reserved")
        if tool_name_ in flow.tools:
            raise ValueError(
                f"tool {tool_name_!r} from {type(instance).__name__} collides "
                f"with an already-bound name"
            )
        pending.append((tool_name_, method))
    flow.toolsets.append(instance)
    for tool_name_, method in pending:
        inject(flow, tool_name_, method)


def remove_tool(flow: Any, name: str) -> Any:
    if name in RESERVED_TOOLS:
        raise ValueError(f"{name!r} is reserved")
    flow.runtime.remove_live(name)
    return flow.tools.pop(name, None)


def finish_tool(flow: Any, node: Node):
    schema = node.parent_agent.config.output_schema

    @tool(
        "Submit this agent's final answer as a string or strict JSON value and end its run.",
        proxy=True,
    )
    def finish(answer: object) -> None:
        value = (
            parse_structured_answer(answer, schema)
            if schema is not None
            else as_finish_value(answer)
        )
        raise DoneSignal(value)

    return finish


def transition_tool(flow: Any, node: Node):
    @tool(
        "End this action immediately and enter a named behavior. Stdout from "
        "the transitioning block is not observed.",
        proxy=True,
    )
    def transition(name: str) -> None:
        raise TransitionSignal(str(name))

    return transition


def wait_agent_tool(flow: Any, node: Node):
    @tool("Wait for an existing agent and return its result.", proxy=True)
    async def wait_agent(agent_id: str) -> Any:
        root = node.root
        if root is None:
            raise RuntimeError("node is detached")
        target = root.find_agent(agent_id)
        if target is None:
            raise KeyError(f"unknown agent {agent_id!r}")
        if not target.terminal:
            queue = flow.queue
            if queue is None:
                raise RuntimeError("waiting for an agent requires an active stream")
            await queue.join(target)
        if isinstance(node, ExecAction):
            node.mark_agent_retrieved(agent_id)
        return target.result()

    return wait_agent


def observe_agent_tool(flow: Any, node: Node):
    @tool("Record access to one completed agent result.", proxy=True)
    def observe_agent(agent_id: str) -> None:
        root = node.root
        if root is None:
            raise RuntimeError("node is detached")
        target = root.find_agent(agent_id)
        if target is None:
            raise KeyError(f"unknown agent {agent_id!r}")
        if not target.terminal:
            raise asyncio.InvalidStateError(f"agent {target.config.path!r} is not completed")
        if isinstance(node, ExecAction):
            node.mark_agent_retrieved(agent_id)

    return observe_agent


def build_namespace(flow: Any, node: Node) -> dict[str, Any]:
    running = (
        tuple(current for current, _task in flow.queue.running.values())
        if flow.queue is not None
        else ()
    )
    agents = build_agent_directory(node.parent_agent, running_nodes=running)
    namespace = {
        **host_tools_for(flow, node),
        "finish": finish_tool(flow, node),
        "transition": transition_tool(flow, node),
        "asyncio": asyncio,
        "INPUTS": node.parent_agent.resolved_inputs(),
        AGENTS_BINDING: agents,
        AGENT_OBSERVE_TOOL: observe_agent_tool(flow, node),
        AGENT_WAIT_TOOL: wait_agent_tool(flow, node),
    }
    if can_spawn(node.parent_agent):
        namespace["launch_subagent"] = launch_tool(flow, node)
    if flow.use_agent_tree:
        namespace["AGENTS"] = agents
    return namespace


__all__ = [
    "add_tool",
    "as_tool_items",
    "bind_toolset",
    "build_namespace",
    "finish_tool",
    "inject",
    "observe_agent_tool",
    "remove_tool",
    "transition_tool",
    "wait_agent_tool",
]
