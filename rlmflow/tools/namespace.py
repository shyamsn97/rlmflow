"""REPL namespace: host tools, reserved builtins, and per-node factories."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from rlmflow.graph.config import Input
from rlmflow.graph.nodes import ExecAction, Node
from rlmflow.tools import RESERVED_TOOLS
from rlmflow.tools.agents import (
    AGENT_OBSERVE_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    build_agent_directory,
)
from rlmflow.tools.tools import tool, tool_allowed
from rlmflow.utils.helpers import tool_name


def as_tool_items(tools: Any) -> list[Any]:
    """Normalize ``Flow(tools=...)`` to a flat list of tools.

    One level of nesting is flattened, so ``tools=[FILE_TOOLS, migrate]`` and
    ``tools=FILE_TOOLS`` both work.
    """
    if tools is None:
        return []
    if callable(tools):
        return [tools]
    if isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
        items: list[Any] = []
        for item in tools:
            if not callable(item) and isinstance(item, Iterable) and not isinstance(
                item, (str, bytes)
            ):
                items.extend(item)
            else:
                items.append(item)
        return items
    raise TypeError(f"tools must be a tool or a sequence of tools, not {type(tools).__name__}")


def inject(flow: Any, name: str, value: Any) -> None:
    if name in RESERVED_TOOLS:
        raise ValueError(f"{name!r} is reserved")
    flow.tools[name] = value
    flow.runtime.inject_live(name, value)


def add_tool(flow: Any, fn: Any, *, name: str | None = None) -> None:
    inject(flow, name or tool_name(fn), fn)


def remove_tool(flow: Any, name: str) -> Any:
    if name in RESERVED_TOOLS:
        raise ValueError(f"{name!r} is reserved")
    flow.runtime.remove_live(name)
    return flow.tools.pop(name, None)


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


def application_tools(flow: Any, node: Node) -> dict[str, Any]:
    """Return configured application tools allowed on this node."""
    return {
        name: value
        for name, value in flow.tools.items()
        if tool_allowed(value, flow, node)
    }


def build_namespace(flow: Any, node: Node, tools: dict[str, Any]) -> dict[str, Any]:
    """Add non-tool runtime bindings to an already-built tool surface."""
    running = (
        tuple(current for current, _task in flow.queue.running.values())
        if flow.queue is not None
        else ()
    )
    agents = build_agent_directory(node.parent_agent, running_nodes=running)
    namespace = {
        **tools,
        "Input": Input,
        "asyncio": asyncio,
        AGENTS_BINDING: agents,
        AGENT_OBSERVE_TOOL: observe_agent_tool(flow, node),
        AGENT_WAIT_TOOL: wait_agent_tool(flow, node),
    }
    # Keep old REPL blocks executable without advertising the legacy name.
    if "submit" in tools:
        namespace["finish"] = tools["submit"]
    if flow.use_agent_tree:
        namespace["AGENTS"] = agents
    return namespace


__all__ = [
    "add_tool",
    "application_tools",
    "as_tool_items",
    "build_namespace",
    "inject",
    "observe_agent_tool",
    "remove_tool",
    "wait_agent_tool",
]
