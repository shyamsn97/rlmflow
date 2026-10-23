"""Tools: the ``@tool`` decorator, prompt formatting, and the REPL namespace."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rlmflow.graph import reads
from rlmflow.graph.config import Input
from rlmflow.graph.nodes import WAITED_AGENTS_KEY, AgentStart, Node, current_step, record_metadata
from rlmflow.runtime.repl import (
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    ToolMetadata,
    ToolRestore,
    get_tool_metadata,
)
from rlmflow.tools.agents_view import (
    build_agent_directory,
)


def _default_tool_name(name: str) -> str:
    return name.removeprefix("tool_")


def tool(
    description: str,
    *,
    name: str | None = None,
    proxy: bool = False,
    is_async: bool | None = None,
    when: Callable[..., bool] | None = None,
    restore: ToolRestore = None,
) -> Callable:
    """Attach tool metadata to ``fn``.

    ``is_async`` records whether agent code must ``await`` the tool. It defaults
    to auto-detection from the function (``async def`` -> ``True``); pass it
    explicitly only for a proxy that returns a coroutine without being
    ``async def``. (``async`` is a keyword, hence ``is_async``.)

    ``when`` is ``(flow, node) -> bool``. When it returns false, the tool is
    omitted from both the prompt and the REPL for that agent.

    ``restore="result"`` saves the result of a successful call and returns it
    while rebuilding a REPL namespace. A callable receives the saved result
    followed by the original arguments and may rebuild a richer runtime value.
    """
    if restore is not None and restore != "result" and not callable(restore):
        raise TypeError("tool restore must be None, 'result', or a callable")
    if restore is not None and not proxy:
        raise ValueError("tool restore requires proxy=True")

    def decorator(fn):
        resolved = inspect.iscoroutinefunction(fn) if is_async is None else is_async
        fn._tool_meta = ToolMetadata(
            name=name or _default_tool_name(fn.__name__),
            description=description.strip(),
            proxy=proxy,
            is_async=resolved,
            when=when,
            restore=restore,
        )
        return fn

    return decorator


def tool_allowed(fn: Any, flow: Any = None, node: Any = None) -> bool:
    """Whether ``fn``'s ``when`` predicate allows it on this node."""
    meta = get_tool_metadata(fn)
    if meta is None or meta.when is None:
        return True
    return bool(meta.when(flow, node))


def format_tool_line(fn: Callable) -> str:
    meta = get_tool_metadata(fn)
    if meta is None:
        return ""
    try:
        sig = str(inspect.signature(fn, eval_str=True))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("rlmflow.tools.agents_view.", "")
    prefix = "await " if meta.is_async else ""
    return f"- `{prefix}{meta.name}{sig}`: {meta.description}"


def format_tool_docs(tools: Mapping[str, Any]) -> str:
    """Render one ``### Tools`` list for the callables an agent has bound."""
    lines = [line for value in tools.values() if (line := format_tool_line(value))]
    return "\n".join(["### Tools", *lines]) if lines else ""


def tool_name(fn: Any) -> str:
    """Registered name of a tool callable, falling back to ``fn.__name__``."""
    meta = get_tool_metadata(fn)
    return meta.name if meta is not None else fn.__name__


RESERVED_TOOLS = frozenset(
    {
        "finish",
        "submit",
        "launch_agent",
        "launch_subagent",
        "fork_agent",
        "send_agent",
        "ask_agent",
        "connect_agent",
        "disconnect_agent",
        "llm_query",
        "Input",
        "asyncio",
        "ENV",
        "AGENTS",
        AGENTS_BINDING,
        AGENT_WAIT_TOOL,
    }
)


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
            if (
                not callable(item)
                and isinstance(item, Iterable)
                and not isinstance(item, (str, bytes))
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


def _target(graph: Any, agent_id: str) -> AgentStart:
    target = graph.node(agent_id) if agent_id in graph else None
    if not isinstance(target, AgentStart):
        raise KeyError(f"unknown agent {agent_id!r}")
    return target


def wait_agent_tool(flow: Any, node: Node):
    from rlmflow.tools.messaging import waiting

    step = current_step()
    graph = step.graph

    @tool("Wait for an existing agent and return its result.", proxy=True)
    async def wait_agent(agent_id: str) -> Any:
        target = _target(graph, agent_id)
        if not reads.terminal(graph, target):
            queue = flow.queue
            if queue is None:
                raise RuntimeError("waiting for an agent requires an active stream")
            with waiting(flow, graph, step.agent, target):
                await queue.join(lambda: reads.terminal(graph, target))
        record_metadata(WAITED_AGENTS_KEY, target.id, append=True)
        return reads.result(graph, target)

    return wait_agent


def application_tools(flow: Any, node: Node) -> dict[str, Any]:
    """Return configured application tools allowed on this node."""
    return {name: value for name, value in flow.tools.items() if tool_allowed(value, flow, node)}


def build_namespace(flow: Any, node: Node, tools: dict[str, Any]) -> dict[str, Any]:
    """Add non-tool runtime bindings to an already-built tool surface."""
    step = current_step()
    agents = build_agent_directory(step.graph, step.agent, running=flow.running_steps())
    namespace = {
        **tools,
        "Input": Input,
        "asyncio": asyncio,
        AGENTS_BINDING: agents,
        AGENT_WAIT_TOOL: wait_agent_tool(flow, node),
    }
    # Keep old REPL blocks executable without advertising the legacy name.
    if "submit" in tools:
        namespace["finish"] = tools["submit"]
    if "launch_agent" in tools:
        namespace["launch_subagent"] = tools["launch_agent"]
    if flow.use_agent_tree:
        namespace["AGENTS"] = agents
    return namespace


__all__ = [
    "RESERVED_TOOLS",
    "add_tool",
    "application_tools",
    "as_tool_items",
    "build_namespace",
    "format_tool_docs",
    "inject",
    "remove_tool",
    "tool",
    "tool_allowed",
    "tool_name",
    "wait_agent_tool",
]
