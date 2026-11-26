"""Assembly of model-visible tools and the REPL namespace."""

from __future__ import annotations

import asyncio
from typing import Any

from rlmflow.graph.config import Input
from rlmflow.repl import Repl
from rlmflow.runtime.repl import (
    AGENT_SEND_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    NotYourAgent,
)
from rlmflow.tools.agents import build_agent_directory
from rlmflow.tools.builtins import (
    await_tool,
    checkout_tool,
    launch_tool,
    llm_query_tool,
    send_tool,
    submit_tool,
)
from rlmflow.tools.declare import tool_allowed


def repl_tools(repl: Repl) -> dict[str, Any]:
    """The model-visible tools for one run, built-ins first."""
    flow = repl.flow
    tools: dict[str, Any] = {"submit": submit_tool(repl)}
    if repl.can_launch():
        tools["launch_agent"] = launch_tool(repl)
    tools["llm_query"] = llm_query_tool(repl)
    if getattr(flow.workspaces, "isolates_launches", False):
        tools["checkout"] = checkout_tool(repl)
    tools.update(
        {name: fn for name, fn in flow.tools_for(repl.graph).items() if tool_allowed(fn, repl)}
    )
    return tools


def build_namespace(repl: Repl) -> dict[str, Any]:
    """Everything the REPL namespace binds: the tools plus ``AGENTS`` and ``Input``."""
    agents = build_agent_directory(repl.graph, repl.root)
    return {
        **repl_tools(repl),
        **repl.flow._bound.get(repl.id, {}),
        "Input": Input,
        "NotYourAgent": NotYourAgent,
        "asyncio": asyncio,
        "AGENTS": agents,
        AGENTS_BINDING: agents,
        AGENT_WAIT_TOOL: await_tool(repl),
        AGENT_SEND_TOOL: send_tool(repl),
    }
