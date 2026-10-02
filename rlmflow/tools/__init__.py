"""Tool metadata, prompt formatting, and built-in file tools."""

from rlmflow.tools.agents import (
    AGENT_OBSERVE_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    AgentDirectory,
    AgentFrontier,
    AgentHandle,
    AgentInfo,
    AgentStatus,
)
from rlmflow.tools.coding import CODING_HANDOFF_SCHEMA
from rlmflow.tools.filesystem import FILE_TOOLS, edit_file
from rlmflow.tools.tools import (
    ToolMetadata,
    format_tool_docs,
    get_tool_metadata,
    tool,
)

RESERVED_TOOLS = frozenset(
    {
        "finish",
        "launch_subagent",
        "llm_query",
        "asyncio",
        "ENV",
        "AGENTS",
        AGENTS_BINDING,
        AGENT_OBSERVE_TOOL,
        AGENT_WAIT_TOOL,
    }
)

__all__ = [
    "AgentDirectory",
    "AgentFrontier",
    "AgentHandle",
    "AgentInfo",
    "AgentStatus",
    "CODING_HANDOFF_SCHEMA",
    "FILE_TOOLS",
    "RESERVED_TOOLS",
    "ToolMetadata",
    "edit_file",
    "format_tool_docs",
    "get_tool_metadata",
    "tool",
]
