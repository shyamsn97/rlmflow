"""Tool metadata, prompt formatting, and built-in file tools."""

from rlmflow.graph.config import at_leaf, can_spawn
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
from rlmflow.tools.filesystem import (
    FILE_TOOLS,
    FileTools,
    append_file,
    edit_file,
    grep,
    line_count,
    ls,
    read_file,
    read_lines,
    write_file,
)
from rlmflow.tools.tools import (
    ToolMetadata,
    ToolsetMetadata,
    format_tool_docs,
    format_tool_line,
    get_tool_metadata,
    get_toolset_metadata,
    is_toolset,
    partition_repl_namespace,
    tool,
    toolset,
    toolset_members,
)

RESERVED_TOOLS = frozenset(
    {
        "finish",
        "transition",
        "launch_subagent",
        "asyncio",
        "INPUTS",
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
    "FileTools",
    "RESERVED_TOOLS",
    "ToolMetadata",
    "ToolsetMetadata",
    "append_file",
    "at_leaf",
    "can_spawn",
    "edit_file",
    "format_tool_docs",
    "format_tool_line",
    "get_tool_metadata",
    "get_toolset_metadata",
    "grep",
    "is_toolset",
    "line_count",
    "ls",
    "partition_repl_namespace",
    "read_file",
    "read_lines",
    "tool",
    "toolset",
    "toolset_members",
    "write_file",
]
