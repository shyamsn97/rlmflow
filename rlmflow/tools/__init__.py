"""Tool metadata, prompt formatting, and built-in file tools."""

from rlmflow.tools.agents import AgentDirectory, AgentFrontier, AgentInfo, AgentStatus
from rlmflow.tools.filesystem import (
    FILE_TOOLS,
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
    format_tool_line,
    get_tool_metadata,
    partition_repl_namespace,
    tool,
)

RESERVED_TOOLS = frozenset({"done", "launch_subagents", "INPUTS", "AGENTS"})

__all__ = [
    "AgentDirectory",
    "AgentFrontier",
    "AgentInfo",
    "AgentStatus",
    "FILE_TOOLS",
    "RESERVED_TOOLS",
    "ToolMetadata",
    "append_file",
    "edit_file",
    "format_tool_line",
    "get_tool_metadata",
    "grep",
    "line_count",
    "ls",
    "partition_repl_namespace",
    "read_file",
    "read_lines",
    "tool",
    "write_file",
]
