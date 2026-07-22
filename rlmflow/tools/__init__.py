"""Tool metadata, prompt formatting, and built-in file tools."""

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
from rlmflow.tools.graph_ops import (
    HaltFn,
    ToolFactory,
    enable_graph_ops,
    inject_tools,
    register_halt,
)
from rlmflow.tools.tools import (
    ToolMetadata,
    format_tool_line,
    get_tool_metadata,
    partition_repl_namespace,
    tool,
)

__all__ = [
    "FILE_TOOLS",
    "HaltFn",
    "ToolFactory",
    "ToolMetadata",
    "append_file",
    "edit_file",
    "enable_graph_ops",
    "format_tool_line",
    "get_tool_metadata",
    "grep",
    "inject_tools",
    "line_count",
    "ls",
    "partition_repl_namespace",
    "read_file",
    "read_lines",
    "register_halt",
    "tool",
    "write_file",
]
