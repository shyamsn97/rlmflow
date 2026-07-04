"""Minimal graph-first rflow.

This package mirrors Tau's clean layering style while keeping rflow's core
choice: the recursive graph is the source of truth.
"""

from rflow.minimal.adapters import FlowLLM, MinimalDSPyLM, messages_to_query
from rflow.minimal.code import find_code_blocks
from rflow.minimal.events import (
    AddChild,
    AppendNode,
    Event,
    GraphAction,
    GraphCreated,
    RemoveChild,
    RemoveNode,
    ReplaceNode,
)
from rflow.minimal.flow import Flow, code_block
from rflow.minimal.graph import (
    ActionNode,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    LLMUsage,
    Node,
    ObservationNode,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    apply_graph_action,
    load_run,
    node_from_dict,
)
from rflow.minimal.pool import AsyncPool, SequentialPool
from rflow.minimal.prompts import DEFAULT_BUILDER, SYSTEM_PROMPT, PromptBuilder
from rflow.minimal.rendering import (
    LiveTreeRenderer,
    render_tree,
)
from rflow.minimal.repl import DoneSignal, MissingReplError, Repl
from rflow.minimal.runtime import (
    DockerRuntime,
    LocalRuntime,
    ModalRuntime,
    ReplLike,
    Runtime,
    SubprocessRuntime,
)
from rflow.minimal.structured import (
    Schema,
    StructuredOutputError,
    StructuredOutputParser,
    json_schema_for,
)
from rflow.minimal.tools import (
    FILE_TOOLS,
    ToolMetadata,
    format_tool_line,
    get_tool_metadata,
    tool,
)
from rflow.minimal.tui import tui
from rflow.minimal.viewer import open_viewer

__all__ = [
    "ActionNode",
    "AddChild",
    "AppendNode",
    "AsyncPool",
    "DEFAULT_BUILDER",
    "DoneOutput",
    "DoneSignal",
    "DockerRuntime",
    "ErrorOutput",
    "Event",
    "ExecAction",
    "ExecOutput",
    "FILE_TOOLS",
    "find_code_blocks",
    "Flow",
    "FlowLLM",
    "Graph",
    "GraphAction",
    "GraphCreated",
    "LLMOutput",
    "LLMUsage",
    "LiveTreeRenderer",
    "LocalRuntime",
    "MissingReplError",
    "ModalRuntime",
    "MinimalDSPyLM",
    "Node",
    "ObservationNode",
    "open_viewer",
    "PromptBuilder",
    "Repl",
    "ReplLike",
    "RemoveChild",
    "RemoveNode",
    "ReplaceNode",
    "ResumeAction",
    "Runtime",
    "SYSTEM_PROMPT",
    "Schema",
    "SequentialPool",
    "StructuredOutputError",
    "StructuredOutputParser",
    "SubprocessRuntime",
    "SupervisingOutput",
    "ToolMetadata",
    "UserQuery",
    "apply_graph_action",
    "code_block",
    "format_tool_line",
    "get_tool_metadata",
    "json_schema_for",
    "load_run",
    "messages_to_query",
    "node_from_dict",
    "render_tree",
    "tool",
    "tui",
]
