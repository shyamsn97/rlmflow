"""Let coding agents build multi-agent systems in code, and record every run as a graph you can edit."""

from importlib import import_module
from typing import Any

from rlmflow.graph import persistence
from rlmflow.graph.config import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    Input,
    at_leaf,
    can_spawn,
)
from rlmflow.graph.edits import RULES, GraphError, Invalidation
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    FORK,
    RETURN,
    SPAWN,
    WRITE,
    Edge,
    LLMUsage,
    Node,
    RootData,
    Step,
    attach_image,
    current_step,
    record_metadata,
    running_step,
)
from rlmflow.llm import (
    LLMChunk,
    LLMClient,
    client_for,
    complete,
    is_retryable,
    retry_transient,
)
from rlmflow.llm.anthropic import AnthropicClient
from rlmflow.llm.openai import OpenAIClient
from rlmflow.schema import SchemaError, json_schema_for

#: Names resolved on first access, so importing ``rlmflow`` doesn't start the REPL
#: runtime or load the renderers' optional dependencies.
_LAZY = {
    "Flow": "rlmflow.flow",
    "StepUntil": "rlmflow.flow",
    "GitWorktrees": "rlmflow.workspaces",
    "LocalWorkspace": "rlmflow.workspaces",
    "AgentDirectory": "rlmflow.agent_tools.view",
    "AgentHandle": "rlmflow.agent_tools.view",
    "AgentStatus": "rlmflow.agent_tools.view",
    "AgentToolsServer": "rlmflow.agent_tools.server",
    "ClaudeCodeHarness": "rlmflow.harnesses.claude_code",
    "Harness": "rlmflow.harnesses.base",
    "InteractiveHarness": "rlmflow.harnesses.interactive",
    "PiHarness": "rlmflow.harnesses.pi",
    "ReplHarness": "rlmflow.harnesses.repl",
    "SessionHarness": "rlmflow.harnesses.sessions",
    "TurnResult": "rlmflow.harnesses.sessions",
    "CODING_HANDOFF_SCHEMA": "rlmflow.tools.files",
    "FILE_TOOLS": "rlmflow.tools.files",
    "MCPTools": "rlmflow.integrations.mcp",
    "NotYourAgent": "rlmflow.runtime.repl",
    "Toolkit": "rlmflow.tools",
    "tool": "rlmflow.tools",
    "tool_from_schema": "rlmflow.tools",
    "ToolMetadata": "rlmflow.runtime",
    "get_tool_metadata": "rlmflow.runtime",
    "DockerRuntime": "rlmflow.runtime",
    "DoneSignal": "rlmflow.runtime",
    "LocalRuntime": "rlmflow.runtime",
    "ModalRuntime": "rlmflow.runtime",
    "Repl": "rlmflow.runtime",
    "ReplRun": "rlmflow.runtime",
    "ReplStatus": "rlmflow.runtime",
    "Runtime": "rlmflow.runtime",
    "SubprocessRuntime": "rlmflow.runtime",
    "WorkerRepl": "rlmflow.runtime",
    "WorkerSession": "rlmflow.runtime",
    "ConsumerGroup": "rlmflow.consumers",
    "GraphCheckpointer": "rlmflow.consumers",
    "LiveGraphTree": "rlmflow.consumers",
    "LiveTreeRenderer": "rlmflow.consumers",
    "StreamConsumer": "rlmflow.consumers",
    "WorkspaceSync": "rlmflow.consumers",
    "code_block": "rlmflow.utils",
    "find_code_blocks": "rlmflow.utils",
    "render_forest": "rlmflow.view.text",
    "render_tree": "rlmflow.view.text",
    "graph_svg": "rlmflow.view",
    "open_viewer": "rlmflow.view",
    "render_html": "rlmflow.view",
    "render_steps": "rlmflow.view",
    "replay": "rlmflow.view",
    "save_frames": "rlmflow.view",
    "save_gif": "rlmflow.view",
    "save_html": "rlmflow.view",
    "save_svg": "rlmflow.view",
    "snapshot": "rlmflow.view",
    "steps": "rlmflow.view",
    "timeline": "rlmflow.view",
}


def __getattr__(name: str) -> Any:
    if name == "view":
        return import_module("rlmflow.view")
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "FORK",
    "RETURN",
    "RULES",
    "SPAWN",
    "WRITE",
    "AgentConfig",
    "AnthropicClient",
    "Edge",
    "Graph",
    "GraphError",
    "Input",
    "Invalidation",
    "LLMChunk",
    "LLMClient",
    "LLMUsage",
    "Node",
    "OpenAIClient",
    "RootData",
    "SchemaError",
    "Step",
    "at_leaf",
    "attach_image",
    "can_spawn",
    "client_for",
    "complete",
    "current_step",
    "is_retryable",
    "json_schema_for",
    "persistence",
    "record_metadata",
    "retry_transient",
    "running_step",
    "view",
    *sorted(_LAZY),
]
