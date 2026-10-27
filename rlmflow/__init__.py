"""Build recursive agents as inspectable, graph-backed runs."""

from importlib import import_module
from typing import Any

from rlmflow.engine.scheduler import (
    Boundary,
    Pool,
    SequentialPool,
    StepResult,
    TaskQueue,
    ThreadPool,
)
from rlmflow.graph import persistence
from rlmflow.graph.config import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    Input,
    at_leaf,
    can_spawn,
)
from rlmflow.graph.graph import (
    DEPENDS_ON,
    FROM,
    NEXT,
    REPORTS_TO,
    Edge,
    Graph,
    InvariantError,
    StaleFrontier,
)
from rlmflow.graph.nodes import (
    AGENT_START,
    CODE,
    DONE,
    LLM_OUTPUT,
    MESSAGE,
    NODE_TYPES,
    WAITING,
    LLMUsage,
    Node,
    Step,
    current_code,
    current_step,
    record_metadata,
    running_step,
)
from rlmflow.llm import LLMChunk, LLMClient, PooledLLMClient, is_retryable, retry_transient
from rlmflow.llm.anthropic import AnthropicClient
from rlmflow.llm.openai import OpenAIClient
from rlmflow.llm.tinker import TinkerClient
from rlmflow.schema import SchemaError, json_schema_for

#: Names resolved on first access, so importing ``rlmflow`` doesn't load the REPL
#: runtime, Textual, the renderers' optional dependencies, or ``dspy``.
_LAZY = {
    "Flow": "rlmflow.flow",
    "FlowLLM": "rlmflow.flow",
    "StepUntil": "rlmflow.flow",
    "messages_to_query": "rlmflow.flow",
    "GitWorktrees": "rlmflow.engine.workspaces",
    "LocalWorkspace": "rlmflow.engine.workspaces",
    "AgentDirectory": "rlmflow.agent_tools.view",
    "AgentHandle": "rlmflow.agent_tools.view",
    "AgentInfo": "rlmflow.agent_tools.view",
    "AgentStatus": "rlmflow.agent_tools.view",
    "AgentToolsServer": "rlmflow.agent_tools.server",
    "ClaudeCodeHarness": "rlmflow.harnesses.claude_code",
    "Harness": "rlmflow.harnesses.base",
    "InteractiveHarness": "rlmflow.harnesses.interactive",
    "PiHarness": "rlmflow.harnesses.pi",
    "ReplHarness": "rlmflow.harnesses.repl.loop",
    "SessionHarness": "rlmflow.harnesses.sessions",
    "TurnResult": "rlmflow.harnesses.sessions",
    "CODING_HANDOFF_SCHEMA": "rlmflow.harnesses.repl.files",
    "FILE_TOOLS": "rlmflow.harnesses.repl.files",
    "tool": "rlmflow.harnesses.repl.tools",
    "ToolMetadata": "rlmflow.harnesses.repl.runtime",
    "get_tool_metadata": "rlmflow.harnesses.repl.runtime",
    "DockerRuntime": "rlmflow.harnesses.repl.runtime",
    "DoneSignal": "rlmflow.harnesses.repl.runtime",
    "LocalRuntime": "rlmflow.harnesses.repl.runtime",
    "ModalRuntime": "rlmflow.harnesses.repl.runtime",
    "Repl": "rlmflow.harnesses.repl.runtime",
    "ReplRun": "rlmflow.harnesses.repl.runtime",
    "ReplStatus": "rlmflow.harnesses.repl.runtime",
    "Runtime": "rlmflow.harnesses.repl.runtime",
    "SubprocessRuntime": "rlmflow.harnesses.repl.runtime",
    "WorkerRepl": "rlmflow.harnesses.repl.runtime",
    "WorkerSession": "rlmflow.harnesses.repl.runtime",
    "ConsumerGroup": "rlmflow.consumers",
    "GraphCheckpointer": "rlmflow.consumers",
    "LiveGraphTree": "rlmflow.consumers",
    "LiveTreeRenderer": "rlmflow.consumers",
    "StreamConsumer": "rlmflow.consumers",
    "WorkspaceSync": "rlmflow.consumers",
    "FlowTUI": "rlmflow.cli.tui",
    "DSPyFlow": "rlmflow.integrations.dspy",
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
    "AGENT_START",
    "CODE",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "DEPENDS_ON",
    "DONE",
    "FROM",
    "LLM_OUTPUT",
    "MESSAGE",
    "NEXT",
    "NODE_TYPES",
    "REPORTS_TO",
    "WAITING",
    "AgentConfig",
    "AnthropicClient",
    "Boundary",
    "Edge",
    "Graph",
    "Input",
    "InvariantError",
    "LLMChunk",
    "LLMClient",
    "LLMUsage",
    "Node",
    "OpenAIClient",
    "Pool",
    "PooledLLMClient",
    "SchemaError",
    "SequentialPool",
    "StaleFrontier",
    "Step",
    "StepResult",
    "TaskQueue",
    "ThreadPool",
    "TinkerClient",
    "at_leaf",
    "can_spawn",
    "current_code",
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
