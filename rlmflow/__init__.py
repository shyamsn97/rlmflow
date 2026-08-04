"""Build recursive agents as inspectable node trees."""

from importlib import import_module
from typing import Any

from rlmflow.consumers import (
    ConsumerGroup,
    FlowTUI,
    GraphCheckpointer,
    LiveGraphTree,
    LiveTreeRenderer,
    StreamConsumer,
    WorkspaceSync,
    render_forest,
    render_tree,
)
from rlmflow.engine import boundaries
from rlmflow.engine.execution import (
    Pool,
    SequentialPool,
    TaskQueue,
    ThreadPool,
    Transition,
)
from rlmflow.engine.parallel import parallel_run, parallel_stream
from rlmflow.graph import persistence
from rlmflow.graph.nodes import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    AgentStart,
    AppendChild,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    Node,
    UserQuery,
    start,
)
from rlmflow.llm import (
    AnthropicClient,
    LLMClient,
    LLMUsage,
    OpenAIClient,
    TinkerClient,
    is_retryable,
    retry_transient,
)
from rlmflow.prompts import (
    DEFAULT_BUILDER,
    SYSTEM_PROMPT,
    PromptBuilder,
    PromptProfile,
    SystemPromptBuilder,
    UserPromptBuilder,
)
from rlmflow.runtime import (
    DockerRuntime,
    DoneSignal,
    LocalRepl,
    LocalRuntime,
    MissingReplError,
    ModalRuntime,
    RemoteRepl,
    Repl,
    ReplRun,
    ReplStatus,
    Runtime,
    SubprocessRuntime,
)
from rlmflow.structured import (
    Schema,
    StructuredOutputError,
    StructuredOutputParser,
    json_schema_for,
    parse_structured_output,
    system_prompt_hint,
)
from rlmflow.tools import (
    FILE_TOOLS,
    ToolMetadata,
    format_tool_line,
    get_tool_metadata,
    tool,
)
from rlmflow.utils import find_code_blocks

#: Names resolved on first access. ``flow`` because it pulls in the shared prompt
#: stack, which imports these nodes back, so an eager import is a cycle;
#: ``adapters`` because it probes for ``dspy``, which is slow and optional.
_LAZY = {
    "Flow": "rlmflow.flow",
    "StepUntil": "rlmflow.flow",
    "code_block": "rlmflow.flow",
    "DSPyFlow": "rlmflow.adapters",
    "FlowLLM": "rlmflow.adapters",
    "messages_to_query": "rlmflow.adapters",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = [
    "DEFAULT_BUILDER",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "FILE_TOOLS",
    "SYSTEM_PROMPT",
    "AgentConfig",
    "AgentStart",
    "AnthropicClient",
    "AppendChild",
    "ConsumerGroup",
    "DSPyFlow",
    "DockerRuntime",
    "DoneOutput",
    "DoneSignal",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "Flow",
    "FlowLLM",
    "FlowTUI",
    "GraphCheckpointer",
    "LLMClient",
    "LLMOutput",
    "LLMUsage",
    "LiveGraphTree",
    "LiveTreeRenderer",
    "LocalRepl",
    "LocalRuntime",
    "MissingReplError",
    "ModalRuntime",
    "Node",
    "OpenAIClient",
    "Pool",
    "PromptBuilder",
    "PromptProfile",
    "RemoteRepl",
    "Repl",
    "ReplRun",
    "ReplStatus",
    "Runtime",
    "Schema",
    "SequentialPool",
    "StepUntil",
    "StreamConsumer",
    "StructuredOutputError",
    "StructuredOutputParser",
    "SubprocessRuntime",
    "SystemPromptBuilder",
    "TaskQueue",
    "ThreadPool",
    "TinkerClient",
    "ToolMetadata",
    "Transition",
    "UserPromptBuilder",
    "UserQuery",
    "WorkspaceSync",
    "boundaries",
    "code_block",
    "find_code_blocks",
    "format_tool_line",
    "get_tool_metadata",
    "is_retryable",
    "json_schema_for",
    "messages_to_query",
    "parallel_run",
    "parallel_stream",
    "parse_structured_output",
    "persistence",
    "render_forest",
    "render_tree",
    "retry_transient",
    "start",
    "system_prompt_hint",
    "tool",
]
