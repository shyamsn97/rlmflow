"""rlmflow (minimal): an LLM in a loop with a stateful REPL, recursive.

Quick start::

    from rflow import Flow
    from rflow.minimal.clients import OpenAIClient

    flow = Flow(OpenAIClient(model="gpt-4o"))
    print(flow.run("What is 17 * 23? Verify with code."))

The whole tree advances by synchronized steps; drive it yourself with
``start`` / ``step`` to inspect or visualize each tick::

    graph = flow.start("research X")
    while not graph.finished:
        graph = flow.step()
        ...  # inspect graph
"""

from rflow.base import BaseFlow, BaseOutputParser
from rflow.minimal.clients import (
    AnthropicClient,
    LLMChannel,
    LLMClient,
    LLMUsage,
    OpenAIClient,
    TinkerClient,
    is_retryable,
    retry_transient,
)
from rflow.flow import Flow, find_code_blocks, parallel_step
from rflow.graph import (
    Action,
    ActionNode,
    CallLLM,
    ChildHandle,
    CodeObservation,
    DoneOutput,
    Edge,
    EdgesView,
    ErrorOutput,
    Exec,
    ExecAction,
    ExecOutput,
    Graph,
    GraphEdited,
    GraphEvent,
    GraphNodeCommitted,
    LLMAction,
    LLMOutput,
    Node,
    NodesView,
    ObservationNode,
    Recover,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    WaitRequest,
    inject,
    inject_output,
    is_action,
    is_code_observation,
    is_done,
    is_errored,
    is_exec_action,
    is_exec_output,
    is_llm_action,
    is_llm_output,
    is_observation,
    is_resume_action,
    is_resumed,
    is_supervising,
    is_user_query,
    parse_node_obj,
    prune_descendants_spawned_after,
    replace_last_action,
    replace_last_observation,
    replace_node,
    retrace_steps,
    truncate_after,
    truncate_agent,
)
from rflow.integrations.tui import tui
from rflow.prompts import (
    DEFAULT_BUILDER,
    SYSTEM_PROMPT,
    PromptBuilder,
    create_final_action_message,
    create_nudge_message,
)
from rflow.runtime import DockerRuntime, LocalRuntime, Runtime, SubprocessRuntime
from rflow.runtime.repl import REPL, DoneSignal
from rflow.tools import FILE_TOOLS, get_tool_metadata, tool
from rflow.utils.trace import Trace, load_trace, save_trace

__all__ = [
    "Action",
    "ActionNode",
    "AnthropicClient",
    "BaseFlow",
    "BaseOutputParser",
    "CallLLM",
    "ChildHandle",
    "CodeObservation",
    "DEFAULT_BUILDER",
    "DockerRuntime",
    "DoneOutput",
    "DoneSignal",
    "Edge",
    "EdgesView",
    "ErrorOutput",
    "Exec",
    "ExecAction",
    "ExecOutput",
    "FILE_TOOLS",
    "Flow",
    "Graph",
    "GraphEdited",
    "GraphEvent",
    "GraphNodeCommitted",
    "LLMAction",
    "LLMChannel",
    "LLMClient",
    "LLMOutput",
    "LLMUsage",
    "LocalRuntime",
    "Node",
    "NodesView",
    "ObservationNode",
    "OpenAIClient",
    "PromptBuilder",
    "parallel_step",
    "Recover",
    "REPL",
    "ResumeAction",
    "Runtime",
    "SYSTEM_PROMPT",
    "SupervisingOutput",
    "SubprocessRuntime",
    "TinkerClient",
    "Trace",
    "UserQuery",
    "WaitRequest",
    "create_final_action_message",
    "create_nudge_message",
    "find_code_blocks",
    "get_tool_metadata",
    "inject",
    "inject_output",
    "load_trace",
    "prune_descendants_spawned_after",
    "replace_last_action",
    "replace_last_observation",
    "replace_node",
    "retrace_steps",
    "save_trace",
    "truncate_after",
    "truncate_agent",
    "is_action",
    "is_code_observation",
    "is_done",
    "is_errored",
    "is_exec_action",
    "is_exec_output",
    "is_llm_action",
    "is_llm_output",
    "is_observation",
    "is_resume_action",
    "is_resumed",
    "is_retryable",
    "is_supervising",
    "is_user_query",
    "parse_node_obj",
    "retry_transient",
    "tool",
    "tui",
]
