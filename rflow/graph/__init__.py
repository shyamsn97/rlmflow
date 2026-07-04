"""RecursiveFlow graph data model — nodes, the :class:`Graph`, views, actions."""

from rflow.graph.actions import Action, ActionPlan, CallLLM, Exec, Recover
from rflow.graph.events import (
    GraphEdited,
    GraphEvent,
    GraphNodeCommitted,
)
from rflow.graph.graph import (
    ActionNode,
    ChildHandle,
    CodeObservation,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    Node,
    ObservationNode,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    WaitRequest,
    append_message,
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
    new_id,
    parse_node_obj,
)
from rflow.graph.injection import (
    inject,
    inject_output,
    is_action_like,
    resolve_injection_targets,
)
from rflow.graph.replace import (
    replace_last_action,
    replace_last_observation,
    replace_node,
)
from rflow.graph.timeline import retrace_steps
from rflow.graph.truncation import (
    prune_descendants_spawned_after,
    truncate_after,
    truncate_agent,
)
from rflow.graph.views import Edge, EdgesView, NodesView

__all__ = [
    # actions
    "Action",
    "ActionPlan",
    "CallLLM",
    "Exec",
    "Recover",
    # handles
    "ChildHandle",
    "WaitRequest",
    # events
    "GraphEdited",
    "GraphEvent",
    "GraphNodeCommitted",
    # graph + helpers
    "Graph",
    "append_message",
    "new_id",
    # node bases
    "Node",
    "ObservationNode",
    "ActionNode",
    "CodeObservation",
    # leaf nodes
    "UserQuery",
    "LLMOutput",
    "ExecOutput",
    "SupervisingOutput",
    "ErrorOutput",
    "DoneOutput",
    "LLMAction",
    "ExecAction",
    "ResumeAction",
    # predicates + parser
    "is_observation",
    "is_action",
    "is_code_observation",
    "is_user_query",
    "is_llm_output",
    "is_exec_output",
    "is_supervising",
    "is_errored",
    "is_done",
    "is_llm_action",
    "is_exec_action",
    "is_resume_action",
    "is_resumed",
    "parse_node_obj",
    # views
    "Edge",
    "NodesView",
    "EdgesView",
    # trajectory editing (pure)
    "inject",
    "inject_output",
    "is_action_like",
    "resolve_injection_targets",
    "replace_node",
    "replace_last_action",
    "replace_last_observation",
    "truncate_after",
    "truncate_agent",
    "prune_descendants_spawned_after",
    "retrace_steps",
]
