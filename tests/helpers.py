"""Shared test doubles for the running-only core."""

from __future__ import annotations

import inspect
from dataclasses import replace

from rlmflow import AgentConfig, AgentStart
from rlmflow.graph import NEXT, Graph, ops, reads
from rlmflow.graph.nodes import ExecAction, Step, active_step, running_step
from rlmflow.runtime.repl import (
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    LocalRepl,
    base_namespace,
    get_tool_metadata,
)
from rlmflow.runtime.repl_client import invoke_tool_call
from rlmflow.runtime.runtime import RLMFLOW_REPLAY, Runtime


def agent_graph(
    task: str = "",
    *,
    config: AgentConfig | None = None,
    args: dict | None = None,
    **overrides,
) -> tuple[Graph, AgentStart]:
    """A fresh graph holding one top-level agent."""
    configured = replace(config or AgentConfig(), **overrides)
    g = Graph()
    return g, ops.spawn(g, task, config=configured, **(args or {}))


def run_result(flow, g: Graph, agent: AgentStart, **kwargs):
    """Run the graph to quiescence and return ``agent``'s result."""
    flow.run(g, **kwargs)
    return reads.result(g, agent)


def stepping(g: Graph, agent: AgentStart, node=None):
    """Run engine code as if inside ``agent``'s step at ``node`` (its frontier by default)."""
    return running_step(Step(g, agent, node if node is not None else reads.frontier(g, agent)))


def next_node(g: Graph, node):
    """The node its own agent appended after ``node``, if any."""
    for edge in g.edges(kind=NEXT, source=node.id):
        target = g.node(edge.target)
        if target.agent_id == node.agent_id and not isinstance(target, AgentStart):
            return target
    return None


def repl(code: str, *, preamble: str | None = None) -> str:
    """Wrap executable test code in the model's strict REPL protocol."""
    prefix = f"{preamble}\n" if preamble else ""
    return f"{prefix}```repl\n{code}\n```"


def reasoning_repl(code: str) -> str:
    """Build the historical flow-test response with a prose preamble."""
    return repl(code, preamble="thinking")


def trailing_user_text(messages) -> str:
    """Join the consecutive user messages at the end of a request."""
    trailing = []
    for message in reversed(messages):
        if message["role"] != "user":
            break
        trailing.append(message["content"])
    return "\n\n".join(reversed(trailing))


class TestRuntime(Runtime):
    """Fast in-process runtime for tests that do not exercise worker transport."""

    __test__ = False

    def open(self, agent: AgentStart, graph: Graph | None = None) -> LocalRepl:
        namespace = base_namespace(self.preimports)
        launcher = reads.launcher(graph, agent) if graph is not None else None
        if agent.config.reuse_worker and launcher is not None:
            if not isinstance(self.repls.get(launcher.id), LocalRepl):
                raise RuntimeError("reuse_worker requires a live parent worker")
        return LocalRepl(self.working_directory, namespace=namespace)

    async def execute(self, agent: AgentStart, code: str, graph: Graph | None = None):
        repl = self.repl_for(agent, graph)
        action = active_step()
        if not isinstance(action, ExecAction):
            raise TypeError("tool execution requires an ExecAction")
        restoring = repl.env.get(RLMFLOW_REPLAY) == "1"
        counts: dict[str, int] = {}

        for name, original in tuple(repl.namespace.items()):
            metadata = get_tool_metadata(original) if callable(original) else None
            if metadata is None or not metadata.proxy:
                continue

            def invoke(*args, _fn=original, _name=name, **kwargs):
                call_id = counts.get(_name, 0)
                counts[_name] = call_id + 1
                return invoke_tool_call(
                    _fn,
                    action,
                    call_id,
                    restoring,
                    args,
                    kwargs,
                    name=_name,
                )

            if inspect.iscoroutinefunction(original):

                async def async_invoke(*args, _invoke=invoke, **kwargs):
                    result = _invoke(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                repl.namespace[name] = async_invoke
            else:
                repl.namespace[name] = invoke

        binding = {
            "args": repl.args,
            "env": repl.env,
        }
        for source, target in (
            (AGENTS_BINDING, "agents"),
            (AGENT_WAIT_TOOL, "wait_agent"),
        ):
            item = repl.namespace.get(source)
            if item is not None:
                binding[target] = item
        return await repl.run(code, binding=binding)


class StubLLM:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.fn(messages)


def first_user(messages):
    return next(message["content"] for message in messages if message["role"] == "user")


def counting_replies(*replies):
    state = {"index": 0}

    def reply(_messages):
        index = min(state["index"], len(replies) - 1)
        state["index"] += 1
        return replies[index]

    return reply
