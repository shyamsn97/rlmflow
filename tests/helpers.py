"""Shared test doubles for the running-only core."""

from __future__ import annotations

import inspect

from rlmflow import AgentConfig, Node
from rlmflow.graph import NEXT, Graph
from rlmflow.graph.nodes import AGENT_START, Step, current_code, running_step
from rlmflow.runtime.repl import (
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    LocalRepl,
    base_namespace,
    get_tool_metadata,
)
from rlmflow.runtime.repl_client import invoke_tool_call
from rlmflow.runtime.runtime import RLMFLOW_REPLAY, Runtime


def agent_graph(task: str = "", *, args: dict | None = None) -> tuple[Graph, Node]:
    """A fresh graph holding one root, bound to the default config as a Flow would before stepping it.

    A Flow rebinds the root to its run's ``root_config``; pass ``root_config=`` to
    ``flow.run`` for anything else.
    """
    g = Graph()
    root = g.create_agent(task, **(args or {}))
    root.bind_config(AgentConfig())
    return g, root


def run_result(flow, g: Graph, agent: Node, **kwargs):
    """Run the graph to quiescence and return ``agent``'s result."""
    flow.run(g, **kwargs)
    return g.result(agent)


def stepping(g: Graph, agent: Node, node=None):
    """Run engine code as if inside ``agent``'s step at ``node`` (its frontier by default)."""
    return running_step(Step(g, agent, node if node is not None else g.frontier(agent)))


def next_node(g: Graph, node):
    """The node its own agent appended after ``node``, if any."""
    for edge in g.edges(kind=NEXT, source=node.id):
        target = g.node(edge.target)
        if target.agent_id == node.agent_id and target.type != AGENT_START:
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

    def open(self, agent: Node) -> LocalRepl:
        return LocalRepl(self.directory(agent), namespace=base_namespace(self.preimports))

    async def execute(self, agent: Node, code: str, graph: Graph | None = None):
        repl = self.repl_for(agent, graph)
        cell = current_code()
        restoring = repl.env.get(RLMFLOW_REPLAY) == "1"
        counts: dict[str, int] = {}

        for name, original in tuple(repl.namespace.items()):
            metadata = get_tool_metadata(original) if callable(original) else None
            if metadata is None or not metadata.proxy:
                continue

            def invoke(*args, _fn=original, _name=name, **kwargs):
                call_id = counts.get(_name, 0)
                counts[_name] = call_id + 1
                return invoke_tool_call(_fn, cell, call_id, restoring, args, kwargs, name=_name)

            if inspect.iscoroutinefunction(original):

                async def async_invoke(*args, _invoke=invoke, **kwargs):
                    result = _invoke(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                repl.namespace[name] = async_invoke
            else:
                repl.namespace[name] = invoke

        binding = {"args": repl.args, "env": repl.env}
        for source, target in ((AGENTS_BINDING, "agents"), (AGENT_WAIT_TOOL, "wait_agent")):
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
