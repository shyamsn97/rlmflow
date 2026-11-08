"""Shared test doubles for the running-only core."""

from __future__ import annotations

import inspect

from rlmflow import Agent, AgentConfig, Flow, PooledLLMClient
from rlmflow.graph import Graph
from rlmflow.graph.nodes import Step, current_code, running_step
from rlmflow.runtime.repl import (
    AGENT_SEND_TOOL,
    AGENT_WAIT_TOOL,
    AGENTS_BINDING,
    LocalRepl,
    base_namespace,
    get_tool_metadata,
)
from rlmflow.runtime.repl_client import invoke_tool_call
from rlmflow.runtime.runtime import Runtime


def agent_graph(task: str = "", *, args: dict | None = None) -> tuple[Graph, Agent]:
    """A fresh graph holding one root, bound to the default config as a Flow would before stepping it.

    A Flow rebinds the root to its run's ``root_config``; pass ``root_config=`` to
    ``flow.run`` for anything else.
    """
    g = Graph()
    root = g.add_agent(task, **(args or {}))
    root.bind_config(AgentConfig())
    return g, root


def run_result(flow, g: Graph, agent: Agent, **kwargs):
    """Run the graph to quiescence and return ``agent``'s result."""
    flow.run(g, **kwargs)
    return g.result(agent)


def stepping(g: Graph, agent: Agent, node=None):
    """Run engine code as if inside ``agent``'s step at ``node`` (its frontier by default)."""
    return running_step(Step(g, agent, node) if node is not None else Step.latest(g, agent))


def next_node(g: Graph, node):
    """The node its own agent appended after ``node``, if any."""
    nodes = g.owner(node).nodes
    index = nodes.index(node) + 1 if node in nodes else len(nodes)
    return nodes[index] if index < len(nodes) else None


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
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part.get("type") == "text")
        trailing.append(content)
    return "\n\n".join(reversed(trailing))


class TestRuntime(Runtime):
    """Fast in-process runtime for tests that do not exercise worker transport."""

    __test__ = False

    def open(self, agent: Agent) -> LocalRepl:
        return LocalRepl(self.directory(agent), namespace=base_namespace(self.preimports))

    async def execute(self, agent: Agent, code: str, graph: Graph | None = None):
        repl = self.repl_for(agent, graph)
        cell = current_code()

        for name, original in tuple(repl.namespace.items()):
            metadata = get_tool_metadata(original) if callable(original) else None
            if metadata is None or not metadata.proxy:
                continue

            def invoke(*args, _fn=original, _name=name, **kwargs):
                return invoke_tool_call(_fn, cell, args, kwargs, name=_name)

            if inspect.iscoroutinefunction(original):

                async def async_invoke(*args, _invoke=invoke, **kwargs):
                    result = _invoke(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                repl.namespace[name] = async_invoke
            else:
                repl.namespace[name] = invoke

        binding = {"args": repl.args, "env": repl.env}
        for source, target in (
            (AGENTS_BINDING, "agents"),
            (AGENT_WAIT_TOOL, "wait_agent"),
            (AGENT_SEND_TOOL, "send_agent"),
        ):
            item = repl.namespace.get(source)
            if item is not None:
                binding[target] = item
        return await repl.run(code, binding=binding)


class ScriptedFlow(Flow):
    """A Flow whose every model is ``llm``, unless ``llm_clients`` names that model."""

    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.test_llm = llm
        self.llms.setdefault("scripted", llm)

    def llm(self, model):
        client = self.llms.get(model, self.test_llm)
        return PooledLLMClient(client, self.pool, timeout=getattr(client, "request_timeout", None))


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
