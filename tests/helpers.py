"""Shared test doubles for the running-only core."""

from __future__ import annotations

import inspect
import time

from rlmflow import Agent, Flow, Node
from rlmflow.agents import repl_reply
from rlmflow.graph import Graph
from rlmflow.llm import Reply
from rlmflow.repl import Repl, current_repl, running
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


def agent_graph(task: str = "", *, args: dict | None = None) -> tuple[Graph, Node]:
    """A fresh graph holding one root run of the ``default`` agent.

    Pass ``agent=`` to ``run_result`` (or call ``name_root``) to run the root with another agent.
    """
    g = Graph()
    root = g.start(task, **(args or {}))
    return g, root


def name_root(flow, g: Graph, agent: Agent | None) -> None:
    """Register ``agent`` on ``flow`` as ``"root"``, and run the top-level ``default`` runs with it."""
    if agent is None:
        return
    flow.agents["root"] = agent
    for root in g.top_level_roots():
        if root.agent == "default":
            root.agent = "root"


def run_result(flow, g: Graph, run: Node, *, agent: Agent | None = None, hooks=()):
    """Run the graph to quiescence and return ``run``'s result; ``agent`` is the root's."""
    name_root(flow, g, agent)
    flow.run(g, hooks=hooks)
    return run.final_result()


def stepping(g: Graph, run: Node, node=None, flow=None):
    """Run engine code as if inside ``run``'s step at ``node`` (its last node by default)."""
    return running(Repl(flow, run.root, node))


def ran(code: str = "pass", output: str = "", *, result=None, error=False, **data) -> Node:
    """An assistant node whose code already ran; keywords besides ``result`` and ``error`` are data."""
    node = Node("assistant", code=code, output=output, result=result, error=error, data=data)
    node.finished_at = time.time()
    return node


def finish(g: Graph, agent: Node, result=None, **data) -> Node:
    """Append the turn that ends ``agent`` with ``result``."""
    data = {"seen": len(agent.nodes()), "terminal": True, **data}
    return g.append(agent, Node("assistant", result=result, data=data))


def own(agent: Node) -> list[Node]:
    """The agent's own nodes after its root."""
    return agent.nodes()[1:]


def code_nodes(g: Graph, agent: Node) -> list[Node]:
    """The agent's nodes that ran code."""
    return [node for node in agent.nodes() if node.runs_code]


def turns(g: Graph, agent: Node) -> list[Node]:
    """The nodes that ended the agent's turns."""
    return [node for node in agent.nodes() if "seen" in node.data]


def repl(code: str, *, preamble: str | None = None) -> Reply:
    """A reply that runs ``code`` through the ``repl`` tool, after ``preamble`` text."""
    return repl_reply(code, preamble or "")


def reasoning_repl(code: str) -> str:
    """Build the historical flow-test response with a prose preamble."""
    return repl(code, preamble="thinking")


def trailing_user_text(messages) -> str:
    """Join the consecutive user and tool messages at the end of a request."""
    trailing = []
    for message in reversed(messages):
        if message["role"] not in ("user", "tool"):
            break
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part.get("type") == "text")
        trailing.append(content)
    return "\n\n".join(reversed(trailing))


class TestRuntime(Runtime):
    """Fast in-process runtime for tests that do not exercise worker transport."""

    __test__ = False

    def open(self, agent: Node) -> LocalRepl:
        return LocalRepl(self.directory(agent), namespace=base_namespace(self.preimports))

    async def execute(self, agent: Node, code: str, graph: Graph | None = None):
        repl = self.repl_for(agent, graph)
        step = current_repl()

        for name, original in tuple(repl.namespace.items()):
            metadata = get_tool_metadata(original) if callable(original) else None
            if metadata is None or not metadata.proxy:
                continue

            def invoke(*args, _fn=original, _name=name, **kwargs):
                return invoke_tool_call(_fn, step, args, kwargs, name=_name)

            if inspect.iscoroutinefunction(original):

                async def async_invoke(*args, _invoke=invoke, **kwargs):
                    result = _invoke(*args, **kwargs)
                    return await result if inspect.isawaitable(result) else result

                repl.namespace[name] = async_invoke
            else:
                repl.namespace[name] = invoke

        binding = {"env": repl.env}
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
        return self.llms.get(model, self.test_llm)


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
