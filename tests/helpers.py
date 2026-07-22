"""Shared stubs and fixtures for the rlmflow test suite."""

from __future__ import annotations

import asyncio

from rlmflow import ExecAction, ExecOutput, Flow, Graph, LLMOutput, LLMUsage, SupervisingOutput


class StubLLM:
    def __init__(self, fn):
        self.fn = fn

    def chat(self, messages):
        return self.fn(messages)


class UsageLLM:
    def __init__(self, reply, input_tokens=0, output_tokens=0):
        self.reply = reply
        self.last_usage = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def chat(self, messages):
        self.last_usage = LLMUsage(self.input_tokens, self.output_tokens)
        return self.reply(messages) if callable(self.reply) else self.reply


def first_user(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


def assert_llm_outputs_are_followed_by_exec_actions(graph):
    for agent in graph.walk():
        for index, node in enumerate(agent.nodes):
            if node.type == "llm_output":
                assert index + 1 < len(agent.nodes)
                assert agent.nodes[index + 1].type == "exec_action"


def seed_exec_graph(*code_blocks):
    """A graph shaped like a real trajectory: (llm_output, exec_action, exec_output)*."""
    graph = Graph(query="q")
    for code in code_blocks:
        graph.commit(LLMOutput(content="turn", code=code))
        graph.commit(ExecAction(code=code))
        graph.commit(ExecOutput(content="", output=""))
    return graph


async def worker_step(flow, graph, code):
    """Simulate one exec turn on a branch: record the action, then run it."""
    graph.commit(ExecAction(code=code))
    await flow.exec_turn(graph, code)


def counting_replies(*replies):
    """StubLLM callable returning ``replies`` in order, repeating the last."""
    state = {"n": 0}

    def reply(_messages):
        index = min(state["n"], len(replies) - 1)
        state["n"] += 1
        return replies[index]

    return reply


# Canonical deep tree for graph-surgery tests (see docs/internal/deep_tree_graph_ops_test_plan.md):
#
#   root
#   ├── root.m0
#   │   ├── root.m0.l0
#   │   └── root.m0.l1
#   └── root.m1
#       └── root.m1.l0
DEEP_TREE_AGENT_IDS = (
    "root",
    "root.m0",
    "root.m0.l0",
    "root.m0.l1",
    "root.m1",
    "root.m1.l0",
)


def deep_tree_reply(messages):
    """StubLLM replies that build :data:`DEEP_TREE_AGENT_IDS` via launch_subagents."""
    task = first_user(messages)
    if task == "root":
        return (
            "```repl\n"
            "root_var = 'root'\n"
            "results = await launch_subagents(["
            "{'name': 'm0', 'query': 'mid0'}, "
            "{'name': 'm1', 'query': 'mid1'}])\n"
            "done('|'.join(results))\n"
            "```"
        )
    if task == "mid0":
        return (
            "```repl\n"
            "mid_var = 'm0'\n"
            "results = await launch_subagents(["
            "{'name': 'l0', 'query': 'leaf'}, "
            "{'name': 'l1', 'query': 'leaf'}])\n"
            "done('+'.join(results))\n"
            "```"
        )
    if task == "mid1":
        return (
            "```repl\n"
            "mid_var = 'm1'\n"
            "results = await launch_subagents(["
            "{'name': 'l0', 'query': 'leaf'}])\n"
            "done(results[0])\n"
            "```"
        )
    return (
        "```repl\n"
        "leaf_var = ENV['RLMFLOW_AGENT_ID']\n"
        "done('leaf')\n"
        "```"
    )


def make_deep_flow(*, max_depth: int = 3, **kwargs) -> Flow:
    """Flow wired with :func:`deep_tree_reply` (StubLLM)."""
    return Flow(StubLLM(deep_tree_reply), max_depth=max_depth, **kwargs)


async def build_deep_tree(flow: Flow | None = None) -> tuple[Flow, Graph]:
    """Run a root→mid→leaf tree through ``launch_subagents``; return ``(flow, root)``.

    Each agent sets a distinct REPL var (``root_var`` / ``mid_var`` / ``leaf_var``)
    so fork/merge isolation is observable.
    """
    flow = flow or make_deep_flow()
    root = Graph(query="root")
    await flow.arun(graph=root)
    return flow, root


def first_supervising(agent: Graph) -> SupervisingOutput:
    """First ``SupervisingOutput`` on ``agent`` (the launch that spawned children)."""
    for node in agent.nodes:
        if isinstance(node, SupervisingOutput):
            return node
    raise AssertionError(f"no SupervisingOutput on {agent.agent_id}")


def run_deep_tree(flow: Flow | None = None) -> tuple[Flow, Graph]:
    """Sync wrapper around :func:`build_deep_tree`."""
    return asyncio.run(build_deep_tree(flow))


# Depth-1 tree: root → root.a (fills the matrix's "Depth-1 child" column).
DEPTH1_AGENT_IDS = ("root", "root.a")


def depth1_reply(messages):
    task = first_user(messages)
    if task == "parent":
        return (
            "```repl\n"
            "root_var = 'root'\n"
            "results = await launch_subagents([{'name': 'a', 'query': 'child'}])\n"
            "done(results[0])\n"
            "```"
        )
    return (
        "```repl\n"
        "child_var = 'a'\n"
        "done('child')\n"
        "```"
    )


def make_depth1_flow(**kwargs) -> Flow:
    return Flow(StubLLM(depth1_reply), max_depth=1, **kwargs)


async def build_depth1_tree(flow: Flow | None = None) -> tuple[Flow, Graph]:
    """Run a root→child tree; return ``(flow, root)``."""
    flow = flow or make_depth1_flow()
    root = Graph(query="parent")
    await flow.arun(graph=root)
    return flow, root


def run_depth1_tree(flow: Flow | None = None) -> tuple[Flow, Graph]:
    return asyncio.run(build_depth1_tree(flow))
