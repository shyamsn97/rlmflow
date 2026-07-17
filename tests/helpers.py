"""Shared stubs and fixtures for the rflow test suite."""

from rflow import ExecAction, ExecOutput, Graph, LLMOutput, LLMUsage


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
