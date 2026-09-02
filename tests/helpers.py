"""Shared test doubles for the running-only core."""

from __future__ import annotations

from rlmflow import AgentStart, LLMUsage
from rlmflow.graph.nodes import Node
from rlmflow.runtime.repl import LocalRepl, base_namespace
from rlmflow.runtime.repl_client import _RPC_CALL_ID
from rlmflow.runtime.runtime import Runtime
from rlmflow.tools.agents import AGENT_OBSERVE_TOOL, AGENT_WAIT_TOOL, AGENTS_BINDING


class TestRuntime(Runtime):
    """Fast in-process runtime for tests that do not exercise worker transport."""

    __test__ = False

    def open(self, agent: AgentStart) -> LocalRepl:
        namespace = base_namespace(self.preimports)
        if agent.config.reuse_repl and agent.parent is not None:
            parent = agent.parent.parent_agent
            parent_repl = self.repls.get(parent.id)
            if not isinstance(parent_repl, LocalRepl):
                raise RuntimeError("reuse_repl requires a live parent REPL")
            namespace = parent_repl.namespace
        return LocalRepl(self.working_directory, namespace=namespace)

    async def execute(self, value: AgentStart | Node, code: str):
        repl = self.repl_for(value)
        launch = repl.namespace.get("launch_subagent")
        if launch is not None:
            call_id = 0

            async def launch_with_call_id(*args, **kwargs):
                nonlocal call_id
                token = _RPC_CALL_ID.set(call_id)
                call_id += 1
                try:
                    return await launch(*args, **kwargs)
                finally:
                    _RPC_CALL_ID.reset(token)

            repl.namespace["launch_subagent"] = launch_with_call_id

        binding = {
            "inputs": repl.inputs,
            "env": repl.env,
            "structured_output": repl.structured_output,
        }
        for source, target in (
            (AGENTS_BINDING, "agents"),
            (AGENT_WAIT_TOOL, "wait_agent"),
            (AGENT_OBSERVE_TOOL, "observe_agent"),
        ):
            item = repl.namespace.get(source)
            if item is not None:
                binding[target] = item
        return await repl.run(code, binding=binding)


class StubLLM:
    def __init__(self, fn):
        self.fn = fn

    def chat(self, messages, **_kwargs):
        return self.fn(messages)


class UsageLLM:
    def __init__(self, reply, input_tokens=0, output_tokens=0):
        self.reply = reply
        self.last_usage = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def chat(self, messages, **_kwargs):
        self.last_usage = LLMUsage(self.input_tokens, self.output_tokens)
        return self.reply(messages) if callable(self.reply) else self.reply


def first_user(messages):
    return next(message["content"] for message in messages if message["role"] == "user")


def counting_replies(*replies):
    state = {"index": 0}

    def reply(_messages):
        index = min(state["index"], len(replies) - 1)
        state["index"] += 1
        return replies[index]

    return reply
