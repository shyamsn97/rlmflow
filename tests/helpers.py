"""Shared test doubles for the running-only core."""

from __future__ import annotations

import inspect

from rlmflow import AgentStart
from rlmflow.graph.nodes import ExecAction, Node
from rlmflow.runtime.env import RLMFLOW_REPLAY
from rlmflow.runtime.repl import LocalRepl, base_namespace
from rlmflow.runtime.repl_client import invoke_tool_call
from rlmflow.runtime.runtime import Runtime
from rlmflow.tools.agents import AGENT_OBSERVE_TOOL, AGENT_WAIT_TOOL, AGENTS_BINDING
from rlmflow.tools.tools import get_tool_metadata


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

    def open(self, agent: AgentStart) -> LocalRepl:
        namespace = base_namespace(self.preimports)
        if agent.config.reuse_worker and agent.parent is not None:
            parent = agent.parent.parent_agent
            if not isinstance(self.repls.get(parent.id), LocalRepl):
                raise RuntimeError("reuse_worker requires a live parent worker")
        return LocalRepl(self.working_directory, namespace=namespace)

    async def execute(self, value: AgentStart | Node, code: str):
        repl = self.repl_for(value)
        if not isinstance(value, ExecAction):
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
                    value,
                    call_id,
                    restoring,
                    args,
                    kwargs,
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
            (AGENT_OBSERVE_TOOL, "observe_agent"),
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
