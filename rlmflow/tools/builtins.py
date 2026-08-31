"""Reserved REPL builtins as a prompt-owning toolset.

``finish`` and ``launch_subagent`` are per-node closures Flow already injects.
``llm_query_batched`` is a factory bound only when ``use_llm_query=True``. This
class is the prompt source for those names, not their implementation: every
``@tool`` method is ``inject=False``.
"""

from __future__ import annotations

from typing import Any

from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import (
    PromptExample,
    format_tool_line,
    tool,
    toolset,
    toolset_members,
)


def _can_spawn(flow: Any, agent: Any) -> bool:
    if flow is None or agent is None:
        return True
    return agent.config.max_depth > 0 and agent.config.depth < agent.config.max_depth


def _agent(node: Any) -> Any:
    if node is None:
        return None
    parent = getattr(node, "parent_agent", None)
    if parent is not None:
        return parent
    return node if getattr(node, "config", None) is not None else None


@toolset("REPL and Delegation", placement="section")
class BuiltIns:
    @tool(
        "Submit `value` as the final answer and end the run immediately. Put this "
        "call alone in a new block after printing and inspecting the candidate.",
        inject=False,
    )
    def finish(self, value: object) -> None:
        raise RuntimeError("finish is bound by Flow, not by BuiltIns")

    @tool(
        "Launch a recursive RLM subagent with its own persistent REPL and return "
        "a handle immediately.",
        inject=False,
    )
    async def launch_subagent(
        self,
        goal: str,
        model: str,
        name: str | None = None,
        inputs: dict[str, str] | None = None,
        output_schema: object | None = None,
        prompt_profile: str | None = None,
        reuse_repl: bool = False,
    ) -> AgentHandle:
        raise RuntimeError("launch_subagent is bound by Flow, not by BuiltIns")

    @tool(
        "A single sub-LLM completion for extraction, summarization, or Q&A over "
        "the text you pass. It has no REPL or history.",
        inject=False,
    )
    async def llm_query(
        self,
        prompt: str,
        *,
        model: str = "default",
        output_schema: object | None = None,
    ) -> object:
        raise RuntimeError("llm_query is bound by Flow(use_llm_query=True)")

    @tool(
        "Concurrently call several sub-LLMs over a list of prompts; results are "
        "returned in the same order.",
        inject=False,
    )
    async def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: object | None = None,
    ) -> list:
        raise RuntimeError("llm_query_batched is bound by Flow(use_llm_query=True)")

    @tool("List every variable currently in the REPL.", inject=False, name="SHOW_VARS")
    def show_vars(self) -> str:
        raise RuntimeError("SHOW_VARS is bound by the runtime")

    def description(self, flow: Any, node: Any) -> str:
        from rlmflow.prompts.prompts import REPL_TEXT

        enabled = {"finish", "SHOW_VARS"}
        if _can_spawn(flow, _agent(node)):
            enabled.add("launch_subagent")
        if flow is None or getattr(flow, "use_llm_query", False):
            enabled.update({"llm_query", "llm_query_batched"})
        members = dict(toolset_members(self))
        order = ("llm_query", "llm_query_batched", "launch_subagent", "SHOW_VARS", "finish")
        declarations = [format_tool_line(members[name]) for name in order if name in enabled]
        return "\n".join([REPL_TEXT.strip(), *declarations])

    def example(self, flow: Any, node: Any) -> list[PromptExample]:
        from rlmflow.prompts.prompts import DELEGATION_EXAMPLE_TEXT, LOCAL_EXAMPLE_TEXT

        agent = _agent(node)
        examples: list[PromptExample] = []
        # Local first: print-then-submit is the behaviour every run depends on, and
        # when the examples and the prose disagree the examples win. Delegation is
        # the exception case and reads as one.
        examples.append(
            PromptExample(
                priority=10,
                title="Local work — probe, print the candidate, then submit it.",
                body=LOCAL_EXAMPLE_TEXT,
            )
        )
        if _can_spawn(flow, agent) and (agent is None or agent.config.depth == 0):
            model = agent.config.model if agent is not None else "default"
            examples.append(
                PromptExample(
                    priority=20,
                    title="Delegation — a source too large to read here.",
                    body=DELEGATION_EXAMPLE_TEXT.replace("{current_model}", model),
                )
            )
        return examples


__all__ = ["BuiltIns"]
