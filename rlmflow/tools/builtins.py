"""Reserved REPL builtins as a prompt-owning toolset.

``finish`` and ``launch_subagent`` are per-node closures Flow already injects.
The one-shot query tools are bound only when their corresponding Flow flags are
enabled. This class is the prompt source for those names, not their implementation:
every ``@tool`` method is ``inject=False``.
"""

from __future__ import annotations

import textwrap
from typing import Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import TurnMode
from rlmflow.tools.agents import AgentHandle
from rlmflow.tools.tools import tool, toolset

AGENTS_TEXT = """\
`AGENTS` is a read-only snapshot of this run's recursive agent tree:

- `AGENTS.get()` returns you; `AGENTS.get(id_or_path_or_name)` finds another agent.
- `AGENTS.get_parent()`, `.get_siblings()`, and `.get_children()` return
  `AgentInfo` objects relative to you (or pass an agent selector).
- `AgentInfo.status` is `blocked`, `running`, `waiting`, `idle`, or `completed`;
  `.dependency_paths`, `.dependent_paths`, and `.blocked_on_paths` expose dataflow.
  `.result()` returns an already-completed result and raises while one is still
  in flight; `await agent.wait_for_result()` waits automatically.
- `AGENTS.print_graph(show_results=True)` prints the whole tree with statuses and
  bounded result previews. Pass `view="dependencies"` for the consumer-rooted
  dependency projection.

The snapshot is refreshed before each REPL action. It does not wait, message,
cancel, or steer agents, and repeated queries within one action are not fresher.
"""

STRUCTURED_OUTPUT_OPTION_TEXT = """\
Child results are JSON-compatible Python values by default. Pass `output_schema`
only when a validated typed result is needed; use explicit `properties`,
`required`, and `additionalProperties: false`. The launcher's `[handoff]`
UserQuery shows that same value; an object is listed as fields.
"""


def _agent(node: Any) -> Any:
    if node is None:
        return None
    parent = getattr(node, "parent_agent", None)
    if parent is not None:
        return parent
    return node if getattr(node, "config", None) is not None else None


def _repl(code: str) -> str:
    return f"```repl\n{textwrap.dedent(code).strip()}\n```"


def _use_llm_query(flow: Any) -> bool:
    return flow is None or bool(getattr(flow, "use_llm_query", False))


def _use_llm_query_batched(flow: Any) -> bool:
    if not _use_llm_query(flow):
        return False
    return flow is None or bool(getattr(flow, "use_llm_query_batched", False))


def _delegated_model(flow: Any) -> str:
    name = getattr(flow, "delegation_model", None) if flow is not None else None
    return name or "default"


@toolset("REPL and Delegation")
class BuiltIns:
    @tool(
        "Submit the final answer as a string or strict JSON value and end the run immediately.",
        inject=False,
    )
    def finish(self, answer: object) -> None:
        raise RuntimeError("finish is bound by Flow, not by BuiltIns")

    @tool(
        "Recursive RLM sub-call with its own persistent REPL. The parent only pays "
        "the small result. Pass a child handle as an `inputs` value to start only "
        "after that child completes and inject its result under the chosen key.",
        inject=False,
    )
    async def launch_subagent(
        self,
        goal: str,
        *,
        model: str,
        name: str | None = None,
        inputs: dict[str, str | AgentHandle] | None = None,
        output_schema: object | None = None,
        prompt_profile: str | None = None,
        reuse_repl: bool = False,
    ) -> AgentHandle:
        raise RuntimeError("launch_subagent is bound by Flow, not by BuiltIns")

    @tool(
        "A single sub-LLM completion (no REPL, no iteration). Use for extraction, "
        "summarization, or Q&A over a chunk of text.",
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
        "Concurrently call several LLM calls in parallel over a list of prompts; "
        "same order out as in.",
        inject=False,
    )
    async def llm_query_batched(
        self,
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: object | None = None,
    ) -> list:
        raise RuntimeError(
            "llm_query_batched is bound by Flow(use_llm_query=True, use_llm_query_batched=True)"
        )

    @tool("List every variable currently in the REPL.", inject=False, name="SHOW_VARS")
    def show_vars(self) -> str:
        raise RuntimeError("SHOW_VARS is bound by the runtime")

    def visible_tools(self, flow: Any, node: Any) -> set[str]:
        """Names whose signatures belong in this node's prompt."""
        if getattr(node, "turn_mode", None) is TurnMode.FINAL:
            return {"finish"}
        names = {"SHOW_VARS", "finish"}
        agent = _agent(node)
        if flow is None or can_spawn(agent):
            names.add("launch_subagent")
        if _use_llm_query(flow):
            names.add("llm_query")
        if _use_llm_query_batched(flow):
            names.add("llm_query_batched")
        return names

    def guidance(self, flow: Any, node: Any) -> str:
        """Context-sensitive usage guidance for the visible builtin tools."""
        if getattr(node, "turn_mode", None) is TurnMode.FINAL:
            return ""
        agent = _agent(node)
        parts: list[str] = []

        local = [
            "Example — one coherent task stays local:",
            _repl(
                'print({name: value[:500] for name, value in INPUTS.items()} or "(empty INPUTS)")'
            ),
        ]
        parts.append("\n".join(local))

        if flow is None or can_spawn(agent):
            model = _delegated_model(flow)
            launch = [
                "Example — launch ready independent work together:",
                _repl(
                    f"""\
                    import asyncio
                    first, second = await asyncio.gather(
                        launch_subagent("Produce deliverable A.", model="{model}", name="a"),
                        launch_subagent("Produce deliverable B.", model="{model}", name="b"),
                    )
                    print(await asyncio.gather(first, second))
                    """
                ),
                "Example — declare a known dependency without waiting first:",
                _repl(
                    f"""\
                    source = await launch_subagent(
                        "Produce the source artifact.", model="{model}", name="source"
                    )
                    review = await launch_subagent(
                        "Validate INPUTS['artifact'].", model="{model}", name="review",
                        inputs={{"artifact": source}},
                    )
                    print(await review)
                    """
                ),
            ]
            parts.append("\n".join(launch))

        if _use_llm_query(flow):
            query = "Use `llm_query` for a one-shot transform, not multi-step owned work."
            if _use_llm_query_batched(flow):
                query += " Use `llm_query_batched` for many independent one-shot transforms."
            parts.append(query)

        if flow is not None and getattr(flow, "use_agent_tree", False):
            parts.append(AGENTS_TEXT.strip())

        if (
            flow is not None
            and getattr(flow, "enable_structured_output", False)
            and can_spawn(agent)
        ):
            parts.append(STRUCTURED_OUTPUT_OPTION_TEXT.strip())

        return "\n\n".join(part for part in parts if part)


__all__ = ["AGENTS_TEXT", "BuiltIns", "STRUCTURED_OUTPUT_OPTION_TEXT"]
