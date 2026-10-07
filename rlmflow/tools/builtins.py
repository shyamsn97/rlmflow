"""Flow-owned tools, built as ordinary node-bound ``@tool`` callables."""

from __future__ import annotations

from typing import Any

from rlmflow.engine import verification
from rlmflow.engine.delegation import launch_tool
from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import Node, TurnMode, turn_mode_for
from rlmflow.runtime.repl import DoneSignal, VerificationSignal
from rlmflow.structured import as_finish_value
from rlmflow.tools.tools import tool
from rlmflow.utils import sampling_kwargs


def _model_help(flow: Any, default_model: str) -> str:
    names = ", ".join(f"`{key}`" for key in sorted(flow._llm_clients))
    choices = f" Available `model=` values: {names}." if names else ""
    return f"{choices} Omitting `model` uses `{default_model}`."


async def query(
    flow: Any,
    prompt: str,
    *,
    model: str | None = None,
    default_model: str | None = None,
    schema: Any = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> Any:
    """Execute one context-free model call for host or tool use."""
    if not isinstance(prompt, str):
        raise TypeError("llm_query(prompt) takes a str")
    llm_kwargs = sampling_kwargs(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )
    selected_model = model if model is not None else default_model
    if schema is not None:
        result = await flow.structured_query(
            prompt,
            output_schema=schema,
            model=selected_model,
            **llm_kwargs,
        )
        return result.value
    reply, _usage = await flow.call_chat(
        [{"role": "user", "content": prompt}],
        selected_model or "default",
        **llm_kwargs,
    )
    return reply


def llm_query_tool(flow: Any, node: Node):
    """Build the one-shot query tool for one agent."""
    default_model = node.parent_agent.config.model
    description = (
        "Make one model completion for extraction, summarization, or Q&A over the "
        "text you pass. The call sees no REPL, history, or tools, so use it for a "
        "one-shot transform, not multi-step owned work."
        + _model_help(flow, default_model)
    )

    @tool(description, proxy=True, restore="result")
    async def llm_query(
        prompt: str,
        *,
        model: str | None = None,
        schema: Any = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Any:
        return await query(
            flow,
            prompt,
            model=model,
            default_model=default_model,
            schema=schema,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

    return llm_query


def submit_tool(flow: Any, node: Node):
    agent = node.parent_agent
    schema = agent.config.output_schema
    schema_hash = verification.schema_hash(schema)

    @tool(
        "Submit this agent's candidate answer as a string or strict JSON value. "
        "A locally valid candidate is recorded durably before the agent completes.",
        proxy=True,
    )
    def submit(answer: object) -> None:
        try:
            candidate = as_finish_value(answer)
            value = as_finish_value(
                flow.output_codec.validate(candidate, schema)
                if schema is not None
                else candidate
            )
        except Exception as exc:  # noqa: BLE001 - codec errors become repair feedback
            raise VerificationSignal(
                repr(answer),
                kind="schema",
                feedback=str(exc),
                schema_hash=schema_hash,
                verifier_hash=agent.config.verifier_hash,
            ) from None
        feedback = verification.check(flow, agent, value)
        if feedback is not None:
            raise VerificationSignal(
                value,
                kind="verifier",
                feedback=feedback,
                schema_hash=schema_hash,
                verifier_hash=agent.config.verifier_hash,
            )
        raise DoneSignal(
            value,
            acceptance={
                "schema_hash": schema_hash,
                "verifier_hash": agent.config.verifier_hash,
                "verified": agent.config.verifier_id is not None,
            },
        )

    return submit


def builtin_tools(flow: Any, node: Node) -> dict[str, Any]:
    """Return Flow's ordinary tools for one node."""
    mode = turn_mode_for(node)
    if mode is TurnMode.FINAL:
        return {"submit": submit_tool(flow, node)}
    tools: dict[str, Any] = {"submit": submit_tool(flow, node)}
    if can_spawn(node.parent_agent):
        tools["launch_subagent"] = launch_tool(flow, node)
    if flow.use_llm_query:
        tools["llm_query"] = llm_query_tool(flow, node)
    return tools


__all__ = [
    "builtin_tools",
    "llm_query_tool",
    "query",
    "submit_tool",
]
