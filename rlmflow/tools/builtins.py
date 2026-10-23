"""Flow-owned tools, built as ordinary node-bound ``@tool`` callables."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import queue
import threading
from copy import deepcopy
from typing import Any, NoReturn

from rlmflow.graph.config import can_spawn
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    Node,
    TurnMode,
    VerificationFailure,
    active_step,
    current_step,
)
from rlmflow.graph.reads import ask_terms
from rlmflow.runtime.repl import DoneSignal, VerificationSignal
from rlmflow.runtime.repl_server import schema_hash as verifier_schema_hash
from rlmflow.structured import as_finish_value
from rlmflow.tools.launch import fork_tool, launch_tool
from rlmflow.tools.messaging import peer_tools
from rlmflow.tools.namespace import tool
from rlmflow.utils import sampling_kwargs


def submit_candidate(
    flow: Any,
    agent: AgentStart,
    answer: object,
    *,
    schema: dict[str, Any] | None,
) -> NoReturn:
    """Validate one candidate against ``schema`` and raise the completion signal."""
    schema_hash = verifier_schema_hash(schema)
    try:
        candidate = as_finish_value(answer)
        value = as_finish_value(
            flow.output_codec.validate(candidate, schema) if schema is not None else candidate
        )
    except Exception as exc:  # noqa: BLE001 - codec errors become repair feedback
        raise VerificationSignal(
            repr(answer),
            kind="schema",
            feedback=str(exc),
            schema_hash=schema_hash,
            verifier_hash=agent.config.verifier_hash,
        ) from None
    feedback = run_verifier(flow, agent, value)
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


def node_for_candidate(
    flow: Any,
    agent: AgentStart,
    answer: object,
    *,
    observations: str = "",
) -> Node:
    """Turn an external candidate into the same durable nodes as ``submit``."""
    try:
        submit_candidate(flow, agent, answer, schema=agent.config.output_schema)
    except VerificationSignal as failure:
        return VerificationFailure(
            content=failure.feedback,
            candidate=failure.candidate,
            kind=failure.kind,
            feedback=failure.feedback,
            schema_hash=failure.schema_hash,
            verifier_hash=failure.verifier_hash,
            observations=observations,
        )
    except DoneSignal as done:
        acceptance = done.acceptance
        return DoneOutput(
            content=observations,
            result=done.answer,
            schema_hash=acceptance.get("schema_hash"),
            verifier_hash=acceptance.get("verifier_hash"),
            verified=bool(acceptance.get("verified", False)),
        )
    raise AssertionError("submit_candidate returned without a completion signal")


def run_verifier(flow: Any, agent: Any, candidate: object) -> str | None:
    """Route a submitted candidate to the verifier owned by its launching agent."""
    verifier_id = agent.config.verifier_id
    if verifier_id is None:
        return None
    verify = flow._verifiers.get(verifier_id)
    if verify is not None:
        result = verify(candidate)
        if inspect.isawaitable(result):
            result = _run_async_verifier(result)
    else:
        from rlmflow.graph import reads
        from rlmflow.graph.nodes import current_step

        step = current_step()
        owner = reads.launcher(step.graph, agent) if step is not None else None
        repl = flow.runtime.get(owner) if owner is not None else None
        worker_verify = getattr(repl, "verify", None)
        if worker_verify is None:
            raise RuntimeError(f"verifier {verifier_id!r} is unavailable")
        result = worker_verify(verifier_id, candidate)
    if result is not None and not isinstance(result, str):
        raise TypeError("verify must return str or None")
    return result


def _run_async_verifier(awaitable: Any) -> Any:
    """Run an async verifier behind the synchronous ``submit`` API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            results.put((True, asyncio.run(awaitable)))
        except BaseException as exc:  # noqa: BLE001 - cross-thread verifier result
            results.put((False, exc))

    thread = threading.Thread(target=run, daemon=True, name="rlmflow-verifier")
    thread.start()
    ok, value = results.get()
    thread.join()
    if not ok:
        raise value
    return value


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
    cache: bool | None = None,
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
    resolved_model = selected_model or "default"
    normalized_schema = flow.output_codec.normalize(schema) if schema is not None else None
    key_data = {
        "model": resolved_model,
        "prompt": prompt,
        "schema": normalized_schema,
        "sampling": llm_kwargs,
    }
    encoded = json.dumps(key_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    cache_key = hashlib.sha256(encoded.encode()).hexdigest()
    enabled = flow.llm_query_cache if cache is None else cache

    async def produce() -> dict[str, Any]:
        if schema is not None:
            result = await flow.structured_query(
                prompt,
                output_schema=schema,
                model=selected_model,
                **llm_kwargs,
            )
            return {"value": result.value, "metadata": result.metadata}
        reply, usage = await flow.call_chat(
            [{"role": "user", "content": prompt}],
            resolved_model,
            **llm_kwargs,
        )
        return {
            "value": reply,
            "metadata": {
                "model": resolved_model,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                },
            },
        }

    hit = False
    if enabled:
        entry = flow._llm_query_cache.get(cache_key)
        if entry is None:
            entry = asyncio.create_task(produce())
            flow._llm_query_cache[cache_key] = entry
        else:
            hit = True
        try:
            payload = await entry if isinstance(entry, asyncio.Task) else entry
        except BaseException:
            if flow._llm_query_cache.get(cache_key) is entry:
                flow._llm_query_cache.pop(cache_key, None)
            raise
        flow._llm_query_cache[cache_key] = deepcopy(payload)
    else:
        payload = await produce()

    step = active_step()
    if step is not None:
        step.annotate(
            "llm_query.calls",
            {
                "cache_key": cache_key,
                "cache_hit": hit,
                "model": resolved_model,
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "structured": schema is not None,
                **payload.get("metadata", {}),
            },
            append=True,
        )
    return deepcopy(payload["value"])


def llm_query_tool(flow: Any, node: Node):
    """Build the one-shot query tool for one agent."""
    default_model = current_step().agent.config.model
    description = (
        "Make one model completion for extraction, summarization, or Q&A over the "
        "text you pass. The call sees no REPL, history, or tools, so use it for a "
        "one-shot transform, not multi-step owned work." + _model_help(flow, default_model)
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
        cache: bool | None = None,
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
            cache=cache,
        )

    return llm_query


def submit_tool(flow: Any, node: Node):
    step = current_step()
    agent = step.agent
    terms = ask_terms(step.graph, node) or {}
    schema = terms.get("schema", agent.config.output_schema)

    @tool(
        "Submit this agent's candidate answer as a string or strict JSON value. "
        "A valid candidate completes the agent.",
        proxy=True,
    )
    def submit(answer: object) -> None:
        submit_candidate(flow, agent, answer, schema=schema)

    return submit


def builtin_tools(flow: Any, node: Node) -> dict[str, Any]:
    """Return Flow's ordinary tools for one node."""
    if node.turn_mode is TurnMode.FINAL:
        return {"submit": submit_tool(flow, node)}
    tools: dict[str, Any] = {"submit": submit_tool(flow, node)}
    step = current_step()
    if can_spawn(step.graph, step.agent):
        tools["launch_agent"] = launch_tool(flow, node)
        tools["fork_agent"] = fork_tool(flow, node)
    if flow.use_agent_tree:
        tools.update(peer_tools(flow, node))
    if flow.use_llm_query:
        tools["llm_query"] = llm_query_tool(flow, node)
    return tools


__all__ = [
    "builtin_tools",
    "llm_query_tool",
    "node_for_candidate",
    "query",
    "run_verifier",
    "submit_candidate",
    "submit_tool",
]
