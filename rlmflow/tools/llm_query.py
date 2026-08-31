"""A fan-out LLM tool for agent REPLs (opt in with ``Flow(use_llm_query=True)``)."""

from __future__ import annotations

import asyncio
from typing import Any

from rlmflow.structured import Schema, json_schema_for, parse_structured_output, system_prompt_hint
from rlmflow.tools.tools import tool
from rlmflow.utils import sampling_kwargs


def llm_query(flow: Any):
    """Build the one-shot query tool bound to ``flow``."""

    @tool(
        "Make one model completion for extraction, summarization, or Q&A over "
        "the text you pass. The call sees no REPL, history, or tools.",
        proxy=True,
    )
    async def llm_query(
        prompt: str,
        *,
        model: str = "default",
        output_schema: Schema | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ):
        if not isinstance(prompt, str):
            raise TypeError("llm_query(prompt) takes a str")
        schema = json_schema_for(output_schema) if output_schema is not None else None
        if schema is not None:
            prompt = f"{prompt}\n\nReturn JSON matching this schema:\n{system_prompt_hint(schema)}"
        reply, _usage = await flow.call_chat(
            [{"role": "user", "content": prompt}],
            model,
            **sampling_kwargs(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            ),
        )
        return reply if schema is None else parse_structured_output(reply, schema)

    return llm_query


def llm_query_batched(flow: Any):
    """Build the batched one-shot query tool bound to ``flow``.

    Independent prompts with no trajectory of their own: cheaper than spawning
    subagents when the work needs no REPL, no history, and no delegation.
    """

    @tool(
        "Run a list of independent one-shot prompts as concurrent model calls, "
        "returning their results in prompt order. Each call sees only the text you "
        "pass it: no REPL, no history, no tools. The cheap way to read at volume.",
        proxy=True,
    )
    async def llm_query_batched(
        prompts: list[str],
        *,
        model: str = "default",
        output_schema: Schema | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list:
        if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
            raise TypeError("llm_query_batched(prompts) takes a list[str]")

        schema = json_schema_for(output_schema) if output_schema is not None else None
        if schema is not None:
            hint = system_prompt_hint(schema)
            prompts = [
                f"{prompt}\n\nReturn JSON matching this schema:\n{hint}" for prompt in prompts
            ]
        llm_kwargs = sampling_kwargs(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop
        )

        async def call(prompt: str) -> str:
            reply, _usage = await flow.call_chat(
                [{"role": "user", "content": prompt}], model, **llm_kwargs
            )
            return reply

        replies = await asyncio.gather(*(call(prompt) for prompt in prompts))
        if schema is None:
            return list(replies)
        return [parse_structured_output(reply, schema) for reply in replies]

    return llm_query_batched


__all__ = ["llm_query", "llm_query_batched"]
