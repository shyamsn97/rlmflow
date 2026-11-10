"""OpenAI and OpenAI-compatible chat clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm import LLMChunk, LLMClient, join_chunks, stream_with_retry


class OpenAIClient(LLMClient):
    """OpenAI-compatible client. Requires `pip install openai`."""

    thread_safe = True

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        reasoning_effort: str | None = None,
        client: Any = None,
        **client_kwargs,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            # Async SDK on purpose: the request runs on the event loop, so Flow's
            # ``asyncio.wait_for`` can actually cancel it (closing the socket) when
            # the per-request timeout elapses. A sync client runs on a pool thread
            # that cancellation cannot preempt, so a wedged call would hang past the
            # timeout — the async client is what makes the timeout real.
            client = AsyncOpenAI(**client_kwargs)
        self.client = client
        self.model = model
        # Sent only when set: a reasoning model otherwise thinks at the provider's
        # default effort, which dominates turn latency for agents whose individual
        # turns are small, and non-reasoning models reject the field outright.
        self.reasoning_effort = reasoning_effort

    async def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = await self.completion(messages, *args, **kwargs)
        return text

    async def aclose(self) -> None:
        await self.client.close()

    def request_kwargs(self, kwargs: dict) -> dict:
        """Per-request fields the streamed create sends."""
        request_kwargs = {}
        if kwargs.get("timeout") is not None:
            request_kwargs["timeout"] = kwargs["timeout"]
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort is not None:
            request_kwargs["reasoning_effort"] = effort
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]
        return request_kwargs

    async def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        text, usage = await join_chunks(self.stream(messages, *args, **kwargs))
        self.last_usage = usage
        return text, usage

    async def stream(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> AsyncIterator[LLMChunk]:
        async for chunk in stream_with_retry(lambda: self._iter_response(messages, kwargs)):
            yield chunk

    async def _iter_response(
        self, messages: list[dict[str, str]], kwargs: dict
    ) -> AsyncIterator[LLMChunk]:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs(kwargs),
        )
        async for event in resp:
            text = ""
            if event.choices:
                text = getattr(event.choices[0].delta, "content", None) or ""
            usage = None
            raw = getattr(event, "usage", None)
            if raw is not None:
                details = getattr(raw, "completion_tokens_details", None)
                usage = LLMUsage(
                    input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(raw, "completion_tokens", 0) or 0,
                    reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                )
                self.last_usage = usage
            if text or usage is not None:
                yield LLMChunk(text=text, usage=usage)


__all__ = ["OpenAIClient"]
