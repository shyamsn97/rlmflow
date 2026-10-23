"""Anthropic chat client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm import LLMChunk, LLMClient, join_chunks, stream_with_retry


class AnthropicClient(LLMClient):
    """Anthropic client. Requires `pip install anthropic`."""

    thread_safe = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        *,
        client: Any = None,
        **client_kwargs,
    ) -> None:
        if client is None:
            import anthropic

            # Async SDK: see OpenAIClient — running on the event loop is what lets
            # Flow's ``asyncio.wait_for`` truly cancel a stuck call at the timeout.
            client = anthropic.AsyncAnthropic(**client_kwargs)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def split_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        return system, chat_msgs

    async def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = await self.completion(messages, *args, **kwargs)
        return text

    async def aclose(self) -> None:
        await self.client.close()

    def request_kwargs(self, kwargs: dict) -> dict:
        """Per-request fields the streamed create sends."""
        request_kwargs: dict[str, Any] = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("timeout") is not None:
            request_kwargs["timeout"] = kwargs["timeout"]
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]
        if kwargs.get("stop") is not None:
            request_kwargs["stop_sequences"] = kwargs["stop"]
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
        system, chat_msgs = self.split_messages(messages)
        async with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=chat_msgs,
            **self.request_kwargs(kwargs),
        ) as s:
            async for text in s.text_stream:
                if text:
                    yield LLMChunk(text=text)
            msg = await s.get_final_message()
            usage = LLMUsage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
            )
            self.last_usage = usage
            yield LLMChunk(usage=usage)


__all__ = ["AnthropicClient"]
