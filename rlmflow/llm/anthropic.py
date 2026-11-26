"""Anthropic chat client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm.base import (
    DEFAULT_REQUEST_TIMEOUT,
    LLMChunk,
    LLMClient,
    ToolCall,
    join_chunks,
    stream_with_retry,
)


def anthropic_content(content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """OpenAI-style content parts (text, ``image_url`` data URLs) as Anthropic blocks."""
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "image_url":
            header, data = part["image_url"]["url"].split(",", 1)
            media_type = header.removeprefix("data:").split(";", 1)[0]
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        else:
            blocks.append(part)
    return blocks


def anthropic_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """One OpenAI-style message as Anthropic content blocks, without empty text."""
    if message["role"] == "tool":
        return [
            {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
        ]
    content = anthropic_content(message.get("content") or "")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    blocks = [b for b in blocks if b.get("type") != "text" or b["text"].strip()]
    for call in message.get("tool_calls") or ():
        blocks.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": json.loads(call["function"]["arguments"] or "{}"),
            }
        )
    return blocks


def anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI function tools as Anthropic tools."""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"]["parameters"],
        }
        for tool in tools
    ]


class AnthropicClient(LLMClient):
    """Anthropic client. Requires `pip install anthropic`."""

    thread_safe = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        *,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
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
        self.request_timeout = request_timeout

    def split_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict]]:
        """The system prompt, and the rest as Anthropic messages: tool results are
        user content, and consecutive messages of one role merge."""
        system = ""
        chat_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "user" if m["role"] == "tool" else m["role"]
            blocks = anthropic_blocks(m)
            if not blocks:
                continue
            if chat_msgs and chat_msgs[-1]["role"] == role:
                chat_msgs[-1]["content"].extend(blocks)
            else:
                chat_msgs.append({"role": role, "content": blocks})
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
        if kwargs.get("tools"):
            request_kwargs["tools"] = anthropic_tools(kwargs["tools"])
            request_kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        return request_kwargs

    async def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        reply = await join_chunks(self.stream(messages, *args, **kwargs))
        self.last_usage = reply.usage
        return reply.text, reply.usage

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
            calls = tuple(
                ToolCall(block.name, json.dumps(block.input), block.id)
                for block in msg.content
                if block.type == "tool_use"
            )
            yield LLMChunk(usage=usage, calls=calls)


__all__ = ["AnthropicClient"]
