"""OpenAI and OpenAI-compatible clients: the Responses API for OpenAI, Chat Completions elsewhere."""

from __future__ import annotations

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

OPENAI_HOST = "api.openai.com"


def responses_content(content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """OpenAI chat content parts (text, ``image_url``) as Responses input parts."""
    if isinstance(content, str):
        return content
    return [
        {"type": "input_image", "image_url": part["image_url"]["url"]}
        if part.get("type") == "image_url"
        else {"type": "input_text", "text": part["text"]}
        for part in content
    ]


def responses_input(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """OpenAI chat messages as Responses ``instructions`` and input items."""
    instructions = None
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions = message["content"]
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            if message.get("content"):
                items.append({"role": role, "content": responses_content(message["content"])})
            for call in message.get("tool_calls") or ():
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
    return instructions, items


def responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat function tools as Responses function tools."""
    return [{"type": "function", **tool["function"]} for tool in tools]


class OpenAIClient(LLMClient):
    """OpenAI-compatible client. Requires `pip install openai`.

    ``api`` is ``"responses"`` or ``"chat"`` (Chat Completions). By default it is
    the Responses API on OpenAI's own endpoint, where GPT-6 can reason and call
    tools in one request, and Chat Completions on any other ``base_url``.
    """

    thread_safe = True

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        reasoning_effort: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        api: str | None = None,
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
        if api is None:
            api = "responses" if OPENAI_HOST in str(getattr(client, "base_url", "")) else "chat"
        if api not in ("responses", "chat"):
            raise ValueError(f"api must be 'responses' or 'chat', not {api!r}")
        self.client = client
        self.model = model
        self.api = api
        # Sent only when set: a reasoning model otherwise thinks at the provider's
        # default effort, which dominates turn latency for agents whose individual
        # turns are small, and non-reasoning models reject the field outright.
        self.reasoning_effort = reasoning_effort
        # Caps the whole call, SDK retries included; a stalled request otherwise
        # waits out the SDK's 600s timeout on every retry.
        self.request_timeout = request_timeout

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
        responses = self.api == "responses"
        if effort is not None:
            if responses:
                request_kwargs["reasoning"] = {"effort": effort}
            else:
                request_kwargs["reasoning_effort"] = effort
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if kwargs.get(key) is None:
                continue
            if responses and key == "stop":
                raise ValueError("the Responses API has no `stop`; use OpenAIClient(api='chat')")
            request_kwargs["max_output_tokens" if responses and key == "max_tokens" else key] = (
                kwargs[key]
            )
        if kwargs.get("tools"):
            tools = kwargs["tools"]
            request_kwargs["tools"] = responses_tools(tools) if responses else tools
            request_kwargs["parallel_tool_calls"] = False
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
        iterate = self._iter_responses if self.api == "responses" else self._iter_chat
        async for chunk in stream_with_retry(lambda: iterate(messages, kwargs)):
            yield chunk

    async def _iter_responses(
        self, messages: list[dict[str, Any]], kwargs: dict
    ) -> AsyncIterator[LLMChunk]:
        instructions, items = responses_input(messages)
        resp = await self.client.responses.create(
            model=self.model,
            input=items,
            stream=True,
            store=False,
            **({"instructions": instructions} if instructions is not None else {}),
            **self.request_kwargs(kwargs),
        )
        calls: list[ToolCall] = []
        async for event in resp:
            if event.type == "response.output_text.delta":
                yield LLMChunk(text=event.delta)
            elif event.type == "response.output_item.done" and event.item.type == "function_call":
                item = event.item
                calls.append(ToolCall(item.name, item.arguments or "{}", item.call_id))
            elif event.type in ("response.completed", "response.incomplete"):
                raw = event.response.usage
                if raw is not None:
                    details = getattr(raw, "output_tokens_details", None)
                    usage = LLMUsage(
                        input_tokens=raw.input_tokens or 0,
                        output_tokens=raw.output_tokens or 0,
                        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                    )
                    self.last_usage = usage
                    yield LLMChunk(usage=usage)
            elif event.type in ("response.failed", "error"):
                error = getattr(getattr(event, "response", None), "error", None) or event
                raise RuntimeError(f"OpenAI response failed: {getattr(error, 'message', error)}")
        if calls:
            yield LLMChunk(calls=tuple(calls))

    async def _iter_chat(
        self, messages: list[dict[str, str]], kwargs: dict
    ) -> AsyncIterator[LLMChunk]:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs(kwargs),
        )
        calls: dict[int, dict[str, str]] = {}
        async for event in resp:
            text = ""
            if event.choices:
                delta = event.choices[0].delta
                text = getattr(delta, "content", None) or ""
                for part in getattr(delta, "tool_calls", None) or ():
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = part.id or call["id"]
                    if part.function is not None:
                        call["name"] += part.function.name or ""
                        call["arguments"] += part.function.arguments or ""
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
        if calls:
            yield LLMChunk(
                calls=tuple(
                    ToolCall(c["name"], c["arguments"] or "{}", c["id"])
                    for _, c in sorted(calls.items())
                )
            )


__all__ = ["OpenAIClient", "responses_input", "responses_tools"]
