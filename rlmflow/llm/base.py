"""Chat clients: the protocol, chunks, replies, retries, ``complete`` and ``client_for``.

Messages are OpenAI chat messages, tool calls included: an assistant message's
``tool_calls`` and the ``tool`` messages answering them. Tools are OpenAI
function tools. Other providers' clients translate both.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import inspect
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from rlmflow.graph.nodes import LLMUsage

# Transient HTTP / streaming faults from the OpenAI / Anthropic client
# stack. Matched by class name so this module doesn't have to import
# httpx, httpcore, openai, or anthropic at module load.
RETRYABLE_EXC_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "InternalServerError",
        "RateLimitError",
        "RemoteProtocolError",
        "ConnectError",
        "ReadError",
    }
)
TIMEOUT_EXC_NAMES = frozenset(
    {
        "APITimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
    }
)


def is_retryable(exc: BaseException) -> bool:
    if type(exc).__name__ in TIMEOUT_EXC_NAMES:
        return False
    if type(exc).__name__ in RETRYABLE_EXC_NAMES:
        return True
    cause = exc.__cause__
    if cause is not None and type(cause).__name__ in TIMEOUT_EXC_NAMES:
        return False
    return cause is not None and type(cause).__name__ in RETRYABLE_EXC_NAMES


retry_transient = retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


def _usage_from_client(client: Any) -> LLMUsage:
    usage = getattr(client, "last_usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(getattr(usage, "reasoning_tokens", 0) or 0),
    )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool the model called: its ``name`` and ``arguments`` (a JSON object, as text)."""

    name: str
    arguments: str = "{}"
    id: str = ""


@dataclass(frozen=True, slots=True)
class LLMChunk:
    """One delta from ``LLMClient.stream``.

    ``text`` is visible content (empty on a usage-only trailing chunk).
    ``usage`` is set when the provider reports it, typically once at the end.
    ``calls`` are whole tool calls, sent once they are complete.
    """

    text: str = ""
    usage: LLMUsage | None = None
    calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class Reply:
    """One whole model reply: its text, the tools it called, and what it cost."""

    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = field(default_factory=LLMUsage)


def as_chunk(item: object) -> LLMChunk:
    """Normalize a stream item to ``LLMChunk`` (plain strings and replies still work)."""
    if isinstance(item, LLMChunk):
        return item
    if isinstance(item, Reply):
        return LLMChunk(item.text, item.usage, item.calls)
    if item is None:
        return LLMChunk()
    return LLMChunk(text=str(item))


def as_reply(value: object, usage: LLMUsage | None = None) -> Reply:
    """Normalize what a client returned (a ``Reply``, ``(text, usage)``, or text) to a ``Reply``.

    ``usage`` is what the client reported apart from the reply, for a reply that has none.
    """
    if isinstance(value, Reply):
        if usage is not None and value.usage == LLMUsage():
            return dataclasses.replace(value, usage=usage)
        return value
    if isinstance(value, tuple):
        text, usage = value
        return Reply(text, usage=usage)
    return Reply("" if value is None else str(value), usage=usage or LLMUsage())


def join(chunks: list[LLMChunk]) -> Reply:
    """One reply from its chunks: the text joined, every call, and the last usage."""
    usage = next((c.usage for c in reversed(chunks) if c.usage is not None), LLMUsage())
    return Reply(
        "".join(chunk.text for chunk in chunks),
        tuple(call for chunk in chunks for call in chunk.calls),
        usage,
    )


async def join_chunks(stream: object) -> Reply:
    """A stream's items as one ``Reply``."""
    if inspect.iscoroutine(stream):
        stream = await stream
    if not hasattr(stream, "__aiter__"):
        return join_items(stream)  # type: ignore[arg-type]
    return join([as_chunk(item) async for item in stream])  # type: ignore[union-attr]


def join_items(items: Iterator[object]) -> Reply:
    """``join_chunks`` for a blocking stream."""
    return join([as_chunk(item) for item in items])


async def stream_with_retry(factory) -> AsyncIterator[LLMChunk]:
    """Retry ``factory()`` only before the first text or tool call is sent on."""
    delay = 0.5
    for attempt in range(3):
        emitted = False
        try:
            async for chunk in factory():
                if chunk.text or chunk.calls:
                    emitted = True
                yield chunk
            return
        except Exception as exc:
            if emitted or not is_retryable(exc) or attempt >= 2:
                raise
            await _retry_sleep(delay)
            delay = min(delay * 2, 4)


async def _retry_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


#: Seconds a built-in client's whole call may take before it fails, retries included.
DEFAULT_REQUEST_TIMEOUT = 300.0


class LLMClient(metaclass=abc.ABCMeta):
    last_usage: LLMUsage | None = None
    thread_safe: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[dict[str, Any]], *args, **kwargs) -> str | Reply:
        """Send messages and return the whole reply: its text, or a ``Reply`` that calls tools.

        Production clients implement ``stream`` and join it here. Test fakes
        implement ``chat`` and inherit a one-chunk ``stream``.
        """

    def stream(self, messages: list[dict[str, Any]], *args, **kwargs) -> Iterator[LLMChunk]:
        """Yield response deltas. Override for live token streaming.

        Default falls back to ``chat()`` and yields the whole reply once.
        """
        reply = as_reply(self.chat(messages, *args, **kwargs), self.last_usage)
        yield LLMChunk(reply.text, reply.usage, reply.calls)

    def completion(self, messages: list[dict[str, Any]], *args, **kwargs) -> tuple[str, LLMUsage]:
        """Return response text and usage for one request.

        The default joins a synchronous ``stream``. Async clients override.
        """
        reply = join_items(self.stream(messages, *args, **kwargs))
        return reply.text, reply.usage

    async def aclose(self) -> None:
        """Release any async transport owned by this client."""


async def complete(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    timeout: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Reply:
    """One model call, through the client's ``stream``, ``completion`` or ``chat``.

    A blocking client runs on a thread. ``timeout`` bounds the call, and is
    handed to clients whose method takes one. So are ``tools`` (OpenAI function
    tools): a client that takes none can still return calls.
    """
    fn = getattr(client, "stream", None) or getattr(client, "completion", None) or client.chat
    if timeout is not None and "timeout" not in kwargs and _accepts_kwarg(fn, "timeout"):
        kwargs["timeout"] = timeout
    if tools and _accepts_kwarg(fn, "tools"):
        kwargs["tools"] = tools
    name = getattr(fn, "__name__", "")

    def finish(value: Any) -> Reply:
        return as_reply(value, _usage_from_client(client))

    async def call() -> Reply:
        if inspect.isasyncgenfunction(fn) or (name == "stream" and inspect.iscoroutinefunction(fn)):
            return await join_chunks(fn(messages, **kwargs))
        if inspect.iscoroutinefunction(fn):
            return finish(await fn(messages, **kwargs))

        def blocking() -> Reply:
            value = fn(messages, **kwargs)
            return join_items(value) if name == "stream" else finish(value)

        return await asyncio.to_thread(blocking)

    if timeout is None:
        return await call()
    try:
        return await asyncio.wait_for(call(), timeout)
    except TimeoutError as exc:
        model = getattr(client, "model", None) or type(client).__name__
        raise TimeoutError(f"model call to {model} exceeded {timeout:g}s") from exc


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in parameters)


def client_for(model: str, *, reasoning_effort: str | None = None, **kwargs: Any) -> LLMClient:
    """Return a client for ``model``: Anthropic for ``claude*``, else OpenAI.

    One rule, in one place, so a model name means the same thing to the CLI, the
    examples, and anything else that only knows a string. ``reasoning_effort``
    reaches OpenAI reasoning models only; elsewhere it is dropped rather than
    raised, so callers can pass it without branching on which model they named.
    """
    if model.startswith("claude"):
        from rlmflow.llm.anthropic import AnthropicClient

        return AnthropicClient(model, **kwargs)
    from rlmflow.llm.openai import OpenAIClient

    return OpenAIClient(model, reasoning_effort=reasoning_effort, **kwargs)


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "LLMChunk",
    "LLMClient",
    "Reply",
    "ToolCall",
    "as_chunk",
    "as_reply",
    "client_for",
    "complete",
    "is_retryable",
    "join_chunks",
    "retry_transient",
    "stream_with_retry",
]
