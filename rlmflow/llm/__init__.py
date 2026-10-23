"""Chat clients: the protocol, chunks, retries, pooling, and ``client_for``."""

from __future__ import annotations

import abc
import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from rlmflow.graph.nodes import LLMUsage

if TYPE_CHECKING:
    from rlmflow.engine.scheduler import Pool

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


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        param.kind is param.VAR_KEYWORD
        or (
            param.name == name
            and param.kind
            in (
                param.POSITIONAL_OR_KEYWORD,
                param.KEYWORD_ONLY,
            )
        )
        for param in params
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
class LLMChunk:
    """One delta from ``LLMClient.stream``.

    ``text`` is visible content (empty on a usage-only trailing chunk).
    ``usage`` is set when the provider reports it, typically once at the end.
    """

    text: str = ""
    usage: LLMUsage | None = None


def as_chunk(item: object) -> LLMChunk:
    """Normalize a stream item to ``LLMChunk`` (plain strings still work)."""
    if isinstance(item, LLMChunk):
        return item
    if item is None:
        return LLMChunk()
    return LLMChunk(text=str(item))


async def join_chunks(stream: object) -> tuple[str, LLMUsage]:
    """Concatenate stream items into ``(text, last usage)``."""
    if inspect.iscoroutine(stream):
        stream = await stream
    parts: list[str] = []
    usage = LLMUsage()
    if hasattr(stream, "__aiter__"):
        async for item in stream:  # type: ignore[union-attr]
            chunk = as_chunk(item)
            parts.append(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
    else:
        for item in stream:  # type: ignore[union-attr]
            chunk = as_chunk(item)
            parts.append(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
    return "".join(parts), usage


async def stream_with_retry(factory) -> AsyncIterator[LLMChunk]:
    """Retry ``factory()`` only before the first non-empty ``text`` delta."""
    delay = 0.5
    for attempt in range(3):
        emitted = False
        try:
            async for chunk in factory():
                if chunk.text:
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


class LLMClient(metaclass=abc.ABCMeta):
    last_usage: LLMUsage | None = None
    thread_safe: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """Send messages and return the full response.

        Production clients implement ``stream`` and join it here. Test fakes
        implement ``chat`` and inherit a one-chunk ``stream``.
        """

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[LLMChunk]:
        """Yield response deltas. Override for live token streaming.

        Default falls back to ``chat()`` and yields the whole reply once.
        """
        text = self.chat(messages, *args, **kwargs)
        yield LLMChunk(text=text, usage=self.last_usage)

    def completion(self, messages: list[dict[str, str]], *args, **kwargs) -> tuple[str, LLMUsage]:
        """Return response text and usage for one request.

        The default joins a synchronous ``stream``. Async clients override.
        """
        parts: list[str] = []
        usage = self.last_usage or LLMUsage()
        for item in self.stream(messages, *args, **kwargs):
            chunk = as_chunk(item)
            parts.append(chunk.text)
            if chunk.usage is not None:
                usage = chunk.usage
        return "".join(parts), usage

    async def aclose(self) -> None:
        """Release any async transport owned by this client."""


class PooledLLMClient:
    """Stream-first LLM primitive backed by a compute pool."""

    def __init__(
        self,
        client: Any,
        pool: Pool,
        *,
        timeout: float | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.pool = pool
        self.timeout = timeout
        self.request_kwargs = dict(request_kwargs or {})

    def _request_kwargs(
        self,
        target: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        request_kwargs = {
            **self.request_kwargs,
            **kwargs,
        }
        if (
            self.timeout is not None
            and "timeout" not in request_kwargs
            and _accepts_kwarg(target, "timeout")
        ):
            request_kwargs["timeout"] = self.timeout
        return request_kwargs

    async def stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[LLMChunk]:
        stream_fn = getattr(self.client, "stream", None)
        if stream_fn is None:
            request_kwargs = self._request_kwargs(self.client.chat, kwargs)
            call = self.pool.call(
                self.client.chat,
                messages,
                **request_kwargs,
            )
            reply = (
                await call if self.timeout is None else await asyncio.wait_for(call, self.timeout)
            )
            yield LLMChunk(
                text=reply,
                usage=_usage_from_client(self.client),
            )
            return

        request_kwargs = self._request_kwargs(stream_fn, kwargs)

        async def iterate() -> AsyncIterator[LLMChunk]:
            async for item in self.pool.stream(
                stream_fn,
                messages,
                **request_kwargs,
            ):
                yield as_chunk(item)

        stream = iterate()
        try:
            if self.timeout is None:
                async for chunk in stream:
                    yield chunk
            else:
                async with asyncio.timeout(self.timeout):
                    async for chunk in stream:
                        yield chunk
        finally:
            await stream.aclose()

    async def completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, LLMUsage]:
        return await join_chunks(self.stream(messages, **kwargs))

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        reply, _usage = await self.completion(
            messages,
            **kwargs,
        )
        return reply

    async def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, LLMUsage]:
        return await self.completion(messages)


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
    "LLMChunk",
    "LLMClient",
    "PooledLLMClient",
    "as_chunk",
    "client_for",
    "is_retryable",
    "join_chunks",
    "retry_transient",
    "stream_with_retry",
]
