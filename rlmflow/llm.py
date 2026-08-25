from __future__ import annotations

import abc
import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from rlmflow.engine.execution import Pool

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


@dataclass
class LLMUsage:
    """Token counts for model calls: one call, or a whole subtree summed.

    Lives with the clients because a client is what reports it; ``LLMOutput``
    only carries what it was told.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
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
    )


class PooledLLMClient:
    """Stream-first LLM primitive backed by a compute pool."""

    def __init__(
        self,
        client: Any,
        pool: Pool,
        *,
        timeout: float | None = None,
        key: object | None = None,
        request_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.pool = pool
        self.timeout = timeout
        self.key = key
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
                key=self.key,
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
                key=self.key,
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
                usage = LLMUsage(
                    input_tokens=getattr(raw, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(raw, "completion_tokens", 0) or 0,
                )
                self.last_usage = usage
            if text or usage is not None:
                yield LLMChunk(text=text, usage=usage)


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


class TinkerClient(LLMClient):
    """Tinker sampling client. Requires `pip install tinker tinker-cookbook`.

    Tinker exposes model sampling over token prompts, so this adapter uses a
    Tinker cookbook renderer to convert chat messages to tokens and parse the
    sampled tokens back into assistant text.

    ``sample()`` / ``sample_async()`` return the finished sequence. There is
    no token stream to iterate, so ``stream`` yields that sample once.
    """

    thread_safe = True

    def __init__(
        self,
        *,
        base_model: str | None = "Qwen/Qwen3-8B",
        model_path: str | None = None,
        renderer: str = "qwen3",
        max_tokens: int = 8192,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        service_client=None,
        sampling_client=None,
        renderer_obj=None,
        **service_kwargs,
    ) -> None:
        if sampling_client is None:
            try:
                import tinker  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - exercised by optional deps
                raise ImportError(
                    "TinkerClient requires the optional Tinker SDK. Install it with "
                    "`pip install tinker tinker-cookbook` or `pip install rlmflow[tinker]`."
                ) from exc

            service_client = service_client or tinker.ServiceClient(**service_kwargs)
            sampling_client = service_client.create_sampling_client(
                base_model=base_model,
                model_path=model_path,
            )

        if renderer_obj is None:
            try:
                from tinker_cookbook import renderers  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - exercised by optional deps
                raise ImportError(
                    "TinkerClient requires tinker-cookbook for chat rendering. Install it with "
                    "`pip install tinker-cookbook` or `pip install rlmflow[tinker]`."
                ) from exc
            renderer_obj = renderers.get_renderer(renderer, sampling_client.get_tokenizer())

        self.sampling_client = sampling_client
        self.renderer = renderer_obj
        self.base_model = base_model
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[LLMChunk]:
        """One chunk: Tinker does not emit tokens incrementally."""
        text, usage = self.completion(messages, *args, **kwargs)
        yield LLMChunk(text=text, usage=usage)

    @retry_transient
    def completion(self, messages: list[dict[str, str]], *args, **kwargs) -> tuple[str, LLMUsage]:
        try:
            from tinker import types  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised by optional deps
            raise ImportError(
                "TinkerClient requires the optional Tinker SDK. Install it with "
                "`pip install tinker tinker-cookbook` or `pip install rlmflow[tinker]`."
            ) from exc

        prompt = self.renderer.build_generation_prompt(messages)
        stop = self.stop
        if stop is None and hasattr(self.renderer, "get_stop_sequences"):
            stop = self.renderer.get_stop_sequences()

        params_kwargs = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": kwargs.get("stop", stop),
        }
        params = types.SamplingParams(
            **{key: value for key, value in params_kwargs.items() if value is not None}
        )
        future = self.sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        )
        output = self._future_result(future, timeout=kwargs.get("timeout"))
        tokens = self._first_sequence_tokens(output)
        message = self.renderer.parse_response(tokens)
        text = self._message_text(message)
        usage = LLMUsage(
            input_tokens=self._token_count(prompt),
            output_tokens=self._token_count(tokens),
        )
        self.last_usage = usage
        return text, usage

    @staticmethod
    def _future_result(future, *, timeout: float | None):
        if timeout is None:
            return future.result()
        try:
            return future.result(timeout=timeout)
        except TypeError:
            return future.result()

    @staticmethod
    def _first_sequence_tokens(output) -> object:
        sequences = getattr(output, "sequences", None)
        if sequences is None and isinstance(output, dict):
            sequences = output.get("sequences")
        if isinstance(sequences, (list, tuple)):
            sequence = sequences[0]
        else:
            sequence = sequences
        if isinstance(sequence, dict):
            return sequence.get("tokens", [])
        return getattr(sequence, "tokens", sequence)

    @staticmethod
    def _message_text(parsed) -> str:
        message = parsed[0] if isinstance(parsed, tuple) else parsed
        if isinstance(message, dict):
            return str(message.get("content", ""))
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
        text = getattr(message, "text", None)
        if text is not None:
            return str(text)
        return str(message)

    @staticmethod
    def _token_count(value: object) -> int:
        tokens = getattr(value, "tokens", None)
        if tokens is not None:
            return len(tokens)
        if isinstance(value, dict) and "tokens" in value:
            return len(value["tokens"])
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return 0


def client_for(model: str, *, reasoning_effort: str | None = None, **kwargs: Any) -> LLMClient:
    """Return a client for ``model``: Anthropic for ``claude*``, else OpenAI.

    One rule, in one place, so a model name means the same thing to the CLI, the
    examples, and anything else that only knows a string. ``reasoning_effort``
    reaches OpenAI reasoning models only; elsewhere it is dropped rather than
    raised, so callers can pass it without branching on which model they named.
    """
    if model.startswith("claude"):
        return AnthropicClient(model, **kwargs)
    return OpenAIClient(model, reasoning_effort=reasoning_effort, **kwargs)


__all__ = [
    "AnthropicClient",
    "LLMChunk",
    "LLMClient",
    "LLMUsage",
    "OpenAIClient",
    "PooledLLMClient",
    "TinkerClient",
    "as_chunk",
    "client_for",
    "is_retryable",
    "join_chunks",
    "retry_transient",
]
