from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

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
    """Token counts from a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(metaclass=abc.ABCMeta):
    last_usage: LLMUsage | None = None
    thread_safe: bool = False

    @abc.abstractmethod
    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        """Send messages and return the full response."""

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        """Yield response token-by-token. Override for real streaming.

        Default falls back to chat() and yields the whole thing at once.
        """
        yield self.chat(messages, *args, **kwargs)

    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        """Return response text and usage for one request.

        The default adapter preserves the existing ``stream`` / ``last_usage``
        contract. Shared schedulers should guard this method for clients that
        do not override it, because ``last_usage`` is mutable client state.
        """
        text = "".join(self.stream(messages, *args, **kwargs))
        return text, self.last_usage or LLMUsage()


class OpenAIClient(LLMClient):
    """OpenAI-compatible client. Requires `pip install openai`."""

    thread_safe = True

    def __init__(self, model: str = "gpt-4o", **client_kwargs) -> None:
        from openai import OpenAI

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    @retry_transient
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        request_kwargs = {}
        if kwargs.get("timeout") is not None:
            request_kwargs["timeout"] = kwargs["timeout"]
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **request_kwargs,
        )
        usage = LLMUsage()
        if resp.usage:
            usage = LLMUsage(
                input_tokens=resp.usage.prompt_tokens or 0,
                output_tokens=resp.usage.completion_tokens or 0,
            )
            self.last_usage = usage
        return resp.choices[0].message.content or "", usage

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        # Buffer until the stream is fully consumed before yielding any
        # tokens, so tenacity can safely retry transient mid-stream
        # drops without double-emitting partial output. Real-time
        # streaming is sacrificed for correctness on retry.
        yield from self.collect_stream(messages)

    @retry_transient
    def collect_stream(self, messages: list[dict[str, str]]) -> list[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: list[str] = []
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                self.last_usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
        return chunks


class AnthropicClient(LLMClient):
    """Anthropic client. Requires `pip install anthropic`."""

    thread_safe = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        **client_kwargs,
    ) -> None:
        import anthropic

        self.client = anthropic.Anthropic(**client_kwargs)
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

    def chat(self, messages: list[dict[str, str]], *args, **kwargs) -> str:
        text, _usage = self.completion(messages, *args, **kwargs)
        return text

    @retry_transient
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
        system, chat_msgs = self.split_messages(messages)
        request_kwargs = {}
        if kwargs.get("timeout") is not None:
            request_kwargs["timeout"] = kwargs["timeout"]
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        for key in ("temperature", "top_p"):
            if kwargs.get(key) is not None:
                request_kwargs[key] = kwargs[key]
        if kwargs.get("stop") is not None:
            request_kwargs["stop_sequences"] = kwargs["stop"]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_msgs,
            **request_kwargs,
        )
        usage = LLMUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self.last_usage = usage
        return resp.content[0].text, usage

    def stream(self, messages: list[dict[str, str]], *args, **kwargs) -> Iterator[str]:
        yield from self.collect_stream(messages)

    @retry_transient
    def collect_stream(self, messages: list[dict[str, str]]) -> list[str]:
        system, chat_msgs = self.split_messages(messages)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=chat_msgs,
        ) as s:
            chunks = list(s.text_stream)
            msg = s.get_final_message()
            self.last_usage = LLMUsage(
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
            )
            return chunks


class TinkerClient(LLMClient):
    """Tinker sampling client. Requires `pip install tinker tinker-cookbook`.

    Tinker exposes model sampling over token prompts, so this adapter uses a
    Tinker cookbook renderer to convert chat messages to tokens and parse the
    sampled tokens back into assistant text.
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
            renderer_obj = renderers.get_renderer(
                renderer, sampling_client.get_tokenizer()
            )

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

    @retry_transient
    def completion(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> tuple[str, LLMUsage]:
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


__all__ = [
    "AnthropicClient",
    "LLMClient",
    "LLMUsage",
    "OpenAIClient",
    "TinkerClient",
    "is_retryable",
    "retry_transient",
]
