"""Tinker sampling client."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm import LLMChunk, LLMClient, retry_transient


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
        self._renderer_lock = threading.Lock()
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

        prompt, renderer_stop = self._render_prompt(messages)
        stop = self.stop
        if stop is None:
            stop = renderer_stop

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
        text = self._parse_tokens(tokens)
        usage = LLMUsage(
            input_tokens=self._token_count(prompt),
            output_tokens=self._token_count(tokens),
        )
        self.last_usage = usage
        return text, usage

    def _render_prompt(self, messages: list[dict[str, str]]) -> tuple[object, object]:
        # Cookbook renderers wrap a Rust tokenizer that cannot be borrowed by
        # concurrent Python threads. Sampling itself remains concurrent.
        with self._renderer_lock:
            prompt = self.renderer.build_generation_prompt(messages)
            stop = (
                self.renderer.get_stop_sequences()
                if hasattr(self.renderer, "get_stop_sequences")
                else None
            )
        return prompt, stop

    def _parse_tokens(self, tokens: object) -> str:
        with self._renderer_lock:
            message = self.renderer.parse_response(tokens)
        return self._message_text(message)

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
        if isinstance(message, list):
            parts = []
            for block in message:
                if isinstance(block, dict):
                    if block.get("type") in ("text", "output_text"):
                        parts.append(str(block.get("text", "")))
                    elif "content" in block:
                        parts.append(TinkerClient._message_text(block))
                    continue
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text))
            return "".join(parts)
        if isinstance(message, dict):
            content = message.get("content", "")
            return (
                TinkerClient._message_text(content)
                if isinstance(content, (list, tuple, dict))
                else str(content)
            )
        content = getattr(message, "content", None)
        if content is not None:
            return (
                TinkerClient._message_text(content)
                if isinstance(content, (list, tuple, dict))
                else str(content)
            )
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


__all__ = ["TinkerClient"]
