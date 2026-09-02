"""OpenAI benchmark model."""

from __future__ import annotations

from benchmarks.eval import model
from benchmarks.eval.types import Model


@model("openai")
class OpenAIModel(Model):
    provider = "openai"

    def __init__(self, name: str = "gpt-5-mini", **kwargs) -> None:
        from openai import OpenAI

        self.name = name
        self.reasoning_effort = kwargs.pop("reasoning_effort", None)
        self._client = OpenAI(**kwargs)
        self._usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        request_kwargs = dict(kwargs)
        effort = request_kwargs.pop("reasoning_effort", self.reasoning_effort)
        if effort is not None:
            request_kwargs["reasoning_effort"] = effort
        response = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            stream=False,
            **request_kwargs,
        )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        self._usage = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            # A subset of output_tokens, not an addition to it.
            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        }
        return response.choices[0].message.content or ""

    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def close(self) -> None:
        self._client.close()


__all__ = ["OpenAIModel"]
